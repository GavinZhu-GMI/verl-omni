# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The PDMD reference's prompt stream (https://github.com/ZeamoxWang/pdmd at 6b6e106: ``SampleStream`` and
``stream_seed`` from ``src/pdmd/data.py``, ``update_kind`` from ``src/pdmd/schedule.py``), copied unchanged but for
formatting and names, and a prompt cache in its on-disk layout.
"""

import hashlib
import json

import torch
from safetensors.torch import save_file


def pdmd_stream_seed(*parts):
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


class PdmdSampleStream:
    def __init__(self, size, global_batch, critic_steps, buckets, seed):
        self.size, self.global_batch = size, global_batch
        self.period, self.buckets, self.seed = critic_steps + 1, buckets, seed
        self._perm_epoch, self._perm = None, None

    def critic_updates_before(self, iteration):
        return iteration - iteration // self.period

    def position(self, iteration, slot):
        return self.critic_updates_before(iteration) * self.global_batch + slot

    def index(self, iteration, slot):
        position = self.position(iteration, slot)
        epoch, offset = divmod(position, self.size)
        if epoch != self._perm_epoch:
            g = torch.Generator().manual_seed(pdmd_stream_seed(self.seed, "permutation", epoch))
            self._perm, self._perm_epoch = torch.randperm(self.size, generator=g), epoch
        return int(self._perm[offset])

    def bucket(self, iteration, slot):
        g = torch.Generator().manual_seed(pdmd_stream_seed(self.seed, "bucket", self.position(iteration, slot)))
        return self.buckets[int(torch.randint(len(self.buckets), (1,), generator=g))]


def pdmd_update_kind(iteration, critic_steps):
    """``pdmd.schedule.update_kind``: ``critic_steps`` critic updates, then one student update."""
    return "student" if iteration % (critic_steps + 1) == critic_steps else "critic"


def write_cache(root, lengths_by_rank, features=5):
    """One manifest and one shard per rank; source rows interleave across ranks, as the cache writer assigns them.

    Returns:
        The ``(embeddings, tags)`` of each prompt, in source-row order.
    """
    expected = {}
    world = len(lengths_by_rank)
    for rank, lengths in enumerate(lengths_by_rank):
        tensors, rows = {}, []
        shard = f"rank-{rank:04d}-shard-000000.safetensors"
        for position, length in enumerate(lengths):
            source_row = position * world + rank
            key = f"sample_{source_row:08d}"
            tensors[key] = torch.randn(length, features).to(torch.bfloat16)
            tensors[key + "_tags"] = torch.arange(length, dtype=torch.int64) % 3
            rows.append({"source_row": source_row, "key": key, "shard": shard})
            expected[source_row] = (tensors[key], tensors[key + "_tags"])
        save_file(tensors, root / shard)
        (root / f"rank-{rank:04d}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return [expected[source_row] for source_row in sorted(expected)]
