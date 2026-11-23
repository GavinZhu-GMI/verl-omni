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
"""Precomputed prompt features read from the PDMD prompt-cache layout, batched with a padding mask."""

import json

import pytest
import torch
from safetensors.torch import save_file

from verl_omni.utils.dataset.prompt_cache_dataset import PromptCacheDataset, collate_prompt_features

BUCKETS = [[4, 6, 1, 4, 4, 3], [8, 6, 2, 4, 4, 3]]


def write_cache(root, lengths_by_rank, features=5):
    """One manifest and one shard per rank; source rows interleave across ranks, as the cache writer assigns them."""
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


def test_rows_follow_the_source_order_and_carry_the_cached_tensors(tmp_path):
    expected = write_cache(tmp_path, [[3, 2], [4, 1]])
    dataset = PromptCacheDataset(tmp_path, BUCKETS, seed=0)

    assert len(dataset) == 4
    for index, (embeds, tags) in enumerate(expected):
        row = dataset[index]
        torch.testing.assert_close(row["prompt_embeds"], embeds, rtol=0, atol=0)
        assert torch.equal(row["prompt_token_tags"], tags)
        assert row["latent_meta"].tolist() in BUCKETS


def test_a_row_keeps_its_geometry_and_the_seed_spreads_rows_over_buckets(tmp_path):
    write_cache(tmp_path, [[2] * 32])
    first, again = PromptCacheDataset(tmp_path, BUCKETS, seed=7), PromptCacheDataset(tmp_path, BUCKETS, seed=7)
    other = PromptCacheDataset(tmp_path, BUCKETS, seed=8)
    geometries = [first[index]["latent_meta"].tolist() for index in range(32)]

    assert geometries == [again[index]["latent_meta"].tolist() for index in range(32)]
    assert geometries == [first[index]["latent_meta"].tolist() for index in range(32)]
    assert {tuple(geometry) for geometry in geometries} == {tuple(bucket) for bucket in BUCKETS}
    assert geometries != [other[index]["latent_meta"].tolist() for index in range(32)]


def test_collate_pads_prompts_and_masks_the_padding(tmp_path):
    write_cache(tmp_path, [[3, 1]])
    dataset = PromptCacheDataset(tmp_path, BUCKETS[:1], seed=0)

    batch = collate_prompt_features([dataset[0], dataset[1]])

    assert batch.batch_size == torch.Size([2])
    assert batch["prompt_embeds"].shape == (2, 3, 5) and batch["prompt_embeds"].dtype == torch.bfloat16
    assert batch["prompt_embeds_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    torch.testing.assert_close(batch["prompt_embeds"][1, :1], dataset[1]["prompt_embeds"], rtol=0, atol=0)
    assert torch.count_nonzero(batch["prompt_embeds"][1, 1:]) == 0
    assert batch["prompt_token_tags"][1].tolist() == [dataset[1]["prompt_token_tags"].item(), 0, 0]
    assert batch["latent_meta"].tolist() == [BUCKETS[0], BUCKETS[0]]


def test_an_empty_cache_or_a_duplicate_row_is_refused(tmp_path):
    with pytest.raises(ValueError, match="No cached prompts"):
        PromptCacheDataset(tmp_path, BUCKETS, seed=0)
    write_cache(tmp_path, [[2]])
    (tmp_path / "rank-0001.jsonl").write_text((tmp_path / "rank-0000.jsonl").read_text())
    with pytest.raises(ValueError, match="source row twice"):
        PromptCacheDataset(tmp_path, BUCKETS, seed=0)
