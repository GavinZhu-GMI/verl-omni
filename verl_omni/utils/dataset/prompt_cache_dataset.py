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
"""Precomputed prompt features, so that a trainer that only needs text conditioning loads no text encoder.

The on-disk layout is the PDMD reference's prompt cache (https://github.com/ZeamoxWang/pdmd, ``data.py``): rank
manifests ``rank-*.jsonl`` whose rows name a safetensors ``shard`` and a ``key``; the shard holds the prompt
embeddings ``(tokens, features)`` under ``key`` and their token tags ``(tokens,)`` under ``key + "_tags"``.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from tensordict import TensorDict
from torch.utils.data import Dataset

__all__ = ["PromptCacheDataset", "collate_prompt_features"]


def _bucket_seed(seed: int, index: int) -> int:
    """A stable 63-bit seed for one row's latent geometry, independent of Python's hash salt."""
    digest = hashlib.sha256(repr((seed, "bucket", index)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


class PromptCacheDataset(Dataset):
    """One row per cached prompt, ordered by source row: its embeddings, token tags and a latent geometry.

    Each row draws its latent geometry from ``latent_meta_buckets`` with a generator seeded by ``(seed, index)``,
    so a row keeps its geometry across epochs and machines.

    Args:
        root: The cache directory.
        latent_meta_buckets: The latent geometries a row may draw, each the pipeline's ``latent_meta`` tuple.
        seed: The seed of the geometry draw.

    Raises:
        ValueError: The cache holds no row or a duplicate source row, or no bucket is given, or the buckets differ
            in length.
    """

    def __init__(self, root: str | Path, latent_meta_buckets: Sequence[Sequence[int]], seed: int):
        self.root = Path(root).expanduser()
        rows = []
        for manifest in sorted(self.root.glob("rank-*.jsonl")):
            rows.extend(json.loads(line) for line in manifest.read_text().splitlines() if line)
        if not rows:
            raise ValueError(f"No cached prompts under {self.root}")
        rows.sort(key=lambda row: row["source_row"])
        if len({row["source_row"] for row in rows}) != len(rows):
            raise ValueError(f"The prompt cache under {self.root} holds a source row twice")
        if not latent_meta_buckets or len({len(bucket) for bucket in latent_meta_buckets}) != 1:
            raise ValueError(
                f"latent_meta_buckets needs one or more geometries of one length, got {latent_meta_buckets}"
            )
        self.rows = rows
        self.buckets = torch.tensor([list(bucket) for bucket in latent_meta_buckets], dtype=torch.long)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return ``prompt_embeds`` ``(tokens, features)``, ``prompt_token_tags`` ``(tokens,)`` and ``latent_meta``."""
        row = self.rows[index]
        # one handle per read: DataLoader workers fork, and a handle opened before the fork is not theirs to share
        with safe_open(self.root / row["shard"], framework="pt", device="cpu") as shard:
            prompt_embeds = shard.get_tensor(row["key"])
            token_tags = shard.get_tensor(row["key"] + "_tags")
        if prompt_embeds.ndim != 2 or token_tags.shape != prompt_embeds.shape[:1]:
            raise ValueError(
                f"Cached prompt {row['key']} has embeddings {tuple(prompt_embeds.shape)} and tags "
                f"{tuple(token_tags.shape)}; expected (tokens, features) and (tokens,)"
            )
        generator = torch.Generator().manual_seed(_bucket_seed(self.seed, index))
        bucket = int(torch.randint(len(self.buckets), (1,), generator=generator))
        return {"prompt_embeds": prompt_embeds, "prompt_token_tags": token_tags, "latent_meta": self.buckets[bucket]}


def collate_prompt_features(rows: list[dict[str, torch.Tensor]]) -> TensorDict:
    """Pad each row's prompt to the longest one; ``prompt_embeds_mask`` marks the real tokens.

    Returns:
        ``prompt_embeds`` ``(B, tokens, features)``, ``prompt_embeds_mask`` and ``prompt_token_tags`` ``(B, tokens)``,
        ``latent_meta`` ``(B, len(latent_meta))``.
    """
    lengths = [row["prompt_embeds"].shape[0] for row in rows]
    longest, features = max(lengths), rows[0]["prompt_embeds"].shape[1]
    embeds = rows[0]["prompt_embeds"].new_zeros(len(rows), longest, features)
    mask = torch.zeros(len(rows), longest, dtype=torch.long)
    tags = rows[0]["prompt_token_tags"].new_zeros(len(rows), longest)
    for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
        embeds[index, :length] = row["prompt_embeds"]
        mask[index, :length] = 1
        tags[index, :length] = row["prompt_token_tags"]
    return TensorDict(
        {
            "prompt_embeds": embeds,
            "prompt_embeds_mask": mask,
            "prompt_token_tags": tags,
            "latent_meta": torch.stack([row["latent_meta"] for row in rows]),
        },
        batch_size=[len(rows)],
    )
