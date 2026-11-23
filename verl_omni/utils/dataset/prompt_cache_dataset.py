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
"""Precomputed prompt features, so that a trainer that only needs text conditioning loads no text encoder, batched
per critic update by the prompt stream.

The on-disk layout is the PDMD reference's prompt cache (https://github.com/ZeamoxWang/pdmd, ``data.py``): rank
manifests ``rank-*.jsonl`` whose rows name a safetensors ``shard`` and a ``key``; the shard holds the prompt
embeddings ``(tokens, features)`` under ``key`` and their token tags ``(tokens,)`` under ``key + "_tags"``.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open
from tensordict import TensorDict
from torch.utils.data import Dataset

from verl_omni.utils.dataset.prompt_stream import PromptStream

__all__ = ["PromptCacheDataset", "PromptCacheLoader", "collate_prompt_features"]


class PromptCacheDataset(Dataset):
    """One row per cached prompt, ordered by source row: its embeddings and token tags.

    Args:
        root: The cache directory.

    Raises:
        ValueError: The cache holds no row or a duplicate source row.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        rows = []
        for manifest in sorted(self.root.glob("rank-*.jsonl")):
            rows.extend(json.loads(line) for line in manifest.read_text().splitlines() if line)
        if not rows:
            raise ValueError(f"No cached prompts under {self.root}")
        rows.sort(key=lambda row: row["source_row"])
        if len({row["source_row"] for row in rows}) != len(rows):
            raise ValueError(f"The prompt cache under {self.root} holds a source row twice")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return ``prompt_embeds`` ``(tokens, features)`` and ``prompt_token_tags`` ``(tokens,)``."""
        row = self.rows[index]
        with safe_open(self.root / row["shard"], framework="pt", device="cpu") as shard:
            prompt_embeds = shard.get_tensor(row["key"])
            token_tags = shard.get_tensor(row["key"] + "_tags")
        if prompt_embeds.ndim != 2 or token_tags.shape != prompt_embeds.shape[:1]:
            raise ValueError(
                f"Cached prompt {row['key']} has embeddings {tuple(prompt_embeds.shape)} and tags "
                f"{tuple(token_tags.shape)}; expected (tokens, features) and (tokens,)"
            )
        return {"prompt_embeds": prompt_embeds, "prompt_token_tags": token_tags}


class PromptCacheLoader:
    """One global batch per critic update, from ``first_update`` on, without end; the stream picks each slot's
    prompt and latent geometry.

    Each batch holds :func:`collate_prompt_features`'s fields plus ``prompt_index`` ``(B,)``, each slot's corpus
    index, and ``global_sample`` ``(B,)``, each slot's index in the global batch.

    Args:
        dataset: The cached prompts.
        stream: The prompt stream over ``dataset``; its buckets are ``latent_meta`` tuples.
        first_update: The first update whose prompts the loader yields; student updates yield none.

    Raises:
        ValueError: The stream's corpus size differs from the dataset's, or its buckets differ in length.
    """

    def __init__(self, dataset: PromptCacheDataset, stream: PromptStream, first_update: int = 0):
        if stream.size != len(dataset):
            raise ValueError(f"The prompt stream covers {stream.size} prompts; the cache holds {len(dataset)}")
        if len({len(bucket) for bucket in stream.buckets}) != 1:
            raise ValueError(f"latent_meta_buckets needs geometries of one length, got {stream.buckets}")
        self.dataset, self.stream, self.first_update = dataset, stream, first_update

    def batch(self, update: int) -> TensorDict:
        """The global batch of critic ``update``."""
        slots = range(self.stream.global_batch)
        indices = [self.stream.index(update, slot) for slot in slots]
        rows = [
            self.dataset[index] | {"latent_meta": torch.tensor(list(self.stream.bucket(update, slot)))}
            for slot, index in zip(slots, indices, strict=True)
        ]
        batch = collate_prompt_features(rows)
        batch["prompt_index"] = torch.tensor(indices)
        batch["global_sample"] = torch.arange(self.stream.global_batch)
        return batch

    def __iter__(self) -> Iterator[TensorDict]:
        update = self.first_update
        while True:
            if self.stream.is_critic(update):
                yield self.batch(update)
            update += 1


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
