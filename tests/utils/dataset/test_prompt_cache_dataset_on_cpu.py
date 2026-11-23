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
"""Precomputed prompt features in the PDMD prompt-cache layout, batched per critic update by the prompt stream."""

import pytest
import torch

from tests.utils.dataset._prompt_stream_fixtures import PdmdSampleStream, write_cache
from verl_omni.utils.dataset.prompt_cache_dataset import PromptCacheDataset, PromptCacheLoader, collate_prompt_features
from verl_omni.utils.dataset.prompt_stream import PromptStream

BUCKETS = [[4, 6, 1, 4, 4, 3], [8, 6, 2, 4, 4, 3]]


def test_rows_follow_the_source_order_and_carry_the_cached_tensors(tmp_path):
    expected = write_cache(tmp_path, [[3, 2], [4, 1]])
    dataset = PromptCacheDataset(tmp_path)

    assert len(dataset) == 4
    for index, (embeds, tags) in enumerate(expected):
        row = dataset[index]
        torch.testing.assert_close(row["prompt_embeds"], embeds, rtol=0, atol=0)
        assert torch.equal(row["prompt_token_tags"], tags)


def test_collate_pads_prompts_and_masks_the_padding(tmp_path):
    write_cache(tmp_path, [[3, 1]])
    dataset = PromptCacheDataset(tmp_path)
    rows = [
        dataset[0] | {"latent_meta": torch.tensor(BUCKETS[0])},
        dataset[1] | {"latent_meta": torch.tensor(BUCKETS[1])},
    ]

    batch = collate_prompt_features(rows)

    assert batch.batch_size == torch.Size([2])
    assert batch["prompt_embeds"].shape == (2, 3, 5) and batch["prompt_embeds"].dtype == torch.bfloat16
    assert batch["prompt_embeds_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    torch.testing.assert_close(batch["prompt_embeds"][1, :1], dataset[1]["prompt_embeds"], rtol=0, atol=0)
    assert torch.count_nonzero(batch["prompt_embeds"][1, 1:]) == 0
    assert batch["prompt_token_tags"][1].tolist() == [dataset[1]["prompt_token_tags"].item(), 0, 0]
    assert batch["latent_meta"].tolist() == BUCKETS


def test_an_empty_cache_or_a_duplicate_row_is_refused(tmp_path):
    with pytest.raises(ValueError, match="No cached prompts"):
        PromptCacheDataset(tmp_path)
    write_cache(tmp_path, [[2]])
    (tmp_path / "rank-0001.jsonl").write_text((tmp_path / "rank-0000.jsonl").read_text())
    with pytest.raises(ValueError, match="source row twice"):
        PromptCacheDataset(tmp_path)


def test_the_loader_yields_each_critic_update_the_prompts_and_buckets_of_pdmds_stream(tmp_path):
    expected = write_cache(tmp_path, [[2, 3, 1, 4, 2], [1, 2, 3, 2]])  # 9 prompts of differing lengths
    dataset, seed = PromptCacheDataset(tmp_path), 11
    reference = PdmdSampleStream(len(dataset), 2, 3, BUCKETS, seed)
    loader = PromptCacheLoader(dataset, PromptStream(len(dataset), 2, 3, BUCKETS, seed))

    batches = iter(loader)
    critic_updates = [update for update in range(16) if update % 4 != 3]  # 12 critic updates: 24 slots, ~3 epochs
    for update in critic_updates:
        batch = next(batches)
        assert batch["prompt_index"].tolist() == [reference.index(update, slot) for slot in range(2)]
        assert batch["latent_meta"].tolist() == [reference.bucket(update, slot) for slot in range(2)]
        assert batch["global_sample"].tolist() == [0, 1]
        for slot, index in enumerate(batch["prompt_index"].tolist()):
            length = expected[index][0].shape[0]
            torch.testing.assert_close(batch["prompt_embeds"][slot, :length], expected[index][0], rtol=0, atol=0)
            assert batch["prompt_embeds_mask"][slot].sum() == length


def test_a_loader_started_at_an_update_yields_the_batches_a_fresh_one_yields_from_there(tmp_path):
    write_cache(tmp_path, [[2] * 7])
    dataset = PromptCacheDataset(tmp_path)
    stream = PromptStream(len(dataset), 2, 5, BUCKETS, 3)
    straight = iter(PromptCacheLoader(dataset, stream))
    straight_batches = [next(straight) for _ in range(10)]  # critic updates 0-4, 6-10

    for first_update, skipped in [(3, 3), (5, 5), (6, 5), (7, 6)]:  # a student update (5) yields no batch
        resumed = iter(PromptCacheLoader(dataset, stream, first_update=first_update))
        for want in straight_batches[skipped:]:
            got = next(resumed)
            assert got["prompt_index"].tolist() == want["prompt_index"].tolist()
            assert got["latent_meta"].tolist() == want["latent_meta"].tolist()


def test_buckets_of_differing_lengths_are_refused(tmp_path):
    write_cache(tmp_path, [[2, 2]])
    dataset = PromptCacheDataset(tmp_path)
    with pytest.raises(ValueError, match="one length"):
        PromptCacheLoader(dataset, PromptStream(len(dataset), 2, 5, [[4, 6, 1], [4, 6]], 0))
