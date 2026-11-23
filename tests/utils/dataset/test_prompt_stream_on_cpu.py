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
"""The prompt stream: which prompt and latent geometry each global slot of a critic update takes.

The reference is the PDMD code's ``SampleStream`` and ``stream_seed``, copied into ``_prompt_stream_fixtures``.
"""

import pytest
import torch

from tests.utils.dataset._prompt_stream_fixtures import PdmdSampleStream, pdmd_stream_seed, pdmd_update_kind
from verl_omni.utils.dataset.prompt_stream import PromptStream, stream_seed

SIZE, GLOBAL_BATCH, SEED = 10, 4, 1234
BUCKETS = [(4, 6, 1, 4, 4, 3), (8, 6, 2, 4, 4, 3), (4, 12, 1, 4, 4, 3)]


@pytest.mark.parametrize("critic_steps", [5, 1])
def test_every_critic_slot_of_three_epochs_takes_pdmds_prompt_and_bucket(critic_steps):
    stream = PromptStream(SIZE, GLOBAL_BATCH, critic_steps, BUCKETS, SEED)
    reference = PdmdSampleStream(SIZE, GLOBAL_BATCH, critic_steps, BUCKETS, SEED)

    taken = []  # (update, slot, prompt index, bucket) in the order the critic updates take them
    update = 0
    while len(taken) < 3 * SIZE:
        if pdmd_update_kind(update, critic_steps) == "critic":
            for slot in range(GLOBAL_BATCH):
                taken.append((update, slot, stream.index(update, slot), tuple(stream.bucket(update, slot))))
        update += 1

    expected = [(u, s, reference.index(u, s), tuple(reference.bucket(u, s))) for u, s, _, _ in taken]
    assert taken == expected
    for epoch in range(3):  # without replacement: each epoch takes every prompt once
        assert sorted(index for _, _, index, _ in taken[epoch * SIZE : (epoch + 1) * SIZE]) == list(range(SIZE))
    assert len({bucket for *_, bucket in taken}) == len(BUCKETS)
    assert [index for _, _, index, _ in taken[:SIZE]] != [index for _, _, index, _ in taken[SIZE : 2 * SIZE]]


@pytest.mark.parametrize("critic_steps", [5, 1])
def test_student_updates_take_no_prompts(critic_steps):
    stream = PromptStream(SIZE, GLOBAL_BATCH, critic_steps, BUCKETS, SEED)

    for update in range(4 * (critic_steps + 1)):
        assert stream.is_critic(update) == (pdmd_update_kind(update, critic_steps) == "critic")
        if not stream.is_critic(update):
            with pytest.raises(ValueError, match="student"):
                stream.index(update, 0)
            with pytest.raises(ValueError, match="student"):
                stream.bucket(update, 0)


def test_the_stream_does_not_depend_on_query_order_or_on_global_rng():
    stream, fresh = (
        PromptStream(SIZE, GLOBAL_BATCH, 5, BUCKETS, SEED),
        PromptStream(SIZE, GLOBAL_BATCH, 5, BUCKETS, SEED),
    )
    keys = [(update, slot) for update in range(12) if stream.is_critic(update) for slot in range(GLOBAL_BATCH)]

    forward = [(stream.index(*key), stream.bucket(*key)) for key in keys]
    torch.manual_seed(99)
    backward = [(fresh.index(*key), fresh.bucket(*key)) for key in reversed(keys)][::-1]
    assert forward == backward


def test_stream_seed_is_pdmds():
    for parts in [(0, "exit", 3), (SEED, "permutation", 2), (7, "noise", 11, 5)]:
        assert stream_seed(*parts) == pdmd_stream_seed(*parts)


@pytest.mark.parametrize(
    "args, message",
    [
        ((0, 4, 5, BUCKETS, 0), "size"),
        ((10, 0, 5, BUCKETS, 0), "global_batch"),
        ((10, 4, 0, BUCKETS, 0), "critic_steps"),
        ((10, 4, 5, [], 0), "bucket"),
    ],
)
def test_an_empty_corpus_batch_schedule_or_bucket_list_is_refused(args, message):
    with pytest.raises(ValueError, match=message):
        PromptStream(*args)


def test_a_slot_outside_the_global_batch_is_refused():
    stream = PromptStream(SIZE, GLOBAL_BATCH, 5, BUCKETS, SEED)
    for slot in (-1, GLOBAL_BATCH):
        with pytest.raises(ValueError, match="slot"):
            stream.index(0, slot)
