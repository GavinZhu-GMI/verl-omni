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
"""Which prompt and latent geometry each global slot of a DMD critic update takes, as a pure function of the key.

It is adapted from the PDMD reference's ``SampleStream`` and ``stream_seed``
(https://github.com/ZeamoxWang/pdmd, ``src/pdmd/data.py``).
"""

import hashlib
from collections.abc import Sequence
from typing import Generic, TypeVar

import torch

__all__ = ["PromptStream", "stream_seed"]

Bucket = TypeVar("Bucket")


def stream_seed(*parts) -> int:
    """A stable 63-bit seed from named parts, independent of Python's hash salt."""
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


class PromptStream(Generic[Bucket]):
    """Prompts without replacement over a seeded permutation per epoch; student updates take none.

    Updates run in periods of ``critic_steps`` critic updates and then one student update. Critic update number c
    takes global slots ``c * global_batch .. c * global_batch + global_batch - 1`` of the stream. Epoch e is a
    permutation of the corpus seeded by ``stream_seed(seed, "permutation", e)``; the slot at stream position p draws
    its bucket with ``stream_seed(seed, "bucket", p)``. So the stream holds no state a resume must restore, and does
    not depend on the number of ranks.

    Args:
        size: The number of prompts in the corpus.
        global_batch: The prompts one critic update takes.
        critic_steps: The critic updates before each student update.
        buckets: The latent geometries a slot may draw.
        seed: The stream's seed.

    Raises:
        ValueError: ``size``, ``global_batch`` or ``critic_steps`` is below 1, or ``buckets`` is empty.
    """

    def __init__(self, size: int, global_batch: int, critic_steps: int, buckets: Sequence[Bucket], seed: int):
        for name, value in (("size", size), ("global_batch", global_batch), ("critic_steps", critic_steps)):
            if value < 1:
                raise ValueError(f"PromptStream needs {name} >= 1, got {value}")
        if not buckets:
            raise ValueError("PromptStream needs one or more buckets")
        self.size, self.global_batch, self.critic_steps = size, global_batch, critic_steps
        self.buckets, self.seed = list(buckets), seed
        self._permutation_epoch: int | None = None
        self._permutation = torch.empty(0, dtype=torch.long)

    def is_critic(self, update: int) -> bool:
        """Whether 0-based ``update`` is a critic update; the last update of each period is a student update."""
        return update % (self.critic_steps + 1) != self.critic_steps

    def _position(self, update: int, slot: int) -> int:
        if not self.is_critic(update):
            raise ValueError(f"Update {update} is a student update: it takes no prompts")
        if not 0 <= slot < self.global_batch:
            raise ValueError(f"slot {slot} is outside the global batch of {self.global_batch}")
        critic_updates_before = update - update // (self.critic_steps + 1)
        return critic_updates_before * self.global_batch + slot

    def index(self, update: int, slot: int) -> int:
        """The corpus index of the prompt that global ``slot`` of critic ``update`` takes.

        Raises:
            ValueError: ``update`` is a student update, or ``slot`` is outside the global batch.
        """
        epoch, offset = divmod(self._position(update, slot), self.size)
        if epoch != self._permutation_epoch:
            generator = torch.Generator().manual_seed(stream_seed(self.seed, "permutation", epoch))
            self._permutation = torch.randperm(self.size, generator=generator)
            self._permutation_epoch = epoch
        return int(self._permutation[offset])

    def bucket(self, update: int, slot: int) -> Bucket:
        """The latent geometry that global ``slot`` of critic ``update`` draws.

        Raises:
            ValueError: ``update`` is a student update, or ``slot`` is outside the global batch.
        """
        generator = torch.Generator().manual_seed(stream_seed(self.seed, "bucket", self._position(update, slot)))
        return self.buckets[int(torch.randint(len(self.buckets), (1,), generator=generator))]
