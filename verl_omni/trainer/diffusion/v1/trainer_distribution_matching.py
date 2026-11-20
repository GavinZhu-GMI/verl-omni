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
"""The v1 ``distribution_matching`` trainer: DMD2 critic and student updates, one update per global step."""

import math
from collections.abc import Iterable, Iterator
from numbers import Real
from typing import Any

from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.trainer.diffusion.v1.trainer_base import register_diffusion_trainer
from verl_omni.workers.config import DiffusionDMDConfig


def _nonfinite_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return the metrics holding a nonfinite number; each value is a number or a list of numbers.

    Raises:
        TypeError: A value is neither, so its finiteness cannot be checked.
    """
    nonfinite = {}
    for key, values in metrics.items():
        for value in values if isinstance(values, list) else [values]:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Metric {key!r} holds a {type(value).__name__}; expected numbers")
            if not math.isfinite(value):
                nonfinite[key] = values
    return nonfinite


@register_diffusion_trainer("distribution_matching")
class DistributionMatchingDiffusionTrainerV1:
    """Run ``fake_update_ratio`` critic updates, then one student update on the last critic update's prompts.

    A nonfinite number in an update's metrics stops training with ``FloatingPointError``, as the reference
    implementation does (https://github.com/ZeamoxWang/pdmd, ``training/train.py``). On a nonfinite gradient norm
    the engine has skipped that optimizer step, so counting it would break the critic-to-student ratio.
    """

    actor_rollout_wg: Any  # the worker group of the DMD engine, assigned before fit()
    train_dataloader: Iterable[TensorDict]  # prompt batches, one per critic update, assigned before fit()

    def __init__(self, config):
        self.config = config
        self.dmd_config = omega_conf_to_dataclass(config.dmd, DiffusionDMDConfig)
        self.total_training_steps = config.trainer.total_training_steps
        self.global_steps = 0

    def init(self):
        """Build the worker group and the dataloader."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot build its worker group yet: no training worker passes dmd_config to the "
            "DMD engine."
        )

    def _prompt_batches(self) -> Iterator[TensorDict]:
        """Yield prompt batches without end, one pass over ``train_dataloader`` per epoch."""
        while True:
            empty = True
            for batch in self.train_dataloader:
                empty = False
                yield batch
            if empty:
                raise ValueError("train_dataloader yields no prompt batch")

    def fit(self):
        """Train until ``total_training_steps`` updates have run."""
        stages = self.dmd_config.training_stages()
        batches = self._prompt_batches()
        batch = None
        while self.global_steps < self.total_training_steps:
            position = self.global_steps % len(stages)
            stage = stages[position]
            self.global_steps += 1
            if stage == "fake_score":
                batch = next(batches)
            tu.assign_non_tensor(batch, dmd_stage=stage)
            output = self.actor_rollout_wg.update_actor(batch)
            nonfinite = _nonfinite_metrics(tu.get(output, "metrics"))
            if nonfinite:
                label = (
                    f"fake_score {position + 1}/{self.dmd_config.fake_update_ratio}" if stage == "fake_score" else stage
                )
                raise FloatingPointError(
                    f"Nonfinite metrics {nonfinite} at global_step={self.global_steps}, stage={label}"
                )
