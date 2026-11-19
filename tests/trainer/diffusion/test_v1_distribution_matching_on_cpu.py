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
"""CPU tests for the v1 ``distribution_matching`` trainer loop, driven through a fake worker group."""

import math
import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from verl.utils import tensordict_utils as tu

import verl_omni

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")


class FakeWorkerGroup:
    """Record each update's stage and answer with the scripted metrics of that global step."""

    def __init__(self, metrics: dict[int, dict[str, list[float]]]):
        self.metrics = metrics
        self.stages: list[str] = []
        self.batch_ids: list[int] = []

    def update_actor(self, data):
        self.stages.append(tu.get(data, "dmd_stage"))
        self.batch_ids.append(int(data["prompt_ids"][0, 0]))
        metrics = {"grad_norm": [1.0], "loss": [0.5]} | self.metrics.get(len(self.stages), {})
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics})


def make_trainer(metrics: dict[int, dict[str, list[float]]], total_steps: int = 12, num_batches: int = 12):
    from verl_omni.trainer.diffusion.v1 import get_diffusion_trainer_cls

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                "trainer.v1.trainer_mode=distribution_matching",
                f"trainer.total_training_steps={total_steps}",
                "actor_rollout_ref.model.algorithm=dmd2",
                "dmd.fake_update_ratio=5",
            ],
        )
    trainer = get_diffusion_trainer_cls("distribution_matching")(config=config)
    trainer.actor_rollout_wg = FakeWorkerGroup(metrics)
    trainer.train_dataloader = [tu.get_tensordict({"prompt_ids": torch.full((2, 3), i)}) for i in range(num_batches)]
    return trainer


@pytest.mark.parametrize(
    "global_step, grad_norm, stage_label",
    [(9, math.nan, "fake_score 3/5"), (6, math.inf, "student")],
)
def test_nonfinite_grad_norm_aborts_at_the_stage_that_produced_it(global_step, grad_norm, stage_label):
    trainer = make_trainer({global_step: {"grad_norm": [grad_norm]}})

    with pytest.raises(FloatingPointError) as excinfo:
        trainer.fit()

    message = str(excinfo.value)
    assert f"global_step={global_step}" in message
    assert f"stage={stage_label}" in message
    assert str(grad_norm) in message
    assert trainer.actor_rollout_wg.stages == ((["fake_score"] * 5 + ["student"]) * 2)[:global_step]


def test_nonfinite_loss_aborts_although_the_grad_norm_is_finite():
    """A nonfinite loss aborts the run although the engine stepped on a finite gradient norm."""
    trainer = make_trainer({4: {"grad_norm": [0.3], "loss": [0.5, math.nan]}})

    with pytest.raises(FloatingPointError) as excinfo:
        trainer.fit()

    message = str(excinfo.value)
    assert "global_step=4" in message
    assert "stage=fake_score 4/5" in message
    assert "loss" in message and "nan" in message
    assert len(trainer.actor_rollout_wg.stages) == 4


def test_fake_score_updates_start_a_new_epoch_when_the_prompts_run_out():
    """Each fake-score update takes the next prompt batch, across epochs; a student update reuses the last one."""
    trainer = make_trainer({}, total_steps=12, num_batches=2)

    trainer.fit()

    assert trainer.actor_rollout_wg.batch_ids == [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]


def test_a_metric_that_is_not_a_number_fails_instead_of_skipping_the_check():
    trainer = make_trainer({2: {"loss": [torch.tensor(math.nan)]}})

    with pytest.raises(TypeError, match="Metric 'loss' holds a Tensor"):
        trainer.fit()
