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
"""The per-step metric log of the v1 ``distribution_matching`` trainer."""

import math

import pytest

from tests.trainer.diffusion.test_v1_distribution_matching_on_cpu import make_trainer


class RecordingLogger:
    def __init__(self):
        self.steps = []

    def log(self, data, step):
        self.steps.append((step, data))


def fit_and_log(metrics, total_steps=12):
    trainer = make_trainer(metrics, total_steps=total_steps)
    trainer.logger = RecordingLogger()
    trainer.fit()
    return trainer.logger.steps


def test_each_step_logs_its_stage_and_the_mean_of_its_metrics():
    steps = fit_and_log({6: {"dmd/critic_teacher_x0_gap": [2.0, 4.0]}})

    assert [step for step, _ in steps] == list(range(1, 13))
    assert [data["dmd/student_step"] for _, data in steps] == ([0.0] * 5 + [1.0]) * 2
    assert steps[5][1]["dmd/critic_teacher_x0_gap"] == 3.0
    assert steps[0][1]["loss"] == 0.5


def test_the_step_that_aborts_is_logged_first():
    trainer = make_trainer({3: {"grad_norm": [math.nan]}})
    trainer.logger = RecordingLogger()
    with pytest.raises(FloatingPointError):
        trainer.fit()
    assert [step for step, _ in trainer.logger.steps] == [1, 2, 3]
    assert math.isnan(trainer.logger.steps[-1][1]["grad_norm"])
