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
"""The fake score's distance from the teacher at the student's score queries, over every modality."""

import pytest
import torch

from tests.pipelines.test_dmd2_program_on_cpu import loss_inputs, make_program
from verl_omni.pipelines.dmd2.program import critic_teacher_x0_metrics


def outputs(fake_video, teacher_video, fake_audio, teacher_audio):
    return {
        "video": {"fake_x0": fake_video, "teacher_x0": teacher_video},
        "audio": {"fake_x0": fake_audio, "teacher_x0": teacher_audio},
    }


def test_the_gap_is_the_per_sample_norm_over_all_modalities_averaged_over_samples():
    """Sample 0 differs by (3, 4) in video and 12 in audio: norm 13. Sample 1 differs by 5 in audio only: norm 5."""
    teacher_video, teacher_audio = torch.zeros(2, 2, 3), torch.zeros(2, 4)
    fake_video, fake_audio = teacher_video.clone(), teacher_audio.clone()
    fake_video[0, 0, 0], fake_video[0, 1, 2], fake_audio[0, 3] = 3.0, 4.0, 12.0
    fake_audio[1, 0] = 5.0

    metrics = critic_teacher_x0_metrics(outputs(fake_video, teacher_video, fake_audio, teacher_audio))

    assert metrics["dmd/critic_teacher_x0_gap"].item() == pytest.approx((13 + 5) / 2)
    # 20 elements, 4 of them differ
    assert metrics["dmd/critic_teacher_x0_equal"].item() == pytest.approx(16 / 20)


def test_the_gap_is_computed_in_fp32():
    """bf16 estimates give an FP32 gap; the norm of (300, 400) is exactly 500."""
    teacher = torch.zeros(1, 2, dtype=torch.bfloat16)
    fake = torch.tensor([[300.0, 400.0]], dtype=torch.bfloat16)
    metrics = critic_teacher_x0_metrics(outputs(fake, teacher, teacher[:, :1], teacher[:, :1]))
    assert metrics["dmd/critic_teacher_x0_gap"].dtype == torch.float32
    assert metrics["dmd/critic_teacher_x0_gap"].item() == 500.0
    assert metrics["dmd/critic_teacher_x0_equal"].item() == pytest.approx(1 / 3)


def test_identical_estimates_give_zero_gap_and_full_equality():
    x0 = torch.randn(3, 5)
    metrics = critic_teacher_x0_metrics(outputs(x0, x0.clone(), x0[:, :2], x0[:, :2].clone()))
    assert metrics["dmd/critic_teacher_x0_gap"].item() == 0.0
    assert metrics["dmd/critic_teacher_x0_equal"].item() == 1.0


@pytest.mark.parametrize("stage", ["student", "fake_score"])
def test_only_the_student_update_reports_the_gap_of_its_own_score_queries(stage):
    """The student update's gap comes from the fake and teacher estimates its own loss used."""
    program, _, initial, _ = make_program(("video", "audio"))
    reference, _, _, _ = make_program(("video", "audio"))
    keys = program.keys(0, tuple(range(next(iter(initial.values())).shape[0])))
    trajectory = program.rollout(initial, keys=keys)
    data, loss_function = loss_inputs()
    stage_args = {"exit_index": 1} if stage == "student" else {}
    torch.manual_seed(5)
    _, metrics, _ = program.compute_loss(trajectory, stage, data, loss_function, **stage_args, keys=keys)
    torch.manual_seed(5)
    expected = reference.stage_outputs(trajectory, stage, **stage_args, keys=reference.keys(0, keys.samples))

    gap_keys = {"dmd/critic_teacher_x0_gap", "dmd/critic_teacher_x0_equal"}
    if stage == "fake_score":
        assert not gap_keys & metrics.keys()
        return
    for key, value in critic_teacher_x0_metrics(expected).items():
        torch.testing.assert_close(metrics[key], value, rtol=0, atol=0)
    assert metrics["dmd/critic_teacher_x0_gap"] > 0
