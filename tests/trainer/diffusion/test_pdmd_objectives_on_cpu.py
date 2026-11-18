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
"""CPU tests for projected DMD (PDMD) on top of the DMD2 loss.

Expected values restate the reference implementation's objectives
(https://github.com/ZeamoxWang/pdmd, ``src/pdmd/objectives.py``; its tests were
``tests/test_objectives.py`` before commit 8beae3c). Name mapping: pdmd ``x`` is
the student endpoint ``x_g``, ``t`` the teacher endpoint ``x0_real`` and ``c`` the
critic endpoint ``x0_fake``.
"""

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.distillation.utils import (
    dmd_gradient,
    dmd_surrogate_loss,
    fake_score_loss,
    project_orthogonal,
)
from verl_omni.workers.config.diffusion.actor import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.config.diffusion.dmd import DiffusionDMDConfig
from verl_omni.workers.utils.losses import diffusion_loss


class TestProjection:
    def test_projection_is_per_sample_and_orthogonal(self):
        torch.manual_seed(12)
        direction, residual = torch.randn(3, 2, 13), torch.randn(3, 2, 13)
        projected = project_orthogonal(direction, residual)
        torch.testing.assert_close((projected * residual).sum((1, 2)), torch.zeros(3), atol=2e-6, rtol=0)
        assert torch.all(projected.square().sum((1, 2)) <= direction.square().sum((1, 2)) + 1e-5)
        torch.testing.assert_close(project_orthogonal(projected, residual), projected, atol=1e-6, rtol=1e-5)
        for index in range(3):
            single = project_orthogonal(direction[index : index + 1], residual[index : index + 1])
            torch.testing.assert_close(projected[index : index + 1], single)

    def test_zero_residual_leaves_direction_unchanged(self):
        direction = torch.tensor([[2.0, 3.0]])
        torch.testing.assert_close(project_orthogonal(direction, torch.zeros_like(direction)), direction)

    def test_stereo_audio_is_projected_as_one_sample(self):
        # H3 audio is (B, 2, 32, T): stereo is a feature axis, not a sample axis.
        direction = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        residual = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        expected = torch.tensor([[[-0.5, 0.0], [0.0, 0.5]]])
        torch.testing.assert_close(project_orthogonal(direction, residual), expected, atol=1e-6, rtol=0)

    def test_residual_sign_does_not_change_the_projection(self):
        torch.manual_seed(5)
        direction, residual = torch.randn(2, 7), torch.randn(2, 7)
        torch.testing.assert_close(project_orthogonal(direction, residual), project_orthogonal(direction, -residual))

    def test_projection_runs_in_fp32(self):
        direction = torch.randn(1, 4, dtype=torch.bfloat16)
        assert project_orthogonal(direction, torch.randn(1, 4, dtype=torch.bfloat16)).dtype == torch.float32


class TestProjectedGradient:
    def test_projected_gradient_matches_student_perpendicular_form(self):
        torch.manual_seed(3)
        x_g, x0_real, x0_fake = torch.randn(1, 4, 9), torch.randn(1, 4, 9), torch.randn(1, 4, 9)
        gradient, normalizer, nonfinite = dmd_gradient(
            x0_fake, x0_real, x_g, normalization_epsilon=1e-5, projected=True, projection_epsilon=1e-8
        )
        p_real, p_fake = x_g - x0_real, x_g - x0_fake
        direction = p_real - p_fake
        expected = direction - (direction * p_fake).sum() / ((p_fake * p_fake).sum() + 1e-8) * p_fake
        torch.testing.assert_close(gradient, expected / p_real.abs().mean())
        torch.testing.assert_close(normalizer.flatten(), p_real.abs().mean().reshape(1))
        assert nonfinite == 0

    def test_projected_and_unprojected_share_the_pre_projection_normalizer(self):
        torch.manual_seed(4)
        x_g, x0_real, x0_fake = torch.randn(2, 3, 5), torch.randn(2, 3, 5), torch.randn(2, 3, 5)
        unprojected, normalizer_a, _ = dmd_gradient(x0_fake, x0_real, x_g, normalization_epsilon=1e-5)
        projected, normalizer_b, _ = dmd_gradient(x0_fake, x0_real, x_g, normalization_epsilon=1e-5, projected=True)
        torch.testing.assert_close(normalizer_a, normalizer_b)
        p_real = x_g - x0_real
        torch.testing.assert_close(unprojected, (x0_fake - x0_real) / p_real.abs().mean((1, 2), keepdim=True))
        assert not torch.allclose(projected, unprojected)

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_entries_are_zeroed_not_saturated(self, value):
        x_g = torch.zeros(1, 1, 2, 2)
        x0_real = torch.ones(1, 1, 2, 2)
        x0_fake = torch.ones(1, 1, 2, 2)
        x0_fake[0, 0, 0, 0] = value
        gradient, _, nonfinite = dmd_gradient(x0_fake, x0_real, x_g)
        assert nonfinite >= 1
        assert gradient[0, 0, 0, 0].item() == 0.0


class TestSurrogateAndCriticLoss:
    def test_unit_scale_surrogate_gradient_is_twice_the_update(self):
        x_g = torch.randn(2, 10, requires_grad=True)
        x0_real = torch.randn_like(x_g, requires_grad=True)
        x0_fake = torch.randn_like(x_g, requires_grad=True)
        gradient, _, _ = dmd_gradient(x0_fake, x0_real, x_g, projected=True)
        loss, _ = dmd_surrogate_loss(x_g, gradient, scale=1.0)
        loss.backward()
        torch.testing.assert_close(x_g.grad, 2 * gradient / x_g.numel())
        assert x0_real.grad is None and x0_fake.grad is None

    def test_clamp_is_per_sample(self):
        x_g = torch.zeros(2, 4, requires_grad=True)
        gradient = torch.stack([torch.full((4,), 10.0), torch.full((4,), 0.5)])
        loss, _ = dmd_surrogate_loss(x_g, gradient, scale=1.0, clamp_max=5.0)
        loss.backward()
        torch.testing.assert_close(x_g.grad[0], torch.zeros(4))
        torch.testing.assert_close(x_g.grad[1], 2 * gradient[1] / x_g.numel())
        assert loss.item() == pytest.approx((5.0 + 0.25) / 2)

    def test_critic_loss_clamp_is_per_sample(self):
        clean = torch.zeros(2, 3)
        noise = torch.stack([torch.full((3,), 10.0), torch.full((3,), 1.0)])
        prediction = torch.zeros(2, 3, requires_grad=True)
        loss, _ = fake_score_loss(prediction, noise, clean, clamp_max=5.0)
        loss.backward()
        torch.testing.assert_close(prediction.grad[0], torch.zeros(3))
        torch.testing.assert_close(prediction.grad[1], -2 * noise[1] / prediction.numel())


class TestDMDLossReadsItsConfig:
    @staticmethod
    def _batch(**non_tensor):
        data = TensorDict({}, batch_size=[2])
        tu.assign_non_tensor(data, gradient_accumulation_steps=1, sp_size=1, **non_tensor)
        return data

    @staticmethod
    def _actor():
        return DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=DiffusionLossConfig(loss_mode="dmd2"))

    def test_student_stage_applies_projection_scale_and_clamp(self):
        torch.manual_seed(8)
        config = DiffusionDMDConfig(projected=True, surrogate_scale=1.0, loss_clamp=5.0, normalization_epsilon=1e-5)
        x_g = torch.randn(2, 3, 4, requires_grad=True)
        x0_real, x0_fake = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
        loss, _ = diffusion_loss(
            self._actor(),
            {"generated_x0": x_g, "fake_x0": x0_fake, "teacher_x0": x0_real},
            self._batch(dmd_stage="student", dmd_config=config),
        )
        gradient, _, _ = dmd_gradient(x0_fake, x0_real, x_g.detach(), 1e-5, projected=True)
        expected, _ = dmd_surrogate_loss(x_g.detach(), gradient, scale=1.0, clamp_max=5.0)
        torch.testing.assert_close(loss, expected)

    def test_fake_score_stage_applies_clamp(self):
        clean = torch.zeros(2, 3)
        noise = torch.stack([torch.full((3,), 10.0), torch.full((3,), 1.0)])
        prediction = torch.zeros(2, 3, requires_grad=True)
        loss, _ = diffusion_loss(
            self._actor(),
            {"generated_x0": clean, "noise_pred": prediction, "noise": noise},
            self._batch(dmd_stage="fake_score", dmd_config=DiffusionDMDConfig(loss_clamp=5.0)),
        )
        expected, _ = fake_score_loss(prediction.detach(), noise, clean, clamp_max=5.0)
        torch.testing.assert_close(loss, expected)

    def test_missing_config_is_an_error_not_plain_dmd(self):
        output = {"generated_x0": torch.zeros(2, 2), "fake_x0": torch.ones(2, 2), "teacher_x0": torch.zeros(2, 2)}
        with pytest.raises(ValueError, match="dmd_config"):
            diffusion_loss(self._actor(), output, self._batch(dmd_stage="student"))

    def test_missing_stage_is_an_error_not_the_student_loss(self):
        output = {"generated_x0": torch.zeros(2, 2), "noise_pred": torch.ones(2, 2), "noise": torch.zeros(2, 2)}
        with pytest.raises(ValueError, match="dmd_stage"):
            diffusion_loss(self._actor(), output, self._batch(dmd_config=DiffusionDMDConfig()))

    @pytest.mark.parametrize(
        "overrides",
        [{"projection_epsilon": 0.0}, {"surrogate_scale": 0.0}, {"loss_clamp": 0.0}, {"loss_clamp": float("inf")}],
    )
    def test_invalid_pdmd_settings_are_rejected(self, overrides):
        with pytest.raises(ValueError):
            DiffusionDMDConfig(**overrides)
