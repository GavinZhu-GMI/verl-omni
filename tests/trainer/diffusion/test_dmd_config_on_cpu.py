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

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.optimizer import FSDPOptimizerConfig

import verl_omni
from verl_omni.workers.config import DiffusionDistillationConfig, DiffusionDMDConfig

CONFIG_DIR = str(Path(verl_omni.__file__).parent / "trainer" / "config")


def compose_config(overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=list(overrides))


class TestDMDConfig:
    @pytest.mark.parametrize("value", [True, 0, -1, 1.5])
    def test_generator_step_count_is_a_positive_integer(self, value):
        with pytest.raises(ValueError, match="num_inference_steps"):
            DiffusionDMDConfig(num_inference_steps=value)

    @pytest.mark.parametrize(
        "shifts,weights",
        [
            ({"image": 2.0}, {}),
            ({"image": 0.5}, {"image": 1.0}),
            ({"image": True}, {"image": 1.0}),
            ({"image": "2"}, {"image": 1.0}),
            ({"image": float("inf")}, {"image": 1.0}),
            ({"image": 2.0}, {"image": 0.0}),
            ({"image": 2.0}, {"image": float("nan")}),
            ({"image": 2.0}, {"image": True}),
        ],
    )
    def test_invalid_modality_settings_are_rejected(self, shifts, weights):
        with pytest.raises(ValueError, match="modality"):
            DiffusionDMDConfig(modality_shifts=shifts, modality_weights=weights)

    def test_modality_defaults_are_not_shared(self):
        first, second = DiffusionDMDConfig(), DiffusionDMDConfig()
        first.modality_shifts["image"] = 2.0
        assert second.modality_shifts == {}

    def test_defaults_are_dmd2_without_opd_activation(self):
        config = DiffusionDMDConfig()
        assert config.fake_update_ratio == 2
        assert config.export_role == "student"
        assert config.score_discrete_steps == 1000
        assert (config.score_sigma_min, config.score_sigma_max) == (0.02, 0.98)
        assert isinstance(config.fake_score_optim, FSDPOptimizerConfig)
        assert config.fake_score_optim.lr == pytest.approx(2e-5)
        assert not DiffusionDistillationConfig().enabled
        assert not hasattr(DiffusionDistillationConfig(), "distribution_matching")

    @pytest.mark.parametrize(
        "field", ["fake_update_ratio", "student_micro_batch_size_per_gpu", "fake_score_micro_batch_size_per_gpu"]
    )
    @pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2"])
    def test_positive_counts_are_not_silently_coerced(self, field, value):
        with pytest.raises(ValueError, match="positive integer"):
            DiffusionDMDConfig(**{field: value})

    def test_stage_micro_batch_sizes_must_match(self):
        """The student trains each microbatch on the fake_score rollout of the same slot."""
        with pytest.raises(ValueError, match="must equal"):
            DiffusionDMDConfig(fake_score_micro_batch_size_per_gpu=1, student_micro_batch_size_per_gpu=2)

    @pytest.mark.parametrize(
        "schedule",
        [{"lr_warmup_steps": 2}, {"lr_warmup_steps_ratio": 0.1}, {"lr_scheduler_type": "cosine"}],
        ids=["warmup", "warmup_ratio", "cosine"],
    )
    def test_a_fake_score_learning_rate_schedule_is_rejected(self, schedule):
        """DMD trains at a constant LR; the fake_score scheduler would step fake_update_ratio times per step."""
        with pytest.raises(ValueError, match="constant learning rate"):
            DiffusionDMDConfig(fake_score_optim=FSDPOptimizerConfig(lr=2e-5, **schedule))

    def test_a_constant_fake_score_learning_rate_is_accepted(self):
        DiffusionDMDConfig(fake_score_optim=FSDPOptimizerConfig(lr=2e-5, lr_warmup_steps=0))

    @pytest.mark.parametrize("field", ["score_discrete_steps", "ema_start_step"])
    @pytest.mark.parametrize("value", [True, -1, 0.5])
    def test_nonnegative_counts_are_validated(self, field, value):
        with pytest.raises(ValueError, match="non-negative integer"):
            DiffusionDMDConfig(**{field: value})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"teacher_guidance_scale": 0},
            {"teacher_guidance_scale": float("inf")},
            {"negative_prompt": None},
            {"normalization_epsilon": 0},
            {"normalization_epsilon": float("nan")},
            {"cfg_norm": "legacy"},
            {"rollout_timestep_shift": 0.5},
            {"score_timestep_shift": float("nan")},
            {"score_sigma_min": 0},
            {"score_sigma_min": 0.9, "score_sigma_max": 0.2},
            {"score_sigma_max": 1.1},
            {"ema_decay": -0.1},
            {"ema_decay": float("nan")},
            {"export_role": "fake_score"},
        ],
    )
    def test_invalid_values_fail_closed(self, kwargs):
        with pytest.raises(ValueError):
            DiffusionDMDConfig(**kwargs)

    @pytest.mark.parametrize("field", ["score_time_min", "score_time_max"])
    def test_removed_score_time_bounds_are_rejected(self, field):
        with pytest.raises(TypeError, match=field):
            DiffusionDMDConfig(**{field: 0.5})

    def test_pdmd_sampling_defaults(self):
        """Defaults are the PDMD reference run's: Euler, uniform grid, noise_hat query, fresh critic noise,
        uniform critic time with the 0.95 / 0.85 rule."""
        config = DiffusionDMDConfig()
        assert config.sampler == "euler"
        assert config.base_time_knots is None
        assert config.score_query_noise == "student_estimate"
        assert config.critic_noise == "fresh"
        assert config.critic_time == "uniform"
        assert config.critic_time_lead_modality is None
        assert (config.critic_time_lead_threshold, config.critic_time_follower_floor) == (0.95, 0.85)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"sampler": "dpm"}, "sampler"),
            ({"score_query_noise": "noise_hat"}, "score_query_noise"),
            ({"critic_noise": "student_estimate"}, "critic_noise"),
            ({"critic_time": "shifted"}, "critic_time"),
            ({"sampler": "renoise", "score_query_noise": "trajectory"}, "renoise"),
            ({"sampler": "renoise", "critic_noise": "trajectory"}, "renoise"),
            ({"critic_time_lead_threshold": 1.5}, "critic_time_lead_threshold"),
            ({"critic_time_lead_threshold": float("nan")}, "critic_time_lead_threshold"),
            ({"critic_time_follower_floor": -0.1}, "critic_time_follower_floor"),
            ({"critic_time_follower_floor": 1.0}, "critic_time_follower_floor"),
            ({"critic_time_follower_floor": True}, "critic_time_follower_floor"),
        ],
    )
    def test_invalid_sampling_choices_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DiffusionDMDConfig(**kwargs)

    @pytest.mark.parametrize(
        "knots",
        [
            [1.0, 0.5, 0.0],  # three knots for four steps
            [0.9, 0.75, 0.5, 0.25, 0.0],  # does not start at pure noise
            [1.0, 0.75, 0.5, 0.25, 0.1],  # does not end clean
            [1.0, 0.75, 0.75, 0.25, 0.0],  # not strictly decreasing
            [1.0, 0.75, float("nan"), 0.25, 0.0],
            [1.0, 0.75, True, 0.25, 0.0],
            "1,0.75,0.5,0.25,0",
        ],
    )
    def test_invalid_base_time_knots_are_rejected(self, knots):
        with pytest.raises(ValueError, match="base_time_knots"):
            DiffusionDMDConfig(base_time_knots=knots)

    def test_valid_knots_are_accepted(self):
        config = DiffusionDMDConfig(num_inference_steps=6, base_time_knots=[1, 0.75, 0.5, 0.25, 0.1, 0.04, 0])
        assert config.base_time_knots == [1, 0.75, 0.5, 0.25, 0.1, 0.04, 0]

    def test_trajectory_noise_is_valid_with_euler(self):
        config = DiffusionDMDConfig(score_query_noise="trajectory", critic_noise="trajectory")
        assert (config.score_query_noise, config.critic_noise) == ("trajectory", "trajectory")

    def test_uniform_critic_time_needs_a_lead_among_several_modalities(self):
        modalities = {"modality_shifts": {"video": 1.0, "audio": 1.0}, "modality_weights": {"video": 0.8, "audio": 0.8}}
        with pytest.raises(ValueError, match="critic_time_lead_modality"):
            DiffusionDMDConfig(**modalities)
        with pytest.raises(ValueError, match="critic_time_lead_modality"):
            DiffusionDMDConfig(**modalities, critic_time_lead_modality="image")
        assert DiffusionDMDConfig(**modalities, critic_time_lead_modality="video").critic_time_lead_modality == "video"
        assert DiffusionDMDConfig(**modalities, critic_time="query_matched").critic_time_lead_modality is None

    def test_single_modality_needs_no_lead(self):
        config = DiffusionDMDConfig(modality_shifts={"image": 3.0}, modality_weights={"image": 1.0})
        assert config.critic_time_lead_modality is None

    def test_optimizer_defaults_are_not_shared(self):
        first, second = DiffusionDMDConfig(), DiffusionDMDConfig()
        first.fake_score_optim.total_training_steps = 4
        assert second.fake_score_optim.total_training_steps == -1

    def test_typed_hydra_group_matches_dataclass_defaults(self):
        config = compose_config()
        dmd = omega_conf_to_dataclass(config.dmd)
        assert isinstance(dmd, DiffusionDMDConfig)
        assert isinstance(dmd.fake_score_optim, FSDPOptimizerConfig)
        assert OmegaConf.structured(dmd) == OmegaConf.structured(DiffusionDMDConfig())
        assert omega_conf_to_dataclass(config.distillation).enabled is False

    def test_dmd_loss_can_be_composed_without_model_loading(self):
        config = compose_config(
            [
                "actor_rollout_ref.model.algorithm=dmd2",
                "dmd.fake_update_ratio=3",
                "dmd.fake_score_optim.lr=1e-5",
            ]
        )
        loss = omega_conf_to_dataclass(config.actor_rollout_ref.actor.diffusion_loss)
        assert loss.loss_mode == "dmd2"
        assert omega_conf_to_dataclass(config.dmd).fake_update_ratio == 3

    def test_pdmd_program_overrides_compose_from_yaml(self):
        from verl_omni.pipelines.dmd2.program import DMD2Program

        config = compose_config(
            [
                "dmd.score_discrete_steps=0",
                "+dmd.modality_shifts.video=12.0",
                "+dmd.modality_shifts.audio=3.0",
                "+dmd.modality_weights.video=0.8",
                "+dmd.modality_weights.audio=0.8",
                "dmd.critic_time_lead_modality=video",
                "dmd.sampler=renoise",
                "dmd.num_inference_steps=6",
                "dmd.base_time_knots=[1.0,0.75,0.5,0.25,0.1,0.04,0.0]",
            ]
        )
        dmd = omega_conf_to_dataclass(config.dmd)
        assert dmd.sampler == "renoise"
        assert dmd.base_time_knots == [1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0]
        assert dmd.critic_time_lead_modality == "video"
        program = DMD2Program(dmd, lambda role, latents, sigmas, negative: latents, seed=42)
        assert program.config is dmd

    @pytest.mark.parametrize("schedule", ["inline", "one_step_off"])
    def test_opd_group_is_unchanged(self, schedule):
        config = compose_config(
            [
                "distillation.enabled=true",
                f"distillation.scheduler={schedule}",
                "distillation.teacher_models.teacher_model.model_path=/ckpt/teacher",
            ]
        )
        opd = omega_conf_to_dataclass(config.distillation)
        assert opd.scheduler == schedule
        assert opd.teacher_models["default"].model_path == "/ckpt/teacher"
        assert "distribution_matching" not in OmegaConf.to_container(config.distillation)
        assert isinstance(omega_conf_to_dataclass(config.dmd), DiffusionDMDConfig)

    @pytest.mark.parametrize("field", ["recipe", "profile", "adversarial", "rollout_strategy", "teacher_models"])
    def test_out_of_scope_options_are_not_accepted(self, field):
        with pytest.raises(ConfigCompositionException):
            compose_config([f"dmd.{field}=unsupported"])

    def test_training_stages_put_fake_score_updates_first(self):
        assert DiffusionDMDConfig(fake_update_ratio=5).training_stages() == ("fake_score",) * 5 + ("student",)
