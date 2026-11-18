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

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

from verl.base_config import BaseConfig
from verl.workers.config.optimizer import FSDPOptimizerConfig

__all__ = ["DiffusionDMDConfig"]


@dataclass
class DiffusionDMDConfig(BaseConfig):
    """DMD2 distribution-only settings, independent of on-policy distillation."""

    # Generator sampling steps, independent of the score-training timestep.
    num_inference_steps: int = 4
    # Explicit per-modality flow shifts and loss weights for continuous sampling.
    modality_shifts: dict[str, float] = field(default_factory=dict)
    modality_weights: dict[str, float] = field(default_factory=dict)
    # Fake-score update attempts per student attempt.
    fake_update_ratio: int = 2
    # Student physical microbatch size per data-parallel rank.
    student_micro_batch_size_per_gpu: int = 1
    # Fake-score physical microbatch size per data-parallel rank.
    fake_score_micro_batch_size_per_gpu: int = 1
    # Fake score uses the existing optimizer type, independently of the actor.
    fake_score_optim: FSDPOptimizerConfig = field(
        default_factory=partial(FSDPOptimizerConfig, lr=2e-5, weight_decay=0.001)
    )
    # Qwen teacher CFG; student and fake score remain conditional-only.
    teacher_guidance_scale: float = 4.0
    # Teacher CFG normalization; DMD2Program layer_norm requires (B, tokens, features).
    cfg_norm: str = "layer_norm"
    # Explicit negative teacher condition; an empty string is also valid.
    negative_prompt: str = " "
    # Per-sample score-difference normalization floor.
    normalization_epsilon: float = 1e-6
    # PDMD (arXiv 2609.35768): remove the update component along the student-critic residual.
    projected: bool = False
    # Denominator floor of the PDMD projection.
    projection_epsilon: float = 1e-8
    # Factor on the student surrogate MSE; PDMD uses 1.0.
    surrogate_scale: float = 0.5
    # Per-sample clamp on the student and fake-score losses; None disables it, PDMD uses 5.0.
    loss_clamp: Optional[float] = None
    # DMD2Program generator sampler: deterministic Euler (the PDMD reference), or x0 then fresh noise at
    # the next knot. The served sampler must match the one trained.
    sampler: str = "euler"
    # Base-time knots from 1 (noise) to 0 (clean), num_inference_steps + 1 of them, before modality shifts.
    # None selects the uniform grid linspace(1, 0, num_inference_steps + 1).
    base_time_knots: Optional[list[float]] = None
    # Noise that re-noises the student's x0 at the score query: the student's own noise estimate (the PDMD
    # reference), the rollout's initial noise (Euler only), or a fresh Gaussian (as the PDMD paper's Eq. 1).
    score_query_noise: str = "student_estimate"
    # Noise that re-noises the rollout endpoint for the fake score: a fresh Gaussian (the PDMD reference)
    # or the rollout's initial noise (Euler only), which pairs with trajectory query noise.
    critic_noise: str = "fresh"
    # Fake-score time law: per-sample U(0, 1) per modality with the lead-modality rule below (the PDMD
    # reference), or the student's score-query law on the training grid, which matches the queries on a
    # shifted grid.
    critic_time: str = "uniform"
    # Lead modality of the uniform critic-time rule; required with several modalities.
    critic_time_lead_modality: Optional[str] = None
    # Lead sigma above which every other modality redraws its critic sigma from U(floor, 1); PDMD uses 0.95.
    critic_time_lead_threshold: float = 0.95
    # Lower bound of that redraw; PDMD uses 0.85.
    critic_time_follower_floor: float = 0.85
    # Fixed rational shift applied once to the inference sigma grid.
    rollout_timestep_shift: float = 3.0
    # Discrete score grid size; zero selects continuous score sampling.
    score_discrete_steps: int = 1000
    # Lower score sigma bound.
    score_sigma_min: float = 0.02
    # Upper score sigma bound.
    score_sigma_max: float = 0.98
    # Rational shift for discrete score sampling only.
    score_timestep_shift: float = 3.0
    # EMA decay, applied only after successful student updates.
    ema_decay: float = 0.999
    # Successful student update count at which EMA starts.
    ema_start_step: int = 0
    # Default inference artifact; EMA is an explicit alternative.
    export_role: str = "student"

    def __post_init__(self):
        if (
            isinstance(self.num_inference_steps, bool)
            or not isinstance(self.num_inference_steps, int)
            or self.num_inference_steps < 1
        ):
            raise ValueError("num_inference_steps must be a positive integer")
        if self.modality_shifts.keys() != self.modality_weights.keys():
            raise ValueError("modality_shifts and modality_weights must have matching keys")
        for name, minimum, inclusive in (("modality_shifts", 1, True), ("modality_weights", 0, False)):
            for value in getattr(self, name).values():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                    or (value < minimum if inclusive else value <= minimum)
                ):
                    raise ValueError(f"Invalid {name}: shifts must be finite and >= 1, weights finite and > 0")
        for name in ("fake_update_ratio", "student_micro_batch_size_per_gpu", "fake_score_micro_batch_size_per_gpu"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in ("score_discrete_steps", "ema_start_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        for name in (
            "teacher_guidance_scale",
            "normalization_epsilon",
            "rollout_timestep_shift",
            "score_sigma_min",
            "score_sigma_max",
            "score_timestep_shift",
            "ema_decay",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.teacher_guidance_scale <= 0:
            raise ValueError("teacher_guidance_scale must be positive")
        if not isinstance(self.negative_prompt, str):
            raise ValueError("negative_prompt must be an explicit string for teacher conditioning")
        if self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        if not isinstance(self.projected, bool):
            raise ValueError(f"projected must be a bool, got {self.projected!r}")
        for name in ("projection_epsilon", "surrogate_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if self.loss_clamp is not None and (
            isinstance(self.loss_clamp, bool)
            or not isinstance(self.loss_clamp, int | float)
            or not math.isfinite(self.loss_clamp)
            or self.loss_clamp <= 0
        ):
            raise ValueError(f"loss_clamp must be None or finite and positive, got {self.loss_clamp!r}")
        if self.rollout_timestep_shift < 1 or self.score_timestep_shift < 1:
            raise ValueError("rollout_timestep_shift and score_timestep_shift must be at least 1")
        if not 0 < self.score_sigma_min < self.score_sigma_max <= 1:
            raise ValueError("score_sigma bounds must satisfy 0 < score_sigma_min < score_sigma_max <= 1")
        for name, valid in (
            ("sampler", {"euler", "renoise"}),
            ("score_query_noise", {"student_estimate", "trajectory", "fresh"}),
            ("critic_noise", {"fresh", "trajectory"}),
            ("critic_time", {"uniform", "query_matched"}),
        ):
            if getattr(self, name) not in valid:
                raise ValueError(f"Invalid {name}: {getattr(self, name)}. Must be one of {sorted(valid)}")
        if self.sampler == "renoise" and "trajectory" in (self.score_query_noise, self.critic_noise):
            raise ValueError(
                "trajectory noise needs sampler=euler: with renoise the initial noise does not determine x0"
            )
        knots = self.base_time_knots
        if knots is not None:
            if (
                not isinstance(knots, list | tuple)
                or len(knots) != self.num_inference_steps + 1
                or any(isinstance(k, bool) or not isinstance(k, int | float) or not math.isfinite(k) for k in knots)
                or knots[0] != 1
                or knots[-1] != 0
                or any(a <= b for a, b in zip(knots, knots[1:], strict=False))
            ):
                raise ValueError(
                    "base_time_knots must be num_inference_steps + 1 finite numbers, strictly decreasing from 1 to 0"
                )
        threshold, floor = self.critic_time_lead_threshold, self.critic_time_follower_floor
        if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0 <= threshold <= 1:
            raise ValueError(f"critic_time_lead_threshold must lie in [0, 1], got {threshold!r}")
        if isinstance(floor, bool) or not isinstance(floor, int | float) or not 0 <= floor < 1:
            raise ValueError(f"critic_time_follower_floor must lie in [0, 1), got {floor!r}")
        lead = self.critic_time_lead_modality
        if lead is not None and lead not in self.modality_shifts:
            raise ValueError(f"critic_time_lead_modality {lead!r} is not one of {sorted(self.modality_shifts)}")
        if self.critic_time == "uniform" and len(self.modality_shifts) > 1 and lead is None:
            raise ValueError("critic_time=uniform with several modalities needs critic_time_lead_modality")
        if not 0 <= self.ema_decay <= 1:
            raise ValueError("ema_decay must be between 0 and 1")
        valid_cfg_norms = {"none", "layer_norm", "scalar"}
        if self.cfg_norm not in valid_cfg_norms:
            raise ValueError(f"Invalid cfg_norm: {self.cfg_norm}. Must be one of {sorted(valid_cfg_norms)}")
        valid_export_roles = {"student", "student_ema"}
        if self.export_role not in valid_export_roles:
            raise ValueError(f"Invalid export_role: {self.export_role}. Must be one of {sorted(valid_export_roles)}")
