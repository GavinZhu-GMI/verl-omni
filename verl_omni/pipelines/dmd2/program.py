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

"""Continuous-flow DMD2 / PDMD computation using caller-owned velocity predictors.

This module constructs inputs for the registered DMDLoss. Model construction,
parameter updates, distributed contexts and checkpoint publication stay with
the caller. No model adapter or training engine is imported here.

Every random draw is keyed, as in the PDMD reference (https://github.com/ZeamoxWang/pdmd, ``training/train.py``):
draw ``name`` of global sample ``sample`` in 0-based update ``update`` uses a generator seeded by
``stream_seed(seed, name, update, sample)``, and the exit index one seeded by ``stream_seed(seed, "exit", update)``.
An update thus reads no global RNG and no generator state, and a resumed run draws what a straight run draws.

Noise-time convention: sigma=1 is pure noise, x_sigma = (1 - sigma) * x0 + sigma * z,
and the predicted velocity is v = z - x0.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import torch
from tensordict import TensorDict
from torch import Tensor
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.distillation.utils import standard_cfg, velocity_to_x0
from verl_omni.utils.dataset.prompt_stream import stream_seed
from verl_omni.workers.config.diffusion.dmd import DiffusionDMDConfig

__all__ = [
    "DMD2Program",
    "DrawKeys",
    "NoiseDrawer",
    "Trajectory",
    "critic_teacher_x0_metrics",
    "standard_noise",
    "training_grid",
]


@dataclass(frozen=True)
class DrawKeys:
    """The keys of one microbatch's random draws: row ``b`` is global sample ``samples[b]`` of 0-based ``update``.

    Use :meth:`DMD2Program.keys` to build one.
    """

    seed: int
    update: int
    samples: tuple[int, ...]

    def generators(self, name: str, device: torch.device | str) -> list[torch.Generator]:
        """One generator per row on ``device``, seeded by ``stream_seed(seed, name, update, sample)``."""
        return [
            torch.Generator(device=device).manual_seed(stream_seed(self.seed, name, self.update, sample))
            for sample in self.samples
        ]


NoiseDrawer = Callable[[list[torch.Generator], dict[str, Tensor]], dict[str, Tensor]]
"""``draw_noise(generators, like)``: standard Gaussian FP32 noise with ``like``'s keys, shapes and device; row ``b``
draws from ``generators[b]`` alone."""


def standard_noise(generators: list[torch.Generator], like: dict[str, Tensor]) -> dict[str, Tensor]:
    """A :data:`NoiseDrawer` for latents whose draw layout is their own: each row is drawn in its ``like`` shape,
    modalities in ``like``'s order."""
    device = next(iter(like.values())).device
    rows = [
        {
            name: torch.randn(value.shape[1:], generator=generator, device=device, dtype=torch.float32)
            for name, value in like.items()
        }
        for generator in generators
    ]
    return {name: torch.stack([row[name] for row in rows]) for name in like}


@dataclass
class Trajectory:
    """A no-grad student rollout.

    ``states`` holds ``num_inference_steps + 1`` dicts of FP32 ``(B, ...)`` latents per modality:
    ``states[0]`` is the initial noise and ``states[-1]`` the generated endpoint. ``sigmas`` holds each
    modality's FP32 training grid of shape ``(num_inference_steps + 1,)``, from 1 to 0.
    """

    states: list[dict[str, Tensor]]
    sigmas: dict[str, Tensor]


def training_grid(config: DiffusionDMDConfig, device: torch.device) -> dict[str, Tensor]:
    """Shift the base-time knots per modality in FP64, then return FP32 grids of shape ``(N + 1,)``.

    The shift is written out rather than taken from ``timestep_shift``, which works in FP32: the PDMD
    reference shifts in FP64, and its grids must match bit for bit.
    """
    steps = config.num_inference_steps
    if config.base_time_knots is None:
        knots = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    else:
        knots = torch.tensor(config.base_time_knots, dtype=torch.float64)
    return {
        name: (shift * knots / (1 + (shift - 1) * knots)).float().to(device)
        for name, shift in config.modality_shifts.items()
    }


def critic_teacher_x0_metrics(outputs: dict[str, dict[str, Tensor]]) -> dict[str, Tensor]:
    """How far the fake score's clean estimate is from the teacher's at the student's score queries.

    Every modality's ``fake_x0`` and ``teacher_x0`` ``(B, ...)`` count, concatenated per sample in FP32, without
    modality weights. ``dmd/critic_teacher_x0_gap`` is the per-sample L2 norm of ``fake_x0 - teacher_x0``, averaged
    over samples; ``dmd/critic_teacher_x0_equal`` is the fraction of elements where the two are exactly equal.
    """
    fake = torch.cat([output["fake_x0"].detach().float().flatten(1) for output in outputs.values()], dim=1)
    teacher = torch.cat([output["teacher_x0"].detach().float().flatten(1) for output in outputs.values()], dim=1)
    return {
        "dmd/critic_teacher_x0_gap": (fake - teacher).norm(dim=1).mean(),
        "dmd/critic_teacher_x0_equal": (fake == teacher).float().mean(),
    }


def _per_sample(sigma: Tensor, like: Tensor) -> Tensor:
    """View a ``(B,)`` sigma so it broadcasts over a ``(B, ...)`` latent."""
    return sigma.view(-1, *([1] * (like.ndim - 1)))


def _renoise(clean: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
    """Forward process at a per-sample ``(B,)`` sigma, in FP32."""
    sigma = _per_sample(sigma, clean)
    return (1 - sigma) * clean.float() + sigma * noise.float()


def _endpoint(latents: Tensor, velocity: Tensor, sigma: Tensor) -> Tensor:
    """x0 estimate from a velocity prediction at a per-sample ``(B,)`` sigma."""
    return velocity_to_x0(latents, velocity, _per_sample(sigma, latents))


def _interval_sigma(grid: Tensor, step: Tensor | int, ratio: Tensor) -> Tensor:
    """``grid[step + 1] + ratio * (grid[step] - grid[step + 1])``: a point inside the step below ``step``."""
    return grid[step + 1] + ratio * (grid[step] - grid[step + 1])


class DMD2Program:
    """Compute DMD2 / PDMD stages with a model-independent velocity callback.

    Args:
        config: Continuous score sampling with explicit modality shifts and weights. The scalar
            shifts and ``score_sigma_min/max`` are unused here and must keep their defaults.
        predict: ``predict(role, latents, sigmas, negative)`` returning velocities with the
            latents' keys and shapes. ``sigmas`` maps each modality to a per-sample FP32 ``(B,)``
            tensor. Roles are ``student``, ``fake_score`` and ``teacher``; ``negative`` selects
            unconditional teacher inputs for CFG. With teacher ``layer_norm`` CFG, inputs and
            outputs must be packed ``(B, tokens, features)``.
        seed: The seed of every draw key; every rank of a run passes the same one.
        draw_noise: The :data:`NoiseDrawer` of every noise draw past the initial noise: the ``renoise`` sampler's,
            the critic's fresh noise and the fresh score-query noise. A model whose latents are packed draws in its
            latent layout and packs, so that its draws equal its reference's.

    Forward/recompute contexts and gradient reduction remain caller-owned.
    """

    def __init__(
        self,
        config: DiffusionDMDConfig,
        predict: Callable[[str, dict[str, Tensor], dict[str, Tensor], bool], dict[str, Tensor]],
        seed: int,
        *,
        draw_noise: NoiseDrawer,
    ):
        if config.score_discrete_steps != 0:
            raise ValueError("DMD2Program supports continuous score sampling only; set score_discrete_steps=0")
        if not config.modality_shifts:
            raise ValueError("DMD2Program requires explicit modality_shifts and modality_weights")
        for name in ("rollout_timestep_shift", "score_timestep_shift", "score_sigma_min", "score_sigma_max"):
            if getattr(config, name) != getattr(DiffusionDMDConfig, name):
                raise ValueError(f"DMD2Program does not read {name}; leave it at its default")
        self.config = config
        self.predict = predict
        self.seed = seed
        self.draw_noise = draw_noise

    def keys(self, update: int, samples: tuple[int, ...]) -> DrawKeys:
        """The draw keys of a microbatch whose rows are global ``samples`` of 0-based ``update``."""
        return DrawKeys(self.seed, update, tuple(samples))

    def draw_exit_index(self, update: int) -> int:
        """The student exit index of 0-based ``update``; every microbatch and rank of the update shares it."""
        generator = torch.Generator().manual_seed(stream_seed(self.seed, "exit", update))
        return int(torch.randint(self.config.num_inference_steps, (1,), generator=generator))

    def _checked_noise(self, generators: list[torch.Generator], like: dict[str, Tensor]) -> dict[str, Tensor]:
        noise = self.draw_noise(generators, like)
        if noise.keys() != like.keys() or any(
            noise[name].shape != value.shape or noise[name].dtype != torch.float32 or noise[name].device != value.device
            for name, value in like.items()
        ):
            raise ValueError("draw_noise must return FP32 noise with the latents' keys, shapes and device")
        return noise

    def _predict(
        self, role: str, latents: dict[str, Tensor], sigmas: dict[str, Tensor], negative: bool = False
    ) -> dict[str, Tensor]:
        predictions = self.predict(role, latents, sigmas, negative)
        if predictions.keys() != latents.keys():
            raise ValueError("Velocity predictions must have the same modality keys as the latents")
        if any(predictions[name].shape != value.shape for name, value in latents.items()):
            raise ValueError("Velocity predictions must have the same shapes as the latents")
        return predictions

    def _check_latents(self, initial: dict[str, Tensor], keys: DrawKeys) -> None:
        if initial.keys() != self.config.modality_shifts.keys():
            raise ValueError("Latent modalities must match modality_shifts and modality_weights")
        first = next(iter(initial.values()))
        if any(
            value.ndim < 2
            or value.numel() == 0
            or not value.is_floating_point()
            or value.shape[0] != first.shape[0]
            or value.device != first.device
            for value in initial.values()
        ):
            raise ValueError("Latents require nonempty floating (B, ...) tensors with a shared batch size and device")
        if len(keys.samples) != first.shape[0]:
            raise ValueError(f"The draw keys name {len(keys.samples)} samples for a batch of {first.shape[0]}")

    @torch.no_grad()
    def rollout(self, initial: dict[str, Tensor], *, keys: DrawKeys) -> Trajectory:
        """Run the student for all ``num_inference_steps`` without autograd and keep every state.

        ``initial`` holds standard Gaussian noise of shape ``(B, ...)`` per modality. ``euler`` steps along
        the predicted velocity; ``renoise`` jumps to the predicted x0 and adds fresh noise, drawn under
        ``keys`` as ``rollout_renoise``, at the next knot.
        """
        self._check_latents(initial, keys)
        first = next(iter(initial.values()))
        batch, device = first.shape[0], first.device
        grid = training_grid(self.config, device)
        state = {name: value.detach().float() for name, value in initial.items()}
        states = [state]
        generators = keys.generators("rollout_renoise", device) if self.config.sampler == "renoise" else []
        for index in range(self.config.num_inference_steps):
            current = {name: grid[name][index] for name in state}
            following = {name: grid[name][index + 1] for name in state}
            velocities = self._predict("student", state, {name: sigma.repeat(batch) for name, sigma in current.items()})
            if self.config.sampler == "euler":
                state = {
                    name: value + (following[name] - current[name]) * velocities[name].float()
                    for name, value in state.items()
                }
            elif self.config.sampler == "renoise":
                clean = {name: value - current[name] * velocities[name].float() for name, value in state.items()}
                noise = self._checked_noise(generators, clean)
                state = {
                    name: (1 - following[name]) * value + following[name] * noise[name] for name, value in clean.items()
                }
            else:
                raise ValueError(f"Unknown sampler {self.config.sampler!r}")
            states.append(state)
        return Trajectory(states=states, sigmas=grid)

    def _critic_sigmas(self, trajectory: Trajectory, keys: DrawKeys) -> dict[str, Tensor]:
        """Per-sample fake-score sigmas, drawn on CPU under ``keys`` as ``draws``."""
        device = next(iter(trajectory.sigmas.values())).device
        names = list(self.config.modality_shifts)
        generators = keys.generators("draws", "cpu")
        if self.config.critic_time == "uniform":
            lead = self.config.critic_time_lead_modality or names[0]
            followers = [name for name in names if name != lead]
            # per row: the lead's sigma, then a draw and a redraw per follower, drawn and compared in FP64 as the PDMD
            # reference does (``critic_times``, https://github.com/ZeamoxWang/pdmd at 6b6e106,
            # ``src/pdmd/schedule.py:45-56``); each sigma is cast to FP32 last
            uniforms = torch.stack(
                [
                    torch.rand(1 + 2 * len(followers), generator=generator, dtype=torch.float64)
                    for generator in generators
                ]
            )
            lead_sigma = uniforms[:, 0]
            sigmas = {lead: lead_sigma.float()}
            floor = self.config.critic_time_follower_floor
            for position, name in enumerate(followers):
                draw, redraw = uniforms[:, 1 + 2 * position], uniforms[:, 2 + 2 * position]
                sigmas[name] = torch.where(
                    lead_sigma > self.config.critic_time_lead_threshold, floor + (1 - floor) * redraw, draw
                ).float()
        elif self.config.critic_time == "query_matched":
            steps = self.config.num_inference_steps
            step = torch.cat([torch.randint(steps, (1,), generator=generator) for generator in generators]).to(device)
            ratio = torch.cat([torch.rand(1, generator=generator) for generator in generators]).to(device)
            sigmas = {name: _interval_sigma(trajectory.sigmas[name], step, ratio) for name in names}
        else:
            raise ValueError(f"Unknown critic_time {self.config.critic_time!r}")
        return {name: sigmas[name].to(device) for name in names}

    def _fake_score_outputs(self, trajectory: Trajectory, keys: DrawKeys) -> dict[str, dict[str, Tensor]]:
        endpoint = trajectory.states[-1]
        sigmas = self._critic_sigmas(trajectory, keys)
        if self.config.critic_noise == "fresh":
            noise = self._checked_noise(keys.generators("renoise", next(iter(endpoint.values())).device), endpoint)
        elif self.config.critic_noise == "trajectory":
            noise = trajectory.states[0]
        else:
            raise ValueError(f"Unknown critic_noise {self.config.critic_noise!r}")
        noisy = {name: _renoise(value, noise[name], sigmas[name]) for name, value in endpoint.items()}
        prediction = self._predict("fake_score", noisy, sigmas)
        return {
            name: {"generated_x0": value, "noise_pred": prediction[name], "noise": noise[name]}
            for name, value in endpoint.items()
        }

    def _student_outputs(
        self, trajectory: Trajectory, exit_index: Optional[int], keys: DrawKeys
    ) -> dict[str, dict[str, Tensor]]:
        steps = self.config.num_inference_steps
        if isinstance(exit_index, bool) or not isinstance(exit_index, int) or not 0 <= exit_index < steps:
            raise ValueError(f"The student stage needs an integer exit_index in [0, {steps}), got {exit_index!r}")
        start = trajectory.states[exit_index]
        if self.config.teacher_guidance_scale != 1 and self.config.cfg_norm == "layer_norm":
            if any(value.ndim != 3 for value in start.values()):
                raise ValueError(
                    "Teacher layer_norm CFG requires packed (B, tokens, features) latents and velocities; "
                    "pack the latents before calling the program or use cfg_norm='none'/'scalar'"
                )
        first = next(iter(start.values()))
        batch, device = first.shape[0], first.device
        grid = trajectory.sigmas
        start_sigmas = {name: grid[name][exit_index].repeat(batch) for name in start}
        velocity = self._predict("student", start, start_sigmas)
        clean = {name: _endpoint(value, velocity[name], start_sigmas[name]) for name, value in start.items()}
        # one per sample, shared by every modality
        ratio = torch.cat([torch.rand(1, generator=generator) for generator in keys.generators("draws", "cpu")]).to(
            device
        )
        query_sigmas = {name: _interval_sigma(grid[name], exit_index, ratio) for name in start}
        with torch.no_grad():
            if self.config.score_query_noise == "student_estimate":
                noise = {
                    name: value.float() + _per_sample(1 - start_sigmas[name], value) * velocity[name].float()
                    for name, value in start.items()
                }
            elif self.config.score_query_noise == "trajectory":
                noise = trajectory.states[0]
            elif self.config.score_query_noise == "fresh":
                noise = self._checked_noise(keys.generators("score_query", device), clean)
            else:
                raise ValueError(f"Unknown score_query_noise {self.config.score_query_noise!r}")
            query = {name: _renoise(value.detach(), noise[name], query_sigmas[name]) for name, value in clean.items()}
            fake = self._predict("fake_score", query, query_sigmas)
            real = self._predict("teacher", query, query_sigmas)
            if self.config.teacher_guidance_scale != 1:
                negative = self._predict("teacher", query, query_sigmas, negative=True)
                real = {
                    name: standard_cfg(value, negative[name], self.config.teacher_guidance_scale, self.config.cfg_norm)
                    for name, value in real.items()
                }
            fake_x0 = {name: _endpoint(query[name], value, query_sigmas[name]) for name, value in fake.items()}
            teacher_x0 = {name: _endpoint(query[name], value, query_sigmas[name]) for name, value in real.items()}
        return {
            name: {"generated_x0": value, "fake_x0": fake_x0[name], "teacher_x0": teacher_x0[name]}
            for name, value in clean.items()
        }

    def stage_outputs(
        self, trajectory: Trajectory, stage: str, exit_index: Optional[int] = None, *, keys: DrawKeys
    ) -> dict[str, dict[str, Tensor]]:
        """Construct the registered loss's inputs for one stage from a stored rollout, drawing under ``keys``.

        ``fake_score`` re-noises the rollout endpoint at per-sample critic times and takes no exit
        index. ``student`` re-runs the student with autograd at ``states[exit_index]``.
        """
        batch = next(iter(trajectory.states[0].values())).shape[0]
        if len(keys.samples) != batch:
            raise ValueError(f"The draw keys name {len(keys.samples)} samples for a batch of {batch}")
        if stage == "fake_score":
            if exit_index is not None:
                raise ValueError("The fake_score stage trains on the rollout endpoint and takes no exit_index")
            return self._fake_score_outputs(trajectory, keys)
        if stage == "student":
            return self._student_outputs(trajectory, exit_index, keys)
        raise ValueError(f"Unknown DMD stage {stage!r}")

    def compute_loss(
        self,
        trajectory: Trajectory,
        stage: str,
        data: TensorDict,
        loss_function: Callable,
        *,
        exit_index: Optional[int] = None,
        keys: DrawKeys,
        dp_group: Any = None,
    ) -> tuple[Tensor, dict[str, Any], bool]:
        """Call the registered loss per modality; report nonfinite tensors to the caller.

        ``loss_function`` is the public diffusion loss wrapper, bound to its actor
        config. The caller supplies ``gradient_accumulation_steps`` and ``sp_size``
        in ``data``; neither is overwritten or applied a second time here.
        """
        for name in ("gradient_accumulation_steps", "sp_size"):
            value = tu.get_non_tensor_data(data, name, default=None)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Caller must supply a positive integer {name} for the public loss wrapper")
        tu.assign_non_tensor(data, dmd_stage=stage, dmd_config=self.config)
        outputs = self.stage_outputs(trajectory, stage, exit_index, keys=keys)
        loss, metrics, invalid = 0, {}, False
        for modality, output in outputs.items():
            part, part_metrics = loss_function(model_output=output, data=data, dp_group=dp_group)
            loss = loss + self.config.modality_weights[modality] * part
            metrics.update({f"{modality}/{key}": value for key, value in part_metrics.items()})
            invalid |= any(not bool(torch.isfinite(value).all()) for value in output.values())
        if stage == "student":
            metrics.update(critic_teacher_x0_metrics(outputs))
        return loss, metrics, invalid or not bool(torch.isfinite(loss.detach()))

    def sample(self, initial: dict[str, Tensor], *, keys: DrawKeys) -> dict[str, Tensor]:
        """Generate final FP32 modality latents, drawing under ``keys``."""
        return self.rollout(initial, keys=keys).states[-1]
