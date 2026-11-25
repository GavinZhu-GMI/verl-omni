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
"""The uniform critic sigmas of DMD2Program equal the PDMD reference's bit for bit, key by key."""

import torch

from verl_omni.pipelines.dmd2.program import DMD2Program, standard_noise
from verl_omni.utils.dataset.prompt_stream import stream_seed
from verl_omni.workers.config.diffusion.dmd import DiffusionDMDConfig

SEED = 42
UPDATE = 0
SAMPLES = 400  # about 20 redraws expected at P(video > 0.95) = 0.05


def pdmd_critic_times(generator, device, video_threshold=0.95, audio_floor=0.85):
    """``critic_times`` of https://github.com/ZeamoxWang/pdmd at 6b6e106, ``src/pdmd/schedule.py:45-56``."""
    u = torch.rand((3,), generator=generator, device=device, dtype=torch.float64)
    video = u[0]
    audio = torch.where(video > video_threshold, audio_floor + (1 - audio_floor) * u[2], u[1])
    return video.float(), audio.float()


def _fake_score_sigmas(samples: tuple[int, ...]) -> dict[str, torch.Tensor]:
    config = DiffusionDMDConfig(
        modality_shifts={"video": 12.0, "audio": 3.0},
        modality_weights={"video": 1.0, "audio": 1.0},
        score_discrete_steps=0,
        cfg_norm="none",
        teacher_guidance_scale=1.0,
        critic_time="uniform",
        critic_time_lead_modality="video",
        critic_noise="trajectory",  # the critic sigmas are then the update's only random draw
        sampler="euler",
    )
    seen = {}

    def predict(role, latents, sigmas, negative):
        if role == "fake_score":
            seen.update({name: value.clone() for name, value in sigmas.items()})
        return {name: torch.zeros_like(value) for name, value in latents.items()}

    program = DMD2Program(config, predict, seed=SEED, draw_noise=standard_noise)
    batch = len(samples)
    initial = {"video": torch.zeros(batch, 2, 3), "audio": torch.zeros(batch, 1, 3)}
    keys = program.keys(UPDATE, samples)
    program.stage_outputs(program.rollout(initial, keys=keys), "fake_score", keys=keys)
    return seen


def test_uniform_critic_sigmas_are_pdmds_float64_draws_cast_to_float32():
    """Catches float32 uniforms, or the redraw done in float32: both move sigmas off pdmd's for some keys."""
    samples = tuple(range(SAMPLES))
    sigmas = _fake_score_sigmas(samples)
    expected = [
        pdmd_critic_times(torch.Generator().manual_seed(stream_seed(SEED, "draws", UPDATE, sample)), "cpu")
        for sample in samples
    ]
    expected_video = torch.stack([video for video, _ in expected])
    expected_audio = torch.stack([audio for _, audio in expected])

    redraws = int((expected_video.double() > 0.95).sum())
    assert redraws >= 10, f"only {redraws} keys take the redraw branch"
    assert sigmas["video"].dtype == sigmas["audio"].dtype == torch.float32
    torch.testing.assert_close(sigmas["video"], expected_video, rtol=0, atol=0)
    torch.testing.assert_close(sigmas["audio"], expected_audio, rtol=0, atol=0)
