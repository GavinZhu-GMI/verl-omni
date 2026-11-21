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
"""MiniMax H3 training adapter for DMD2 / PDMD distillation, text to video and audio."""

from dataclasses import dataclass

import torch
from diffusers import ModelMixin
from tensordict import TensorDict

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    AUDIO_ROW_WIDTH,
    VIDEO_ROW_WIDTH,
    build_layout_from_meta,
    h3_dit_timestep,
    h3_sample_model_inputs,
    h3_target_velocities,
    h3_ulysses_forward,
)
from verl_omni.pipelines.model_base import DiffusionModelBase

__all__ = ["MiniMaxH3DMD2"]

_CONDITION_KEYS = ("prompt_embeds", "prompt_embeds_mask", "prompt_token_tags", "latent_meta")
_MEDIA_CONDITION_KEYS = ("condition_video_rows", "condition_audio_rows", "keyframe_frame_indices", "ref_block_meta")


@dataclass
class H3TextCondition:
    """The text conditioning of one t2va microbatch and its shared latent geometry.

    ``prompt_embeds`` holds one ``(L_i, D)`` tensor per sample and ``token_tags`` one ``(L_i,)`` tensor;
    ``latent_meta`` is H3's six-int target geometry (video rows, audio rows, latent T, H, W, audio T).
    """

    prompt_embeds: list[torch.Tensor]
    token_tags: list[torch.Tensor]
    latent_meta: tuple[int, ...]


def _latent_meta(micro_batch: TensorDict) -> tuple[int, ...]:
    latent_meta = micro_batch["latent_meta"]
    if not torch.all(latent_meta == latent_meta[0]):
        raise ValueError("MiniMax H3 DMD needs one latent geometry per microbatch; bucket the rows by latent_meta.")
    return tuple(int(value) for value in latent_meta[0].tolist())


@DiffusionModelBase.register("MiniMaxH3Pipeline", algorithm="dmd2")
class MiniMaxH3DMD2(DiffusionModelBase):
    """Predict H3 video and audio velocities for the DMD engine, each modality at its own per-sample sigma.

    Rows carry precomputed prompt embeddings and token tags, so the trainer loads no text encoder. Latents are
    ``{"video": (B, video_rows, VIDEO_ROW_WIDTH), "audio": (B, audio_rows, AUDIO_ROW_WIDTH)}``.
    """

    @classmethod
    def dmd_condition(cls, micro_batch: TensorDict) -> H3TextCondition:
        """Read each row's prompt embeddings and token tags, cut to its prompt length.

        Raises:
            KeyError: A conditioning key is missing.
            ValueError: The microbatch carries conditioning media, or rows of different latent geometry.
        """
        missing = [key for key in _CONDITION_KEYS if key not in micro_batch.keys()]
        if missing:
            raise KeyError(f"MiniMax H3 DMD needs precomputed text conditioning; the microbatch lacks {missing}.")
        media = [key for key in _MEDIA_CONDITION_KEYS if key in micro_batch.keys()]
        if media:
            raise ValueError(
                f"MiniMax H3 DMD trains text to video and audio only; the microbatch carries conditioning {media}."
            )
        latent_meta = _latent_meta(micro_batch)
        lengths = micro_batch["prompt_embeds_mask"].sum(dim=-1).tolist()  # one host sync per microbatch
        return H3TextCondition(
            prompt_embeds=[micro_batch["prompt_embeds"][index, :length] for index, length in enumerate(lengths)],
            token_tags=[micro_batch["prompt_token_tags"][index, :length] for index, length in enumerate(lengths)],
            latent_meta=latent_meta,
        )

    @classmethod
    def dmd_initial_noise(cls, micro_batch: TensorDict) -> dict[str, torch.Tensor]:
        """Standard Gaussian FP32 video and audio rows of the microbatch's latent geometry, on its device."""
        latent_meta = _latent_meta(micro_batch)
        batch, device = len(micro_batch), micro_batch["latent_meta"].device
        return {
            "video": torch.randn(batch, latent_meta[0], VIDEO_ROW_WIDTH, device=device, dtype=torch.float32),
            "audio": torch.randn(batch, latent_meta[1], AUDIO_ROW_WIDTH, device=device, dtype=torch.float32),
        }

    @classmethod
    def dmd_predict(
        cls,
        module: ModelMixin,
        latents: dict[str, torch.Tensor],
        sigmas: dict[str, torch.Tensor],
        condition: H3TextCondition,
        negative: bool,
    ) -> dict[str, torch.Tensor]:
        """Run the DiT once per sample, video at t(sigma_video) and audio at t(sigma_audio), t = 1 - sigma.

        Returns velocities ``z - x0`` with the latents' keys and shapes.

        Raises:
            NotImplementedError: ``negative`` is set; rows carry no unconditional prompt embeddings.
        """
        if negative:
            raise NotImplementedError(
                "MiniMax H3 DMD rows carry no unconditional prompt embeddings; set teacher_guidance_scale=1."
            )
        video, audio = latents["video"], latents["audio"]
        patch_t, patch_h, patch_w = (int(value) for value in module.config["patch_size"])
        video_timesteps = h3_dit_timestep(sigmas["video"].float() * 1000).tolist()
        audio_timesteps = h3_dit_timestep(sigmas["audio"].float() * 1000).tolist()
        video_velocities, audio_velocities = [], []
        for index, (prompt_embeds, token_tags) in enumerate(
            zip(condition.prompt_embeds, condition.token_tags, strict=True)
        ):
            layout = build_layout_from_meta(
                condition.latent_meta, prompt_embeds.shape[0], (patch_t, patch_h, patch_w), text_token_tags=token_tags
            )
            video_t, audio_t = video_timesteps[index], audio_timesteps[index]
            sample_inputs = h3_sample_model_inputs(
                layout,
                video[index : index + 1],
                audio[index : index + 1],
                prompt_embeds.unsqueeze(0),
                video_timestep=video_t,
                audio_timestep=audio_t,
                condition_video_timestep=video_t,  # t2va has no condition rows
                condition_audio_timestep=audio_t,
            )
            video_velocity, audio_velocity = h3_target_velocities(h3_ulysses_forward(module, sample_inputs), layout)
            video_velocities.append(video_velocity)
            audio_velocities.append(audio_velocity)
        return {"video": torch.cat(video_velocities), "audio": torch.cat(audio_velocities)}
