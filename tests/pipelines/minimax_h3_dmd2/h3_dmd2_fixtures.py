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
"""A tiny MiniMax H3 and t2va DMD microbatches for it, for the ``(MiniMaxH3Pipeline, dmd2)`` adapter."""

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

import torch
from peft import LoraConfig
from tensordict import TensorDict

from tests.model_merger.model_fixtures import tiny_transformer
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import H3_LORA_TARGETS, TEXT_TAG
from verl_omni.pipelines.minimax_h3_dmd2 import diffusers_training_adapter
from verl_omni.pipelines.minimax_h3_dmd2.diffusers_training_adapter import MiniMaxH3DMD2

ARCHITECTURE = "MiniMaxH3Pipeline"
ADAPTERS = ("default", "fake_score")
TEXT_DIM = 16  # the tiny H3's text_dim
# 4 video rows (1 latent frame of 4x4 with 2x2 patches), 6 audio rows (3 audio latents of 2 rows each)
LATENT_META = (4, 6, 1, 4, 4, 3)
TINY_VIDEO_ROW_WIDTH = 16  # in_channels 4 x patch 1x2x2
TINY_AUDIO_ROW_WIDTH = 4  # audio_in_channels


def tiny_h3(adapters: tuple[str, ...] = ()) -> torch.nn.Module:
    """The tiny H3 transformer on native attention, with a LoRA on the H3 targets per name in ``adapters``.

    Each adapter's ``lora_B`` is random and distinct, so each adapter changes the output.
    """
    module = tiny_transformer(ARCHITECTURE)
    module.set_attention_backend("native")
    lora = LoraConfig(r=4, lora_alpha=4, target_modules=sorted(H3_LORA_TARGETS))
    for index, name in enumerate(adapters):
        module.add_adapter(lora, adapter_name=name)
        generator = torch.Generator().manual_seed(100 + index)
        with torch.no_grad():
            for parameter_name, parameter in module.named_parameters():
                if f".lora_B.{name}." in parameter_name:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    if adapters:
        module.set_adapter(adapters[0])
    module.eval()
    return module


def micro_batch(text_lengths: tuple[int, ...] = (3, 2), seed: int = 0) -> TensorDict:
    """A t2va DMD microbatch: one row per entry of ``text_lengths``, prompt embeddings padded to the longest."""
    generator = torch.Generator().manual_seed(seed)
    batch, longest = len(text_lengths), max(text_lengths)
    mask = torch.tensor([[index < length for index in range(longest)] for length in text_lengths])
    return TensorDict(
        {
            "prompt_embeds": torch.randn(batch, longest, TEXT_DIM, generator=generator),
            "prompt_embeds_mask": mask,
            "prompt_token_tags": torch.full((batch, longest), TEXT_TAG, dtype=torch.long),
            "latent_meta": torch.tensor(LATENT_META).repeat(batch, 1),
        },
        batch_size=[batch],
    )


@contextmanager
def tiny_row_widths() -> Iterator[None]:
    """Give the adapter the tiny H3's row widths in place of the full model's."""
    with (
        patch.object(diffusers_training_adapter, "VIDEO_ROW_WIDTH", TINY_VIDEO_ROW_WIDTH),
        patch.object(diffusers_training_adapter, "AUDIO_ROW_WIDTH", TINY_AUDIO_ROW_WIDTH),
    ):
        yield


__all__ = [
    "ADAPTERS",
    "ARCHITECTURE",
    "LATENT_META",
    "TINY_AUDIO_ROW_WIDTH",
    "TINY_VIDEO_ROW_WIDTH",
    "MiniMaxH3DMD2",
    "micro_batch",
    "tiny_h3",
    "tiny_row_widths",
]
