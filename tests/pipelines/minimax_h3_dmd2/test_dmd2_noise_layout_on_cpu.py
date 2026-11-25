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
"""Every DMD noise draw of the H3 engine is a latent-layout draw packed into rows, as the PDMD reference draws it.

https://github.com/ZeamoxWang/pdmd at 6b6e106 draws ``(1, C, T, H, W)`` video and ``(A, D, T_audio)`` audio
latents (``src/pdmd/backend.py:56-62``, ``training/train.py:213-214,266-268``) and packs them into DiT rows.
"""

import torch
from tensordict import TensorDict
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_pack_audio_latent,
    minimax_h3_patchify_video_latent,
)

from tests.pipelines.minimax_h3_dmd2.h3_dmd2_fixtures import (
    ADAPTERS,
    ARCHITECTURE,
    LATENT_META,
    TINY_AUDIO_ROW_WIDTH,
    TINY_VIDEO_ROW_WIDTH,
    MiniMaxH3DMD2,
    micro_batch,
    tiny_h3,
    tiny_row_widths,
)
from tests.workers.engine.dmd_engine_fixtures import dmd_config, make_dmd_engine
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import AUDIO_ROW_WIDTH, VIDEO_ROW_WIDTH
from verl_omni.pipelines.minimax_h3_dmd2.diffusers_training_adapter import H3TextCondition
from verl_omni.utils.dataset.prompt_stream import stream_seed

ENGINE_SEED = 3  # make_dmd_engine's engine seed
UPDATE = 4
SAMPLES = (5, 9)
PATCH = (1, 2, 2)
NAMES = ("noise", "rollout_renoise", "renoise", "score_query")
# full H3 widths on a non-square geometry, so an H/W swap moves rows: 2 latent frames of 4x6, 3 audio latents
FULL_META = (2 * 2 * 3, 2 * 3, 2, 4, 6, 3)


def latent_pairs(seed: int, count: int, meta, video_row_width: int, audio_row_width: int, swap_hw: bool = False):
    """The first ``count`` (video rows, audio rows) pairs pdmd draws in the latent layout from one generator."""
    _, audio_rows, frames, height, width, audio_t = meta
    channels = video_row_width // (PATCH[0] * PATCH[1] * PATCH[2])
    generator = torch.Generator().manual_seed(seed)
    pairs = []
    for _ in range(count):
        shape = (1, channels, frames, width, height) if swap_hw else (1, channels, frames, height, width)
        video = torch.randn(shape, generator=generator)
        if swap_hw:
            video = video.transpose(3, 4)
        audio = torch.randn((audio_rows // audio_t, audio_row_width, audio_t), generator=generator)
        pairs.append((minimax_h3_patchify_video_latent(video, patch_size=PATCH), minimax_h3_pack_audio_latent(audio)))
    return pairs


def packed_pairs(seed: int, count: int, meta, video_row_width: int, audio_row_width: int):
    """The control: the same numbers drawn straight in the packed row shape."""
    generator = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(meta[0], video_row_width, generator=generator),
            torch.randn(meta[1], audio_row_width, generator=generator),
        )
        for _ in range(count)
    ]


def equal_pair(noise: dict[str, torch.Tensor], row: int, pair) -> bool:
    return torch.equal(noise["video"][row], pair[0]) and torch.equal(noise["audio"][row], pair[1])


def _engine(**overrides):
    config = dmd_config(
        modality_shifts={"video": 12.0, "audio": 3.0},
        modality_weights={"video": 1.0, "audio": 1.0},
        critic_time_lead_modality="video",
        **overrides,
    )
    return make_dmd_engine(tiny_h3(ADAPTERS), config, architecture=ARCHITECTURE)


def test_every_draw_of_an_update_is_the_packed_latent_draw_of_its_key():
    """Catches noise drawn straight in the packed shape, a re-noise generator reseeded per step, or one name
    left on the old draw: each recorded row must be the k-th latent pair of its own key."""
    steps = 3
    with tiny_row_widths():
        engine = _engine(num_inference_steps=steps, sampler="renoise", critic_noise="fresh", score_query_noise="fresh")
        recorded = []
        draw_noise = engine._program.draw_noise

        def recording(generators, like):
            noise = draw_noise(generators, like)
            recorded.append(([generator.initial_seed() for generator in generators], noise))
            return noise

        engine._program.draw_noise = recording
        keys = engine._program.keys(UPDATE, SAMPLES)
        with torch.no_grad():
            ((trajectory, condition, _),) = engine._roll_out([micro_batch()], [SAMPLES], UPDATE)
            with engine._conditioned(condition):
                engine._program.stage_outputs(trajectory, "fake_score", keys=keys)
                engine._program.stage_outputs(trajectory, "student", 1, keys=keys)

    key_of = {stream_seed(ENGINE_SEED, name, UPDATE, sample): (name, sample) for name in NAMES for sample in SAMPLES}
    widths = (TINY_VIDEO_ROW_WIDTH, TINY_AUDIO_ROW_WIDTH)
    for row, sample in enumerate(SAMPLES):
        seed = stream_seed(ENGINE_SEED, "noise", UPDATE, sample)
        (pair,) = latent_pairs(seed, 1, LATENT_META, *widths)
        assert equal_pair(trajectory.states[0], row, pair), f"initial noise of sample {sample}"
        (control,) = packed_pairs(seed, 1, LATENT_META, *widths)
        assert not equal_pair(trajectory.states[0], row, control)

    drawn = {}
    for seeds, noise in recorded:
        for row, seed in enumerate(seeds):
            name, sample = key_of[seed]
            index = drawn.get((name, sample), 0)
            drawn[(name, sample)] = index + 1
            pair = latent_pairs(seed, index + 1, LATENT_META, *widths)[index]
            assert equal_pair(noise, row, pair), f"{name} draw {index} of sample {sample}"
            control = packed_pairs(seed, index + 1, LATENT_META, *widths)[index]
            assert not equal_pair(noise, row, control)
    assert drawn == {
        **{("rollout_renoise", sample): steps for sample in SAMPLES},
        **{("renoise", sample): 1 for sample in SAMPLES},
        **{("score_query", sample): 1 for sample in SAMPLES},
    }


def test_the_engine_drawer_uses_the_full_h3_latent_shapes_on_a_non_square_geometry():
    """Catches a wrong channel count or an H/W swap in the latent draw: (1, 24, T, H, W) video, (2, 32, T) audio."""
    engine = _engine()
    condition = H3TextCondition(prompt_embeds=[], token_tags=[], latent_meta=FULL_META)
    like = {
        "video": torch.zeros(2, FULL_META[0], VIDEO_ROW_WIDTH),
        "audio": torch.zeros(2, FULL_META[1], AUDIO_ROW_WIDTH),
    }
    widths = (VIDEO_ROW_WIDTH, AUDIO_ROW_WIDTH)
    keys = engine._program.keys(UPDATE, SAMPLES)
    data = TensorDict({"latent_meta": torch.tensor(FULL_META).repeat(2, 1)}, batch_size=[2])
    for name in NAMES:
        if name == "noise":
            noise = MiniMaxH3DMD2.dmd_initial_noise(data, keys.generators(name, "cpu"))
        else:
            with engine._conditioned(condition):
                noise = engine._program.draw_noise(keys.generators(name, "cpu"), like)
        for row, sample in enumerate(SAMPLES):
            seed = stream_seed(ENGINE_SEED, name, UPDATE, sample)
            (pair,) = latent_pairs(seed, 1, FULL_META, *widths)
            assert equal_pair(noise, row, pair), f"{name} of sample {sample}"
            (swapped,) = latent_pairs(seed, 1, FULL_META, *widths, swap_hw=True)
            assert not equal_pair(noise, row, swapped)
            (packed,) = packed_pairs(seed, 1, FULL_META, *widths)
            assert not equal_pair(noise, row, packed)
