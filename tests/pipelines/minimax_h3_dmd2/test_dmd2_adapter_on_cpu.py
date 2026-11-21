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
"""The ``(MiniMaxH3Pipeline, dmd2)`` adapter on a tiny H3: timesteps, roles, conditioning and refused inputs."""

import pytest
import torch
from peft.tuners.tuners_utils import BaseTunerLayer

import verl_omni.pipelines
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
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import h3_dit_timestep
from verl_omni.pipelines.model_base import DiffusionModelBase


@pytest.fixture(autouse=True)
def _tiny_widths():
    with tiny_row_widths():
        yield


def _record_dit_calls(module) -> list[dict]:
    calls = []

    def record(_module, args, kwargs):
        lora_layers = [layer for layer in module.modules() if isinstance(layer, BaseTunerLayer)]
        disabled = bool(lora_layers) and all(layer.disable_adapters for layer in lora_layers)
        active = {layer.active_adapters[0] for layer in lora_layers}
        calls.append({**kwargs, "_adapter": None if disabled else (active.pop() if len(active) == 1 else active)})

    module.register_forward_pre_hook(record, with_kwargs=True)
    return calls


def _predict(module, data, sigmas, negative=False):
    latents = MiniMaxH3DMD2.dmd_initial_noise(data)
    condition = MiniMaxH3DMD2.dmd_condition(data)
    with torch.no_grad():
        return MiniMaxH3DMD2.dmd_predict(module, latents, sigmas, condition, negative)


def test_adapter_is_registered_for_h3_dmd2_through_the_pipelines_package():
    """Only the pipelines package's star import binds the adapter there.

    In production that import is the one that registers it.
    """
    assert verl_omni.pipelines.MiniMaxH3DMD2 is MiniMaxH3DMD2
    assert DiffusionModelBase.get_class_by_name(ARCHITECTURE, "dmd2") is MiniMaxH3DMD2


def test_predictions_are_flow_match_velocities_of_the_dit_output():
    """The DiT predicts x0 - z per row; DMD2Program reads z - x0, so each sample's output is the DiT's negated."""
    module = tiny_h3()
    dit_outputs = []
    module.register_forward_hook(lambda _module, _args, output: dit_outputs.append(output))
    sigmas = {"video": torch.tensor([0.9, 0.3]), "audio": torch.tensor([0.4, 0.75])}

    prediction = _predict(module, micro_batch(), sigmas)

    assert len(dit_outputs) == 2
    for index, (video_output, audio_output) in enumerate(dit_outputs):
        torch.testing.assert_close(prediction["video"][index : index + 1], -video_output, rtol=0, atol=0)
        torch.testing.assert_close(prediction["audio"][index : index + 1], -audio_output, rtol=0, atol=0)


def test_initial_noise_is_standard_gaussian_fp32_rows_of_the_row_geometry():
    """Video then audio rows of the microbatch's latent geometry, drawn from the global RNG."""
    data = micro_batch()
    torch.manual_seed(5)
    noise = MiniMaxH3DMD2.dmd_initial_noise(data)
    torch.manual_seed(5)
    expected_video = torch.randn(2, LATENT_META[0], TINY_VIDEO_ROW_WIDTH, dtype=torch.float32)
    expected_audio = torch.randn(2, LATENT_META[1], TINY_AUDIO_ROW_WIDTH, dtype=torch.float32)

    assert noise.keys() == {"video", "audio"}
    torch.testing.assert_close(noise["video"], expected_video, rtol=0, atol=0)
    torch.testing.assert_close(noise["audio"], expected_audio, rtol=0, atol=0)


def test_video_and_audio_sigmas_reach_the_dit_separately_per_sample():
    """Every target video row runs at t(a) and every audio row at t(b), t from ``h3_dit_timestep``."""
    module = tiny_h3()
    calls = _record_dit_calls(module)
    sigmas = {"video": torch.tensor([0.9, 0.3]), "audio": torch.tensor([0.4, 0.75])}

    _predict(module, micro_batch(), sigmas)

    seen = set()
    for call in calls:
        row_timesteps = call["timestep"][call["timestep_indices"]]
        video_t = row_timesteps[call["video_indices"]].unique()
        audio_t = row_timesteps[call["audio_indices"]].unique()
        assert video_t.numel() == 1 and audio_t.numel() == 1
        seen.add((video_t.item(), audio_t.item()))
    expected = {
        (h3_dit_timestep(sigmas["video"][index] * 1000).item(), h3_dit_timestep(sigmas["audio"][index] * 1000).item())
        for index in range(2)
    }
    assert seen == expected


def test_each_role_runs_under_its_adapter_and_the_teacher_is_the_base_model():
    """Student under ``default``, fake_score under ``fake_score``, teacher with adapters disabled."""
    module = tiny_h3(ADAPTERS)
    config = dmd_config(
        modality_shifts={"video": 12.0, "audio": 3.0},
        modality_weights={"video": 1.0, "audio": 1.0},
        critic_time_lead_modality="video",
    )
    engine = make_dmd_engine(module, config, architecture=ARCHITECTURE)
    calls = _record_dit_calls(module)
    data = micro_batch()
    latents = MiniMaxH3DMD2.dmd_initial_noise(data)
    sigmas = {"video": torch.tensor([0.9, 0.3]), "audio": torch.tensor([0.4, 0.75])}

    outputs, adapters = {}, {}
    with torch.no_grad(), engine._conditioned(MiniMaxH3DMD2.dmd_condition(data)):
        for role in ("student", "fake_score", "teacher"):
            first = len(calls)
            outputs[role] = engine._predict(role, latents, sigmas, False)
            adapters[role] = {call["_adapter"] for call in calls[first:]}
    assert adapters == {"student": {"default"}, "fake_score": {"fake_score"}, "teacher": {None}}

    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if ".lora_B." in name:
                parameter.zero_()
        base = MiniMaxH3DMD2.dmd_predict(module, latents, sigmas, MiniMaxH3DMD2.dmd_condition(data), False)
    for name in latents:
        torch.testing.assert_close(outputs["teacher"][name], base[name], rtol=0, atol=0)
        assert not torch.equal(outputs["student"][name], base[name])
        assert not torch.equal(outputs["fake_score"][name], outputs["student"][name])


def test_text_conditioning_comes_from_the_row_prompt_embeddings_and_token_tags():
    """The DiT reads each row's own prompt embeddings, cut to its length, and its token tags."""
    module = tiny_h3()
    calls = _record_dit_calls(module)
    data = micro_batch(text_lengths=(3, 2))
    data["prompt_token_tags"][1, 0] = 0

    _predict(module, data, {"video": torch.tensor([0.5, 0.5]), "audio": torch.tensor([0.5, 0.5])})

    assert len(calls) == 2
    for index, call in enumerate(calls):
        length = int(data["prompt_embeds_mask"][index].sum())
        torch.testing.assert_close(call["encoder_hidden_states"], data["prompt_embeds"][index : index + 1, :length])
        text_tags = call["token_tags"][call["text_indices"]]
        assert text_tags.tolist() == data["prompt_token_tags"][index, :length].tolist()


@pytest.mark.parametrize("missing", ["prompt_embeds", "prompt_embeds_mask", "prompt_token_tags", "latent_meta"])
def test_a_microbatch_without_its_conditioning_is_refused_by_name(missing):
    """No text encoder runs here, so a row without its precomputed conditioning is an error."""
    data = micro_batch()
    del data[missing]
    with pytest.raises(KeyError, match=f"precomputed text conditioning.*'{missing}'"):
        MiniMaxH3DMD2.dmd_condition(data)


def test_a_microbatch_of_two_latent_geometries_is_refused():
    """Rows of one microbatch share one packed layout and one noise shape."""
    data = micro_batch()
    data["latent_meta"][1, 2] = 2
    with pytest.raises(ValueError, match="one latent geometry per microbatch"):
        MiniMaxH3DMD2.dmd_initial_noise(data)


@pytest.mark.parametrize(
    "condition_key",
    ["condition_video_rows", "condition_audio_rows", "keyframe_frame_indices", "ref_block_meta"],
)
def test_a_microbatch_with_conditioning_media_is_refused_by_name(condition_key):
    """DMD here is text-to-video-and-audio only; I2V, FL2VA anchors and references are refused."""
    data = micro_batch()
    data[condition_key] = torch.zeros(len(data), 1)
    with pytest.raises(ValueError, match=condition_key):
        MiniMaxH3DMD2.dmd_condition(data)


def test_teacher_guidance_is_refused():
    """No unconditional embeddings reach the adapter, so a negative prediction is an error, not a silent copy."""
    module = tiny_h3()
    with pytest.raises(NotImplementedError, match="teacher_guidance_scale"):
        _predict(module, micro_batch(), {"video": torch.tensor([0.5, 0.5]), "audio": torch.tensor([0.5, 0.5])}, True)
