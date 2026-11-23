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

"""DMD computation on independent CPU models, without a production engine."""

import copy
from functools import partial

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.dmd2.program import DMD2Program, Trajectory
from verl_omni.trainer.diffusion.distillation.utils import dmd_gradient, dmd_surrogate_loss
from verl_omni.workers.config.diffusion.actor import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.config.diffusion.dmd import DiffusionDMDConfig
from verl_omni.workers.utils.losses import diffusion_loss


def make_program(modalities=("image",), *, device="cpu", batch=2, **overrides):
    config = DiffusionDMDConfig(
        **{
            "modality_shifts": {name: float(index + 2) for index, name in enumerate(modalities)},
            "modality_weights": {name: 0.7 + index for index, name in enumerate(modalities)},
            "score_discrete_steps": 0,
            "cfg_norm": "none",
            "teacher_guidance_scale": 1.0,
            "critic_time_lead_modality": modalities[0] if len(modalities) > 1 else None,
            **overrides,
        }
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        # Teacher parameters also require grad so a missing no_grad is observable.
        models = {
            role: torch.nn.ModuleDict({name: torch.nn.Linear(3, 3) for name in modalities})
            for role in ("student", "fake_score", "teacher")
        }
    models = {role: model.to(device) for role, model in models.items()}
    calls = []

    def predict(role, latents, sigmas, negative):
        calls.append(
            {
                "role": role,
                "negative": negative,
                "grad": torch.is_grad_enabled(),
                "latents": {name: value.detach().clone() for name, value in latents.items()},
                "sigmas": {name: value.clone() for name, value in sigmas.items()},
            }
        )
        return {
            name: models[role][name](value)
            + (0.2 if negative else 0.4) * sigmas[name].view(-1, *([1] * (value.ndim - 1)))
            for name, value in latents.items()
        }

    initial = {
        name: torch.linspace(-1, 1, batch * (index + 2) * 3).reshape(batch, index + 2, 3).to(device)
        for index, name in enumerate(modalities)
    }
    return DMD2Program(config, predict, seed=42), models, initial, calls


def keys_of(program, latents, update=0):
    """The draw keys of global samples ``0 .. B-1`` of ``update``."""
    return program.keys(update, tuple(range(next(iter(latents.values())).shape[0])))


def loss_inputs(accumulation=1, sp_size=1):
    data = TensorDict({}, batch_size=[2])
    tu.assign_non_tensor(data, gradient_accumulation_steps=accumulation, sp_size=sp_size)
    actor = DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=DiffusionLossConfig(loss_mode="dmd2"))
    return data, partial(diffusion_loss, config=actor)


def linear_program(sampler, *, knots=None, steps=4, shift=2.0):
    """One image modality whose student velocity is 0.5 * latents + 0.1 * sigma."""
    config = DiffusionDMDConfig(
        num_inference_steps=steps,
        modality_shifts={"image": shift},
        modality_weights={"image": 1.0},
        score_discrete_steps=0,
        cfg_norm="none",
        teacher_guidance_scale=1.0,
        sampler=sampler,
        base_time_knots=knots,
    )
    calls = []

    def predict(role, latents, sigmas, negative):
        calls.append((role, sigmas["image"].clone(), torch.is_grad_enabled()))
        sigma = sigmas["image"].view(-1, *([1] * (latents["image"].ndim - 1)))
        return {"image": 0.5 * latents["image"] + 0.1 * sigma}

    return DMD2Program(config, predict, seed=3), calls


# shift 2 on linspace(1, 0, 5): 2t / (1 + t)
SHIFT2_GRID = [1.0, 6 / 7, 2 / 3, 0.4, 0.0]


def test_euler_rollout_keeps_every_state_and_follows_the_shifted_grid():
    program, calls = linear_program("euler")
    initial = {"image": torch.tensor([[1.0, -2.0], [0.5, 0.25]], dtype=torch.bfloat16)}
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    assert isinstance(trajectory, Trajectory)
    torch.testing.assert_close(trajectory.sigmas["image"], torch.tensor(SHIFT2_GRID))
    expected = initial["image"].float()
    torch.testing.assert_close(trajectory.states[0]["image"], expected, rtol=0, atol=0)
    for index in range(4):
        velocity = 0.5 * expected + 0.1 * SHIFT2_GRID[index]
        expected = expected + (SHIFT2_GRID[index + 1] - SHIFT2_GRID[index]) * velocity
        torch.testing.assert_close(trajectory.states[index + 1]["image"], expected)
    assert len(trajectory.states) == 5
    assert all(state["image"].dtype == torch.float32 for state in trajectory.states)
    # one student call per step, no autograd, per-sample sigmas
    assert [role for role, _, _ in calls] == ["student"] * 4
    assert not any(enabled for _, _, enabled in calls)
    for index, (_, sigma, _) in enumerate(calls):
        torch.testing.assert_close(sigma, torch.full((2,), SHIFT2_GRID[index]))


def test_each_modality_rolls_out_on_its_own_grid():
    program, _, initial, calls = make_program(("video", "audio"))
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    # shift 2 and shift 3 on linspace(1, 0, 5)
    torch.testing.assert_close(trajectory.sigmas["video"], torch.tensor(SHIFT2_GRID))
    torch.testing.assert_close(trajectory.sigmas["audio"], torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0]))
    for index, call in enumerate(calls):
        for name in ("video", "audio"):
            torch.testing.assert_close(call["sigmas"][name], trajectory.sigmas[name][index].repeat(2))


def test_renoise_rollout_draws_fresh_noise_at_each_inner_knot():
    program, _ = linear_program("renoise")
    initial = {"image": torch.tensor([[1.0, -2.0, 0.5]])}
    trajectory = program.rollout(initial, keys=program.keys(11, (5,)))
    (generator,) = program.keys(11, (5,)).generators("rollout_renoise", "cpu")
    expected = initial["image"]
    for index in range(4):
        clean = expected - SHIFT2_GRID[index] * (0.5 * expected + 0.1 * SHIFT2_GRID[index])
        if index < 3:
            noise = torch.randn(3, generator=generator).unsqueeze(0)
            expected = (1 - SHIFT2_GRID[index + 1]) * clean + SHIFT2_GRID[index + 1] * noise
        else:
            expected = clean
        torch.testing.assert_close(trajectory.states[index + 1]["image"], expected)


def test_explicit_knots_are_shifted_per_modality():
    program, _ = linear_program("renoise", knots=[1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0], steps=6, shift=12.0)
    initial = {"image": torch.zeros(1, 2)}
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    # knots 1000/750/500/250/100/40/0 (on a 1000 scale) through shift 12
    torch.testing.assert_close(
        trajectory.sigmas["image"], torch.tensor([1.0, 0.973, 0.923, 0.8, 0.571, 0.333, 0.0]), rtol=0, atol=5e-4
    )
    assert len(trajectory.states) == 7


def test_rollout_builds_no_graph_and_sample_returns_its_endpoint():
    program, models, initial, calls = make_program(("image", "audio"))
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    assert not any(value.requires_grad for step in trajectory.states for value in step.values())
    assert all(parameter.grad is None for parameter in models["student"].parameters())
    sampled = program.sample(initial, keys=keys_of(program, initial))
    for name in initial:
        torch.testing.assert_close(sampled[name], trajectory.states[-1][name], rtol=0, atol=0)
    assert {call["role"] for call in calls} == {"student"}


def fake_score_call(calls):
    (call,) = [call for call in calls if call["role"] == "fake_score"]
    return call


@pytest.mark.parametrize("modalities", [("image",), ("video", "audio")])
def test_fake_score_regresses_on_the_rollout_endpoint_with_fresh_noise(modalities):
    program, models, initial, calls = make_program(modalities)
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    calls.clear()
    outputs = program.stage_outputs(trajectory, "fake_score", keys=keys_of(program, trajectory.states[0]))
    call = fake_score_call(calls)
    assert [c["role"] for c in calls] == ["fake_score"] and call["grad"]
    for name in modalities:
        endpoint, noise = trajectory.states[-1][name], outputs[name]["noise"]
        sigma = call["sigmas"][name]
        assert sigma.shape == (2,) and sigma.dtype == torch.float32
        torch.testing.assert_close(outputs[name]["generated_x0"], endpoint, rtol=0, atol=0)
        torch.testing.assert_close(
            call["latents"][name], (1 - sigma.view(-1, 1, 1)) * endpoint + sigma.view(-1, 1, 1) * noise
        )
        assert not torch.allclose(noise, trajectory.states[0][name])
        assert outputs[name]["noise_pred"].requires_grad
    assert all(parameter.grad is None for parameter in models["student"].parameters())


def test_trajectory_critic_noise_is_the_rollout_initial_noise():
    program, _, initial, _ = make_program(("video", "audio"), critic_noise="trajectory")
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    outputs = program.stage_outputs(trajectory, "fake_score", keys=keys_of(program, trajectory.states[0]))
    for name in ("video", "audio"):
        torch.testing.assert_close(outputs[name]["noise"], trajectory.states[0][name], rtol=0, atol=0)


@pytest.mark.parametrize("modalities", [("video", "audio"), ("audio", "video")])
def test_uniform_critic_time_follows_the_lead_modality_rule(modalities):
    """PDMD reference rule: video ~ U(0, 1); audio ~ U(0, 1), redrawn from U(0.85, 1) when video > 0.95."""
    program, _, initial, calls = make_program(modalities, batch=20000, critic_time_lead_modality="video")
    program.stage_outputs(
        program.rollout(initial, keys=keys_of(program, initial)), "fake_score", keys=keys_of(program, initial)
    )
    sigmas = fake_score_call(calls)["sigmas"]
    lead, follower = sigmas["video"], sigmas["audio"]
    assert bool(((lead >= 0) & (lead <= 1) & (follower >= 0) & (follower <= 1)).all())
    high = lead > 0.95
    assert bool((follower[high] >= 0.85).all())
    assert abs(float(follower[high].mean()) - 0.925) < 0.01
    # below the threshold the follower is independent of the lead and uniform
    low_lead, low_follower = lead[~high], follower[~high]
    correlation = torch.corrcoef(torch.stack([low_lead, low_follower]))[0, 1]
    assert abs(float(correlation)) < 0.03
    assert abs(float(low_follower.mean()) - 0.5) < 0.01 and abs(float(lead.mean()) - 0.5) < 0.01


def test_query_matched_critic_time_shares_one_step_and_ratio_across_modalities():
    program, _, initial, calls = make_program(("video", "audio"), batch=500, critic_time="query_matched")
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    program.stage_outputs(trajectory, "fake_score", keys=keys_of(program, trajectory.states[0]))
    sigmas = fake_score_call(calls)["sigmas"]
    video_grid, audio_grid = trajectory.sigmas["video"], trajectory.sigmas["audio"]
    for sample in range(500):
        step = int(
            ((video_grid[1:] <= sigmas["video"][sample]) & (sigmas["video"][sample] <= video_grid[:-1])).nonzero()[0]
        )
        ratios = [
            (sigmas[name][sample] - grid[step + 1]) / (grid[step] - grid[step + 1])
            for name, grid in (("video", video_grid), ("audio", audio_grid))
        ]
        torch.testing.assert_close(ratios[0], ratios[1], rtol=0, atol=1e-5)


def test_fake_score_stage_rejects_an_exit_index():
    program, _, initial, _ = make_program()
    with pytest.raises(ValueError, match="exit_index"):
        program.stage_outputs(
            program.rollout(initial, keys=keys_of(program, initial)),
            "fake_score",
            exit_index=0,
            keys=keys_of(program, initial),
        )


def by_role(calls, role, negative=False):
    return [call for call in calls if call["role"] == role and call["negative"] == negative]


def student_velocity(models, name, latents, sigma):
    """The student's velocity in make_program, recomputed from its weights."""
    return models["student"][name](latents) + 0.4 * sigma


@pytest.mark.parametrize("sampler", ["euler", "renoise"])
@pytest.mark.parametrize("exit_index", range(4))
def test_student_reforwards_once_at_the_exit_state_and_trains_only_itself(exit_index, sampler):
    program, models, initial, calls = make_program(("video", "audio"), sampler=sampler)
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    calls.clear()
    outputs = program.stage_outputs(trajectory, "student", exit_index, keys=keys_of(program, trajectory.states[0]))
    assert [(call["role"], call["grad"]) for call in calls] == [
        ("student", True),
        ("fake_score", False),
        ("teacher", False),
    ]
    student = calls[0]
    loss = 0
    for name in ("video", "audio"):
        start, sigma = trajectory.states[exit_index][name], trajectory.sigmas[name][exit_index]
        torch.testing.assert_close(student["latents"][name], start, rtol=0, atol=0)
        torch.testing.assert_close(student["sigmas"][name], sigma.expand(2), rtol=0, atol=0)
        with torch.no_grad():
            expected = start - sigma * student_velocity(models, name, start, sigma)
        torch.testing.assert_close(outputs[name]["generated_x0"].detach(), expected)
        assert not outputs[name]["fake_x0"].requires_grad and not outputs[name]["teacher_x0"].requires_grad
        loss = loss + outputs[name]["generated_x0"].sum()
    loss.backward()
    assert all(parameter.grad is not None for parameter in models["student"].parameters())
    assert all(parameter.grad is None for role in ("fake_score", "teacher") for parameter in models[role].parameters())


def test_score_query_time_lies_below_the_exit_step_and_critic_and_teacher_see_one_input():
    exit_index = 1
    program, _, initial, calls = make_program(("video", "audio"), batch=300)
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    calls.clear()
    program.stage_outputs(trajectory, "student", exit_index, keys=keys_of(program, trajectory.states[0]))
    (fake,), (teacher,) = by_role(calls, "fake_score"), by_role(calls, "teacher")
    ratios = {}
    for name in ("video", "audio"):
        grid = trajectory.sigmas[name]
        sigma = fake["sigmas"][name]
        assert sigma.shape == (300,)
        assert bool(((grid[exit_index + 1] <= sigma) & (sigma <= grid[exit_index])).all())
        ratios[name] = (sigma - grid[exit_index + 1]) / (grid[exit_index] - grid[exit_index + 1])
        torch.testing.assert_close(teacher["sigmas"][name], sigma, rtol=0, atol=0)
        torch.testing.assert_close(teacher["latents"][name], fake["latents"][name], rtol=0, atol=0)
    torch.testing.assert_close(ratios["video"], ratios["audio"], rtol=0, atol=1e-5)
    assert float(ratios["video"].std()) > 0.2  # one ratio per sample, not per batch


def query_noise(outputs, calls, name):
    """Recover the noise the query used: query = (1 - s) x0 + s z."""
    (fake,) = by_role(calls, "fake_score")
    sigma = fake["sigmas"][name].view(-1, 1, 1)
    return (fake["latents"][name] - (1 - sigma) * outputs[name]["generated_x0"].detach()) / sigma


@pytest.mark.parametrize("exit_index", [0, 2])
def test_student_estimate_query_noise_is_the_students_own_noise_estimate(exit_index):
    program, models, initial, calls = make_program(("video", "audio"))
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    calls.clear()
    outputs = program.stage_outputs(trajectory, "student", exit_index, keys=keys_of(program, trajectory.states[0]))
    for name in ("video", "audio"):
        start, sigma = trajectory.states[exit_index][name], trajectory.sigmas[name][exit_index]
        with torch.no_grad():
            estimate = start + (1 - sigma) * student_velocity(models, name, start, sigma)
        torch.testing.assert_close(query_noise(outputs, calls, name), estimate, rtol=1e-4, atol=1e-4)
        if exit_index == 0:  # at sigma = 1 the estimate is the initial noise itself
            torch.testing.assert_close(estimate, trajectory.states[0][name])


def test_trajectory_and_fresh_query_noise():
    program, _, initial, calls = make_program(("video", "audio"), score_query_noise="trajectory")
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    calls.clear()
    outputs = program.stage_outputs(trajectory, "student", 2, keys=keys_of(program, trajectory.states[0]))
    for name in ("video", "audio"):
        torch.testing.assert_close(query_noise(outputs, calls, name), trajectory.states[0][name], rtol=1e-4, atol=1e-4)
    program, models, initial, calls = make_program(("video", "audio"), score_query_noise="fresh")
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    calls.clear()
    outputs = program.stage_outputs(trajectory, "student", 2, keys=keys_of(program, trajectory.states[0]))
    for name in ("video", "audio"):
        used = query_noise(outputs, calls, name)
        start, sigma = trajectory.states[2][name], trajectory.sigmas[name][2]
        with torch.no_grad():
            estimate = start + (1 - sigma) * student_velocity(models, name, start, sigma)
        assert not torch.allclose(used, trajectory.states[0][name], atol=1e-2)
        assert not torch.allclose(used, estimate, atol=1e-2)


def test_critic_and_teacher_endpoints_use_their_own_velocities_at_the_query():
    program, models, initial, calls = make_program()
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    calls.clear()
    outputs = program.stage_outputs(trajectory, "student", 3, keys=keys_of(program, trajectory.states[0]))
    (fake,) = by_role(calls, "fake_score")
    query, sigma = fake["latents"]["image"], fake["sigmas"]["image"].view(-1, 1, 1)
    with torch.no_grad():
        for role, key in (("fake_score", "fake_x0"), ("teacher", "teacher_x0")):
            velocity = models[role]["image"](query) + 0.4 * sigma
            torch.testing.assert_close(outputs["image"][key], query - sigma * velocity)


@pytest.mark.parametrize("guidance,norm", [(1.0, "none"), (4.0, "none"), (4.0, "layer_norm"), (4.0, "scalar")])
def test_teacher_cfg_uses_only_detached_teacher_predictions(guidance, norm):
    program, models, initial, calls = make_program(teacher_guidance_scale=guidance, cfg_norm=norm)
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    calls.clear()
    program.stage_outputs(trajectory, "student", 1, keys=keys_of(program, trajectory.states[0]))
    teacher_calls = [(call["negative"], call["grad"]) for call in calls if call["role"] == "teacher"]
    assert teacher_calls == [(False, False)] + ([(True, False)] if guidance != 1 else [])
    assert all(parameter.grad is None for parameter in models["teacher"].parameters())


def test_layer_norm_cfg_matches_packed_feature_axis_formula():
    program, _, initial, calls = make_program(teacher_guidance_scale=4.0, cfg_norm="layer_norm")
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    calls.clear()
    outputs = program.stage_outputs(trajectory, "student", 2, keys=keys_of(program, trajectory.states[0]))
    (fake,) = by_role(calls, "fake_score")
    query, sigma = fake["latents"]["image"], fake["sigmas"]["image"].view(-1, 1, 1)
    _, models, _, _ = make_program(teacher_guidance_scale=4.0, cfg_norm="layer_norm")
    with torch.no_grad():
        cond = models["teacher"]["image"](query) + 0.4 * sigma
        uncond = models["teacher"]["image"](query) + 0.2 * sigma
    guided = uncond + 4 * (cond - uncond)
    guided = guided * (cond.square().sum(-1, keepdim=True).sqrt() / guided.square().sum(-1, keepdim=True).sqrt())
    torch.testing.assert_close(outputs["image"]["teacher_x0"], query - sigma * guided)


@pytest.mark.parametrize("exit_index", [None, -1, 4, True, 1.0])
def test_student_stage_needs_a_valid_exit_index(exit_index):
    program, _, initial, calls = make_program()
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    calls.clear()
    with pytest.raises(ValueError, match="exit_index"):
        program.stage_outputs(trajectory, "student", exit_index, keys=keys_of(program, trajectory.states[0]))
    assert not calls


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4, 5)])
def test_layer_norm_cfg_rejects_unpacked_layout_before_any_forward(shape):
    program, _, _, calls = make_program(teacher_guidance_scale=4.0, cfg_norm="layer_norm")
    trajectory = Trajectory(states=[{"image": torch.ones(shape)}] * 5, sigmas={"image": torch.linspace(1, 0, 5)})
    with pytest.raises(ValueError, match="packed"):
        program.stage_outputs(trajectory, "student", 0, keys=keys_of(program, trajectory.states[0]))
    assert not calls


def test_unpacked_latents_remain_supported_without_layer_norm_cfg():
    program, _, _, _ = make_program(teacher_guidance_scale=4.0, cfg_norm="scalar")
    program.predict = lambda role, latents, sigmas, negative: {
        name: 0.5 * value + sigmas[name].view(-1, 1, 1, 1) for name, value in latents.items()
    }
    initial = {"image": torch.ones(2, 3, 4, 3)}
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    for stage, index in (("student", 2), ("fake_score", None)):
        outputs = program.stage_outputs(trajectory, stage, index, keys=keys_of(program, trajectory.states[0]))
        assert outputs["image"]["generated_x0"].shape == initial["image"].shape


def stage_args(stage):
    return {"exit_index": 1} if stage == "student" else {}


@pytest.mark.parametrize("stage", ["student", "fake_score"])
@pytest.mark.parametrize("modalities", [("image",), ("video", "audio")])
@pytest.mark.parametrize("accumulation,sp_size", [(1, 1), (3, 1), (2, 4)])
def test_loss_wrapper_scaling_and_gradients_match_hand_arithmetic(stage, modalities, accumulation, sp_size):
    """Stage outputs come from the program; the loss arithmetic on them is written out by hand."""
    actual, models, initial, _ = make_program(modalities, teacher_guidance_scale=4.0)
    reference, expected_models, _, _ = make_program(modalities, teacher_guidance_scale=4.0)
    trajectory = actual.rollout(initial, keys=keys_of(actual, initial))
    before = {role: copy.deepcopy(model.state_dict()) for role, model in models.items()}
    data, loss_function = loss_inputs(accumulation, sp_size)
    torch.manual_seed(107)
    actual_loss, metrics, invalid = actual.compute_loss(
        trajectory, stage, data, loss_function, **stage_args(stage), keys=keys_of(actual, trajectory.states[0])
    )
    torch.manual_seed(107)
    outputs = reference.stage_outputs(
        trajectory, stage, **stage_args(stage), keys=keys_of(reference, trajectory.states[0])
    )
    expected_loss = 0
    for name, output in outputs.items():
        generated = output["generated_x0"]
        if stage == "fake_score":
            target = output["noise"] - generated.detach()
            part = (output["noise_pred"] - target).square().mean()
        else:
            axes = tuple(range(1, generated.ndim))
            normalizer = (generated.detach() - output["teacher_x0"]).abs().mean(axes, keepdim=True).clamp_min(1e-6)
            gradient = (output["fake_x0"] - output["teacher_x0"]) / normalizer
            target = (generated - gradient).detach()
            part = 0.5 * (generated - target).square().mean()
        expected_loss = expected_loss + reference.config.modality_weights[name] * (part / accumulation * sp_size)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert not invalid and all(any(key.startswith(name + "/") for key in metrics) for name in modalities)
    assert tu.get_non_tensor_data(data, "sp_size", default=None) == sp_size
    assert tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None) == accumulation
    actual_loss.backward()
    expected_loss.backward()
    for role in models:
        for actual_param, expected_param in zip(
            models[role].parameters(), expected_models[role].parameters(), strict=True
        ):
            if role == stage:
                torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=0, atol=0)
            else:
                assert actual_param.grad is None and expected_param.grad is None
    torch.optim.SGD(models[stage].parameters(), lr=0.01).step()
    torch.optim.SGD(expected_models[stage].parameters(), lr=0.01).step()
    assert any(not torch.equal(value, before[stage][key]) for key, value in models[stage].state_dict().items())
    for role in models:
        for key, value in models[role].state_dict().items():
            torch.testing.assert_close(value, expected_models[role].state_dict()[key], rtol=0, atol=0)
            if role != stage:
                torch.testing.assert_close(value, before[role][key], rtol=0, atol=0)


def test_pdmd_settings_reach_the_registered_loss():
    """The program hands the loss its own config, so a PDMD run cannot silently compute plain DMD."""
    settings = {"projected": True, "surrogate_scale": 1.0, "loss_clamp": 5.0}
    actual, _, initial, _ = make_program(("video", "audio"), **settings)
    reference, _, _, _ = make_program(("video", "audio"), **settings)
    trajectory = actual.rollout(initial, keys=keys_of(actual, initial))
    data, loss_function = loss_inputs()
    torch.manual_seed(107)
    actual_loss, _, _ = actual.compute_loss(
        trajectory, "student", data, loss_function, exit_index=2, keys=keys_of(actual, trajectory.states[0])
    )
    torch.manual_seed(107)
    outputs = reference.stage_outputs(trajectory, "student", 2, keys=keys_of(reference, trajectory.states[0]))
    expected_loss = 0
    for name, output in outputs.items():
        gradient, _, _ = dmd_gradient(
            output["fake_x0"], output["teacher_x0"], output["generated_x0"].detach(), 1e-6, projected=True
        )
        part, _ = dmd_surrogate_loss(output["generated_x0"], gradient, scale=1.0, clamp_max=5.0)
        expected_loss = expected_loss + reference.config.modality_weights[name] * part
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert tu.get_non_tensor_data(data, "dmd_config", default=None) is actual.config


RANDOM_DRAWS = {  # every setting under which the program draws at random
    "sampler": "renoise",
    "critic_noise": "fresh",
    "score_query_noise": "fresh",
}


def keyed_run(stage, keys_update, global_rng_seed, critic_time="uniform", batch=2):
    """Roll out on the keys of update ``keys_update - 1`` for a student, ``keys_update`` for a fake score; return the
    loss, the trained role's gradients and the rollout. No draw may touch the global CPU RNG."""
    program, models, initial, _ = make_program(("video", "audio"), batch=batch, critic_time=critic_time, **RANDOM_DRAWS)
    samples = tuple(range(batch))
    torch.manual_seed(global_rng_seed)
    global_rng = torch.get_rng_state()
    rollout_update = keys_update - 1 if stage == "student" else keys_update
    trajectory = program.rollout(initial, keys=program.keys(rollout_update, samples))
    exit_index = program.draw_exit_index(keys_update) if stage == "student" else None
    data, loss_function = loss_inputs()
    loss, _, _ = program.compute_loss(
        trajectory, stage, data, loss_function, exit_index=exit_index, keys=program.keys(keys_update, samples)
    )
    loss.backward()
    assert torch.equal(torch.get_rng_state(), global_rng), "an update drew from the global CPU RNG"
    return loss.detach(), [parameter.grad.clone() for parameter in models[stage].parameters()], trajectory


@pytest.mark.parametrize("critic_time", ["uniform", "query_matched"])
@pytest.mark.parametrize("stage", ["student", "fake_score"])
def test_an_update_draws_from_its_keys_alone_and_leaves_the_global_rng_alone(stage, critic_time):
    loss, gradients, _ = keyed_run(stage, 7, global_rng_seed=0, critic_time=critic_time)
    again, again_gradients, _ = keyed_run(stage, 7, global_rng_seed=123, critic_time=critic_time)

    torch.testing.assert_close(again, loss, rtol=0, atol=0)
    for gradient, expected in zip(again_gradients, gradients, strict=True):
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["student", "fake_score"])
def test_another_update_draws_other_numbers(stage):
    loss, _, trajectory = keyed_run(stage, 7, global_rng_seed=0)
    other, _, other_trajectory = keyed_run(stage, 13, global_rng_seed=0)
    assert not torch.equal(loss, other)
    assert not torch.equal(trajectory.states[-1]["video"], other_trajectory.states[-1]["video"])


def test_a_sample_draws_its_own_numbers_in_any_microbatch():
    """Sample 1 draws the same renoise sampler noise alone as beside sample 0, so microbatching and ranks do not
    change a sample's draws."""
    program, _, initial, _ = make_program(("video", "audio"), **RANDOM_DRAWS)
    pair = program.rollout(initial, keys=program.keys(4, (0, 1)))
    alone = program.rollout({name: value[1:] for name, value in initial.items()}, keys=program.keys(4, (1,)))
    for name in initial:
        torch.testing.assert_close(alone.states[-1][name][0], pair.states[-1][name][1], rtol=0, atol=0)
    swapped = program.rollout(initial, keys=program.keys(4, (1, 0)))
    assert not torch.equal(swapped.states[-1]["video"], pair.states[-1]["video"])


DRAWS_ONLY = {  # settings under which the critic times and score ratios are the only random draws
    "sampler": "euler",
    "critic_noise": "trajectory",
    "score_query_noise": "student_estimate",
}


@pytest.mark.parametrize(
    "stage, critic_time", [("fake_score", "uniform"), ("fake_score", "query_matched"), ("student", "uniform")]
)
def test_a_sample_draws_its_own_critic_times_and_score_ratios_in_any_microbatch(stage, critic_time):
    """The sigmas the fake score sees for sample 1 are the same alone as beside sample 0, and differ from sample 0's:
    ``draws`` is keyed by the sample."""
    program, _, initial, calls = make_program(("video", "audio"), critic_time=critic_time, **DRAWS_ONLY)
    exit_index = 1 if stage == "student" else None

    def fake_score_sigmas(samples, rows):
        trajectory = program.rollout(
            {name: value[rows] for name, value in initial.items()}, keys=program.keys(3, samples)
        )
        calls.clear()
        program.stage_outputs(trajectory, stage, exit_index, keys=program.keys(4, samples))
        (call,) = [call for call in calls if call["role"] == "fake_score"]
        return call["sigmas"]

    pair = fake_score_sigmas((0, 1), slice(None))
    alone = fake_score_sigmas((1,), slice(1, None))
    for name in pair:
        torch.testing.assert_close(alone[name][0], pair[name][1], rtol=0, atol=0)
    assert not torch.equal(pair["video"][0], pair["video"][1])


def test_the_exit_index_is_pdmds_keyed_draw_of_the_update():
    from tests.utils.dataset._prompt_stream_fixtures import pdmd_stream_seed

    program, _, _, _ = make_program()
    before = torch.get_rng_state()
    indices = [program.draw_exit_index(update) for update in range(200)]
    expected = [
        int(torch.randint(4, (1,), generator=torch.Generator().manual_seed(pdmd_stream_seed(42, "exit", update))))
        for update in range(200)
    ]
    assert indices == expected
    assert set(indices) == {0, 1, 2, 3}
    assert torch.equal(before, torch.get_rng_state())


def test_keys_must_name_one_sample_per_row():
    program, _, initial, calls = make_program()
    with pytest.raises(ValueError, match="samples"):
        program.rollout(initial, keys=program.keys(0, (0,)))
    assert not calls


@pytest.mark.parametrize("stage", ["student", "fake_score"])
def test_nonfinite_output_is_reported_to_the_update_owner(stage):
    program, models, initial, _ = make_program()
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    with torch.no_grad():
        models["fake_score"]["image"].weight.fill_(float("nan"))
    data, loss_function = loss_inputs()
    _, _, invalid = program.compute_loss(
        trajectory,
        stage,
        data,
        loss_function,
        exit_index=0 if stage == "student" else None,
        keys=keys_of(program, trajectory.states[0]),
    )
    assert invalid


@pytest.mark.parametrize("name", ["gradient_accumulation_steps", "sp_size"])
@pytest.mark.parametrize("value", [None, 0, True, 1.5])
def test_loss_scaling_must_be_explicit_before_any_forward(name, value):
    program, _, initial, calls = make_program()
    trajectory = program.rollout(initial, keys=keys_of(program, initial))
    calls.clear()
    data, loss_function = loss_inputs()
    tu.assign_non_tensor(data, **{name: value})
    with pytest.raises(ValueError, match=name):
        program.compute_loss(
            trajectory, "student", data, loss_function, exit_index=0, keys=keys_of(program, trajectory.states[0])
        )
    assert not calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score_discrete_steps": 1000},
        {"rollout_timestep_shift": 4},
        {"score_timestep_shift": 4},
        {"score_sigma_min": 0.1},
        {"score_sigma_max": 0.9},
    ],
)
def test_settings_the_program_does_not_read_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_program(**kwargs)


def test_invalid_stage_and_layout_fail_before_any_forward():
    program, _, initial, calls = make_program()
    for invalid in ({"other": torch.ones(2, 3)}, {"image": torch.ones(2)}, {"image": torch.empty(0, 3)}):
        with pytest.raises(ValueError):
            program.rollout(invalid, keys=keys_of(program, invalid))
    with pytest.raises(ValueError, match="Unknown DMD stage"):
        program.stage_outputs(
            program.rollout(initial, keys=keys_of(program, initial)), "teacher", keys=keys_of(program, initial)
        )


@pytest.mark.parametrize("prediction", [{}, {"image": torch.zeros(2, 1, 3)}])
def test_prediction_layout_cannot_silently_broadcast(prediction):
    program, _, initial, _ = make_program()
    program.predict = lambda *args: prediction
    with pytest.raises(ValueError, match="Velocity predictions"):
        program.rollout(initial, keys=keys_of(program, initial))
