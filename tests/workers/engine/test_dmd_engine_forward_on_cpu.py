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
"""DMD engine orchestration on one CPU rank: a tiny DiT under a fake architecture adapter that records each call."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from dmd_engine_fixtures import (
    ADAPTERS,
    STUDENT_LR,
    RecordingDMDModel,
    TinyDiT,
    dmd_config,
    make_dmd_engine,
    model_config,
    optimizer_config,
)
from peft import LoraConfig
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.workers.config.diffusion.actor import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.config.diffusion.dmd import DiffusionDMDConfig
from verl_omni.workers.engine.fsdp import dmd_impl
from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine
from verl_omni.workers.engine.fsdp.dmd_impl import DMDDiffusersFSDPEngine
from verl_omni.workers.engine.lora_adapter_mixin import adapter_parameters
from verl_omni.workers.utils.losses import diffusion_loss

LOSS = partial(
    diffusion_loss,
    config=DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=DiffusionLossConfig(loss_mode="dmd2")),
)


@pytest.fixture
def process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(dmd_impl, "get_device_id", lambda: torch.device("cpu"))
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def build_engine(config=None) -> DMDDiffusersFSDPEngine:
    torch.manual_seed(0)
    model = TinyDiT()
    lora = LoraConfig(r=2, lora_alpha=2, target_modules=["proj_in", "proj_out"])
    for role in ADAPTERS:
        model.add_adapter(lora, adapter_name=role)
    model.set_adapter("default")
    RecordingDMDModel.calls = []
    return make_dmd_engine(model, config)


@pytest.fixture
def engine(process_group):
    return build_engine()


def batch(stage: str, prompts: tuple[int, ...], **batch_settings) -> TensorDict:
    """A ``stage`` batch of one prompt id per row, with the batch settings the training worker sets.

    A fake_score batch is update 0 and a student batch update 1, unless ``dmd_update`` says otherwise.
    """
    data = TensorDict(
        {
            "prompt_ids": torch.tensor(prompts, dtype=torch.long).view(-1, 1),
            "global_sample": torch.arange(len(prompts)),
        },
        batch_size=[len(prompts)],
    )
    settings = {"use_dynamic_bsz": False, "dmd_update": int(stage == "student"), **batch_settings}
    tu.assign_non_tensor(data, dmd_stage=stage, **settings)
    return data


def train(engine, stage: str, prompts: tuple[int, ...] = (0, 1), loss=LOSS, **batch_settings) -> list[dict]:
    """Run one update of ``stage`` on ``prompts``, one per microbatch, and return the predictions it made."""
    first = len(RecordingDMDModel.calls)
    engine.train_batch(batch(stage, prompts, **batch_settings), loss)
    return RecordingDMDModel.calls[first:]


def test_student_update_trains_on_the_fake_score_rollout_of_each_microbatch(engine):
    """Each student microbatch restarts at a state of the rollout its prompts got in the fake-score update."""
    critic = train(engine, "fake_score")
    student = train(engine, "student")

    # per microbatch: the student's one graph-bearing forward, then no-grad fake score and teacher; no rollout
    assert len(student) == 3 * 2
    restarts = [call for call in student if call["grad"]]
    assert sorted(call["prompts"] for call in restarts) == [[0], [1]]
    for restart in restarts:
        rollout_states = [call["latents"] for call in critic if call["prompts"] == restart["prompts"]]
        assert any(torch.equal(restart["latents"], state) for state in rollout_states)


def test_student_update_conditions_on_the_prompts_of_the_kept_rollout(engine):
    """A student batch with other prompts still trains on the kept rollouts under the prompts that made them."""
    train(engine, "fake_score", prompts=(0, 1))
    student = train(engine, "student", prompts=(5, 6))

    assert {tuple(call["prompts"]) for call in student} == {(0,), (1,)}


def test_student_loss_reads_the_microbatch_of_the_kept_rollout(engine):
    """Per-sample loss inputs come with the kept rollout, not from the student batch."""
    student_loss_prompts = []

    def recording_loss(model_output, data, dp_group):
        if tu.get_non_tensor_data(data, "dmd_stage", default=None) == "student":
            student_loss_prompts.append(data["prompt_ids"].reshape(-1).tolist())
        return LOSS(model_output=model_output, data=data, dp_group=dp_group)

    train(engine, "fake_score", prompts=(0, 1))
    train(engine, "student", prompts=(5, 6), loss=recording_loss)

    assert sorted(student_loss_prompts) == [[0], [1]]


@pytest.mark.parametrize(
    "earlier, student_prompts",
    [
        ((("fake_score", (0, 1)), ("student", (0, 1))), (0, 1)),  # each rollout serves one student update
        ((("fake_score", (0, 1)), ("fake_score", (0,))), (0, 1)),  # the last fake_score update kept one slot
        ((("fake_score", (0, 1)),), (0,)),  # the last fake_score update kept two slots
    ],
)
def test_student_without_its_kept_rollouts_fails_before_any_forward(engine, earlier, student_prompts):
    for stage, prompts in earlier:
        train(engine, stage, prompts)
    forwards = len(RecordingDMDModel.calls)

    with pytest.raises(RuntimeError, match="fake_score update kept"):
        train(engine, "student", student_prompts)
    assert len(RecordingDMDModel.calls) == forwards


def test_a_fresh_engines_student_update_rebuilds_the_rollouts_of_the_update_before(process_group):
    """After a resume no rollout is kept: the student rolls its batch out again on the keys of the update before, and
    trains as it would on the rollouts that update kept (pdmd ``training/train.py`` rebuilds them the same way)."""
    straight = build_engine()
    train(straight, "fake_score", prompts=(0, 1), dmd_update=4)
    kept = train(straight, "student", prompts=(0, 1), dmd_update=5)
    resumed = build_engine()
    torch.manual_seed(77)  # the rebuild must not read the global RNG

    rebuilt = train(resumed, "student", prompts=(0, 1), dmd_update=5)

    rollout, student = rebuilt[: 2 * 4], rebuilt[2 * 4 :]  # first the rollouts: two microbatches of four steps
    assert all(call["adapter"] == "default" and not call["grad"] for call in rollout)
    assert len(student) == len(kept)
    for call, expected in zip(student, kept, strict=True):
        assert call["prompts"] == expected["prompts"] and call["adapter"] == expected["adapter"]
        torch.testing.assert_close(call["latents"], expected["latents"], rtol=0, atol=0)


def test_rebuilt_rollouts_serve_one_student_update(process_group):
    """A second student update on rollouts a fresh engine rebuilt fails as one on kept rollouts does."""
    resumed = build_engine()
    train(resumed, "student", prompts=(0, 1), dmd_update=5)
    forwards = len(RecordingDMDModel.calls)

    with pytest.raises(RuntimeError, match="fake_score update kept"):
        train(resumed, "student", prompts=(0, 1), dmd_update=5)
    assert len(RecordingDMDModel.calls) == forwards


@pytest.mark.parametrize(
    "fake_score_update, student_update, error",
    [(3, 5, "kept those of update 3"), (0, 0, "dmd_update")],
)
def test_a_student_update_refuses_rollouts_of_another_update(engine, fake_score_update, student_update, error):
    train(engine, "fake_score", prompts=(0, 1), dmd_update=fake_score_update)
    forwards = len(RecordingDMDModel.calls)
    with pytest.raises((RuntimeError, ValueError), match=error):
        train(engine, "student", prompts=(0, 1), dmd_update=student_update)
    assert len(RecordingDMDModel.calls) == forwards


@pytest.mark.parametrize("missing", ["dmd_update", "global_sample"])
def test_a_batch_without_its_draw_keys_is_rejected_before_any_forward(engine, missing):
    data = batch("fake_score", (0, 1))
    del data[missing]
    with pytest.raises(ValueError, match=missing):
        engine.train_batch(data, LOSS)
    assert RecordingDMDModel.calls == []


def test_a_student_update_reports_its_keyed_exit_index(engine):
    train(engine, "fake_score", prompts=(0, 1), dmd_update=10)
    output = engine.train_batch(batch("student", (0, 1), dmd_update=11), LOSS)
    assert output["metrics"]["dmd/exit_index"] == [float(engine._program.draw_exit_index(11))] * 2


def test_a_failed_fake_score_update_keeps_no_rollout(engine, monkeypatch):
    """A rollout that raises mid-update leaves nothing for the student: neither its own first slots nor the rollouts
    of the update before it."""
    train(engine, "fake_score", prompts=(0, 1))
    initial_noise = RecordingDMDModel.dmd_initial_noise

    def fail_on_prompt_one(cls, micro_batch, generators):
        if micro_batch["prompt_ids"].item() == 1:
            raise InterruptedError
        return initial_noise(micro_batch, generators)

    monkeypatch.setattr(RecordingDMDModel, "dmd_initial_noise", classmethod(fail_on_prompt_one))
    with pytest.raises(InterruptedError):
        train(engine, "fake_score", prompts=(0, 1))

    for slots in (1, 2):
        with pytest.raises(RuntimeError, match="fake_score update kept"):
            train(engine, "student", prompts=tuple(range(slots)))


def test_a_fake_score_update_whose_loss_fails_keeps_no_rollout(engine):
    """Rollouts of a fake_score update that never finished its backward do not reach the student."""
    train(engine, "fake_score", prompts=(0, 1))

    def failing_loss(model_output, data, dp_group):
        raise InterruptedError

    with pytest.raises(InterruptedError):
        train(engine, "fake_score", prompts=(0, 1), loss=failing_loss)
    with pytest.raises(RuntimeError, match="fake_score update kept"):
        train(engine, "student", prompts=(0, 1))


def test_microbatches_of_two_share_one_exit_index_and_scale_by_their_count(process_group):
    """With microbatches of two, each student update trains three slots from the same exit step of their rollouts."""
    engine = build_engine(dmd_config(fake_score_micro_batch_size_per_gpu=2, student_micro_batch_size_per_gpu=2))
    accumulation_steps = []

    def recording_loss(model_output, data, dp_group):
        accumulation_steps.append(tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None))
        return LOSS(model_output=model_output, data=data, dp_group=dp_group)

    for _ in range(3):
        critic = train(engine, "fake_score", prompts=tuple(range(6)), loss=recording_loss)
        student = train(engine, "student", prompts=tuple(range(6)), loss=recording_loss)
        restarts = [call for call in student if call["grad"]]
        assert sorted(call["prompts"] for call in restarts) == [[0, 1], [2, 3], [4, 5]]
        exit_steps = set()
        for restart in restarts:
            rollout = [call["latents"] for call in critic if call["prompts"] == restart["prompts"] and not call["grad"]]
            exit_steps.add(next(step for step, state in enumerate(rollout) if torch.equal(restart["latents"], state)))
        assert len(exit_steps) == 1
    assert set(accumulation_steps) == {3}


def test_a_batch_with_a_jagged_field_splits_into_microbatches(engine, monkeypatch):
    """Variable-length prompt embeddings, as H3 rows carry them, split per microbatch like dense fields."""
    lengths = []

    def record_length(cls, micro_batch):
        lengths.append([embeds.shape[0] for embeds in micro_batch["prompt_embeds"].unbind()])
        return micro_batch["prompt_ids"]

    monkeypatch.setattr(RecordingDMDModel, "dmd_condition", classmethod(record_length))
    data = batch("fake_score", (0, 1))
    data["prompt_embeds"] = torch.nested.nested_tensor([torch.ones(3, 4), torch.ones(5, 4)], layout=torch.jagged)
    engine.train_batch(data, LOSS)

    assert lengths == [[3], [5]]


def test_each_role_predicts_under_its_adapter(engine):
    """Rollout and student run on ``default``, the fake score on ``fake_score``, the teacher with adapters off."""
    critic = train(engine, "fake_score")
    student = train(engine, "student")

    def roles(calls):
        return {(call["grad"], call["adapter"]) for call in calls}

    assert roles(critic) == {(False, "default"), (True, "fake_score")}
    assert roles(student) == {(True, "default"), (False, "fake_score"), (False, None)}


def forward_only_update(engine, data):
    with engine.train_stage("fake_score"):
        engine.forward_backward_batch(data, LOSS, forward_only=True)


@pytest.mark.parametrize(
    "stage, prompts, run, error",
    [
        ("fake_score", (0, 1), forward_only_update, "no forward-only path"),
        ("fake_score", (0, 1, 2), lambda engine, data: engine.train_batch(data, LOSS), "does not split into"),
        ("teacher", (0, 1), lambda engine, data: engine.train_batch(data, LOSS), "Unknown DMD stage"),
        ("fake_score", (), lambda engine, data: engine.train_batch(data, LOSS), "empty"),
    ],
    ids=["forward_only", "partial_microbatch", "unknown_stage", "empty_fake_score"],
)
def test_an_invalid_update_is_rejected_before_any_forward(process_group, stage, prompts, run, error):
    engine = build_engine(dmd_config(fake_score_micro_batch_size_per_gpu=2, student_micro_batch_size_per_gpu=2))
    with pytest.raises(ValueError, match=error):
        run(engine, batch(stage, prompts))
    assert RecordingDMDModel.calls == []


def test_an_empty_student_batch_is_rejected_before_it_draws_an_exit_index(engine, monkeypatch):
    """With no kept rollout, an empty student update would otherwise draw an exit index and step the student."""
    monkeypatch.setattr(engine._program, "draw_exit_index", lambda: pytest.fail("an empty update drew an exit index"))
    with pytest.raises(ValueError, match="empty"):
        train(engine, "student", prompts=())
    assert engine._last_optimized_adapter is None


@pytest.mark.parametrize(
    "settings, error",
    [
        ({"use_dynamic_bsz": True}, "use_dynamic_bsz"),
        ({"use_no_sync_for_gradient_accumulation": True}, "use_no_sync_for_gradient_accumulation"),
    ],
    ids=["dynamic_bsz", "no_sync"],
)
def test_an_unsupported_batch_setting_is_rejected_before_any_forward(engine, settings, error):
    with pytest.raises(ValueError, match=error):
        train(engine, "fake_score", **settings)
    assert RecordingDMDModel.calls == []


def test_a_batch_without_use_dynamic_bsz_is_rejected(engine):
    """verl splits a batch that does not set ``use_dynamic_bsz`` by token count, which DMD does not support."""
    data = batch("fake_score", (0, 1))
    del data["use_dynamic_bsz"]
    with pytest.raises(ValueError, match="use_dynamic_bsz"):
        engine.train_batch(data, LOSS)


def test_an_update_outside_train_batch_is_rejected_before_any_forward(engine):
    """Outside a stage the default adapter is active, so a fake_score update would train under the wrong adapter."""
    with pytest.raises(RuntimeError, match="inside train_batch"):
        engine.forward_backward_batch(batch("fake_score", (0, 1)), LOSS)
    assert RecordingDMDModel.calls == []


def test_a_rejected_fake_score_update_keeps_no_rollout(engine):
    """A fake_score update rejected before its rollouts still replaces the rollouts of the update before it."""
    train(engine, "fake_score", prompts=(0, 1))
    with pytest.raises(ValueError, match="use_dynamic_bsz"):
        train(engine, "fake_score", prompts=(0, 1), use_dynamic_bsz=True)
    with pytest.raises(RuntimeError, match="fake_score update kept"):
        train(engine, "student", prompts=(0, 1))


@pytest.mark.parametrize("stage, trained", [("fake_score", "fake_score"), ("student", "default")])
def test_each_stage_update_changes_only_its_adapter(process_group, stage, trained):
    # a large critic step: after one step at the fixture's LR, the student's loss and gradient are exactly 0,
    # presumably because fake score and teacher then differ below float32 resolution
    engine = build_engine(dmd_config(fake_score_optim=optimizer_config(1e-1)))
    if stage == "student":
        train(engine, "fake_score")
    parameters = {adapter: adapter_parameters(engine.module, adapter) for adapter in ADAPTERS}
    before = {adapter: [param.detach().clone() for param in params] for adapter, params in parameters.items()}

    train(engine, stage)

    for adapter, params in parameters.items():
        changed = any(not torch.equal(param, old) for param, old in zip(params, before[adapter], strict=True))
        assert changed is (adapter == trained), adapter


def test_an_invalid_dmd_config_fails_before_the_model_loads():
    """The default config samples discrete score times, which the DMD program rejects at construction."""
    base_init = MagicMock()
    with patch.object(DiffusersFSDPEngine, "__init__", base_init):
        with pytest.raises(ValueError, match="continuous score sampling"):
            DMDDiffusersFSDPEngine(
                model_config(),
                SimpleNamespace(strategy="fsdp2", forward_only=False, seed=3),
                optimizer_config(STUDENT_LR),
                None,
                dmd_config=DiffusionDMDConfig(),
            )
    base_init.assert_not_called()


def test_microbatches_reach_the_architecture_adapter_on_the_engine_device(engine, monkeypatch):
    """The engine moves each microbatch to the accelerator verl names before the adapter reads it."""
    devices = []

    def record_device(cls, micro_batch):
        devices.append(micro_batch["prompt_ids"].device)
        raise InterruptedError

    monkeypatch.setattr(dmd_impl, "get_device_id", lambda: "meta")
    monkeypatch.setattr(RecordingDMDModel, "dmd_condition", classmethod(record_device))
    with pytest.raises(InterruptedError):
        train(engine, "fake_score")
    assert devices == [torch.device("meta")]


def test_a_student_learning_rate_schedule_is_rejected():
    """The engine applies the config's constant-LR check to the actor optimizer too."""
    with pytest.raises(ValueError, match="constant learning rate"):
        DMDDiffusersFSDPEngine(
            model_config(),
            SimpleNamespace(strategy="fsdp2", forward_only=False),
            optimizer_config(STUDENT_LR, lr_warmup_steps=2),
            None,
            dmd_config=dmd_config(),
        )
