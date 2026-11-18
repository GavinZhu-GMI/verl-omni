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
"""Adapter stages of the DMD engine: two trained LoRA adapters on one base, each with its own optimizer."""

import pytest
import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.loaders.peft import PeftAdapterMixin
from dmd_engine_fixtures import CRITIC_LR, STUDENT_LR, make_dmd_engine, optimizer_config
from peft import LoraConfig

from verl_omni.utils.config import resolve_lora_config
from verl_omni.workers.engine.fsdp import diffusers_impl
from verl_omni.workers.engine.fsdp.dmd_impl import DMDDiffusersFSDPEngine
from verl_omni.workers.engine.lora_adapter_mixin import adapter_parameters


class TinyDiT(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """Two linear layers with the diffusers PEFT adapter API."""

    @register_to_config
    def __init__(self, dim: int = 8):
        """Build the input and output projections."""
        super().__init__()
        self.proj_in = torch.nn.Linear(dim, dim)
        self.proj_out = torch.nn.Linear(dim, dim)

    def forward(self, hidden_states):
        """Return the projected hidden states."""
        return self.proj_out(torch.tanh(self.proj_in(hidden_states)))


def make_engine() -> DMDDiffusersFSDPEngine:
    """An engine on a plain CPU module with nonzero ``lora_B``, so both LoRA factors get gradients."""
    torch.manual_seed(0)
    module = TinyDiT()
    lora = LoraConfig(r=2, lora_alpha=2, target_modules=["proj_in", "proj_out"])
    module.add_adapter(lora, adapter_name="default")
    module.add_adapter(lora, adapter_name="fake_score")
    with torch.no_grad():  # peft starts lora_B at 0, which would leave lora_A without a gradient
        for name, param in module.named_parameters():
            if ".lora_B." in name:
                param.normal_()
    module.set_adapter("default")
    return make_dmd_engine(module)


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in module.named_parameters()}


def _active(engine: DMDDiffusersFSDPEngine) -> list[str]:
    return engine.module.active_adapters()


def test_adapter_parameters_partition_the_lora_parameters():
    """Each adapter's parameters are its own LoRA tensors, and the two sets cover every LoRA tensor."""
    module = make_engine().module
    student = {id(param) for param in adapter_parameters(module, "default")}
    critic = {id(param) for param in adapter_parameters(module, "fake_score")}
    lora = {id(param) for name, param in module.named_parameters() if ".lora_" in name}
    assert student and critic
    assert not student & critic
    assert student | critic == lora
    with pytest.raises(ValueError, match="no parameters"):
        adapter_parameters(module, "old")


def test_each_optimizer_holds_only_its_adapter_with_its_own_lr():
    """The student optimizer gets the default adapter at the actor LR; the critic gets fake_score at its LR."""
    engine = make_engine()
    for adapter, lr in (("default", STUDENT_LR), ("fake_score", CRITIC_LR)):
        with engine.train_stage(adapter):
            optimizer = engine._stage_optimizer().optimizer
        held = [param for group in optimizer.param_groups for param in group["params"]]
        assert {id(param) for param in held} == {id(param) for param in adapter_parameters(engine.module, adapter)}
        assert [group["lr"] for group in optimizer.param_groups] == [lr] * len(optimizer.param_groups)


@pytest.mark.parametrize("trained, other", [("default", "fake_score"), ("fake_score", "default")])
def test_stage_updates_only_its_adapter(trained, other):
    """A stage's backward and step change only the trained adapter, with other adapters' forwards in between."""
    engine = make_engine()
    before = _snapshot(engine.module)
    inputs = torch.randn(4, 8)
    with engine.train_stage(trained):
        engine.optimizer_zero_grad()
        output = engine.module(inputs)
        with torch.no_grad():
            with engine.use_adapter(other):
                engine.module(inputs)
            with engine.use_adapter("reference"):
                engine.module(inputs)
        assert _active(engine) == [trained]
        output.square().mean().backward()
        grad_norm = engine.optimizer_step()
        engine.lr_scheduler_step()
    assert grad_norm > 0
    trained_ids = {id(param) for param in adapter_parameters(engine.module, trained)}
    for name, param in engine.module.named_parameters():
        if id(param) in trained_ids:
            assert not torch.equal(param, before[name]), name
        else:
            assert param.grad is None, name
            torch.testing.assert_close(param, before[name], rtol=0, atol=0)


def test_use_adapter_inside_a_stage_returns_to_the_stage_adapter():
    """Inside a stage use_adapter ends on the stage's adapter, also after an exception and through 'reference'."""
    engine = make_engine()
    with engine.train_stage("fake_score"):
        with engine.use_adapter("default"):
            assert _active(engine) == ["default"]
        assert _active(engine) == ["fake_score"]
        with pytest.raises(RuntimeError, match="boom"):
            with engine.use_adapter("default"):
                raise RuntimeError("boom")
        assert _active(engine) == ["fake_score"]
        with engine.use_adapter("reference"):
            assert all(param.requires_grad is False for param in adapter_parameters(engine.module, "fake_score"))
        assert _active(engine) == ["fake_score"]
        assert all(param.requires_grad for param in adapter_parameters(engine.module, "fake_score"))
    assert _active(engine) == ["default"]


def test_use_adapter_outside_a_stage_returns_to_default():
    """Outside a stage use_adapter keeps the single-adapter engine's behaviour and ends on 'default'."""
    engine = make_engine()
    engine._set_adapter("fake_score")
    with engine.use_adapter("fake_score"):
        assert _active(engine) == ["fake_score"]
    assert _active(engine) == ["default"]


def test_optimizer_calls_outside_a_stage_act_on_the_default_adapter():
    """Outside any stage the engine trains 'default', as the single-adapter engine does."""
    engine = make_engine()
    before = _snapshot(engine.module)
    engine.optimizer_zero_grad()
    engine.module(torch.randn(4, 8)).square().mean().backward()
    engine.optimizer_step()
    assert engine.lr_scheduler_step() == pytest.approx(STUDENT_LR)
    student_ids = {id(param) for param in adapter_parameters(engine.module, "default")}
    for name, param in engine.module.named_parameters():
        assert torch.equal(param, before[name]) is (id(param) not in student_ids), name


def test_critic_stage_steps_the_critic_scheduler():
    """lr_scheduler_step inside a stage advances that adapter's scheduler and returns its LR."""
    engine = make_engine()
    with engine.train_stage("fake_score"):
        assert engine.lr_scheduler_step() == pytest.approx(CRITIC_LR)


def test_stage_and_optimizer_misuse_raises():
    """Nested stages, stages without an optimizer and duplicate or unknown optimizers are rejected."""
    engine = make_engine()
    with engine.train_stage("fake_score"):
        with pytest.raises(RuntimeError, match="already in the 'fake_score' stage"):
            with engine.train_stage("default"):
                pass
    with pytest.raises(ValueError, match="no optimizer"):
        with engine.train_stage("old"):
            pass
    with pytest.raises(ValueError, match="already has an optimizer"):
        engine.add_adapter_optimizer("fake_score", optimizer_config(CRITIC_LR))
    with pytest.raises(ValueError, match="already has an optimizer"):
        engine.add_adapter_optimizer("default", optimizer_config(CRITIC_LR))
    with pytest.raises(ValueError, match="no parameters"):
        engine.add_adapter_optimizer("old", optimizer_config(CRITIC_LR))
    assert _active(engine) == ["default"]


def test_to_moves_every_adapter_optimizer(monkeypatch):
    """Optimizer offload and reload cover the critic optimizer as well as the student's."""
    engine = make_engine()
    moved = []
    monkeypatch.setattr(diffusers_impl, "offload_fsdp_optimizer", lambda optimizer: moved.append(("cpu", optimizer)))
    monkeypatch.setattr(
        diffusers_impl, "load_fsdp_optimizer", lambda optimizer, device: moved.append((device, optimizer))
    )
    monkeypatch.setattr(diffusers_impl, "get_device_name", lambda: "cuda")
    engine.to("cpu", model=False, optimizer=True, grad=False)
    engine.to("cuda", model=False, optimizer=True, grad=False)
    with engine.train_stage("fake_score"):
        critic = engine._stage_optimizer().optimizer
    assert moved == [
        ("cpu", engine.optimizer),
        ("cpu", critic),
        ("cuda", engine.optimizer),
        ("cuda", critic),
    ]


def test_fake_score_is_a_known_policy_state():
    """The critic adapter can be requested through policy_state_adapters."""
    settings = resolve_lora_config({"lora_rank": 2, "policy_state_adapters": ["default", "fake_score"]})
    assert settings.adapters == ("default", "fake_score")
