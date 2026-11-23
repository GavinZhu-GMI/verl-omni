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
"""The DMD engine's update outputs in the shape the training worker's mini-batch loop aggregates."""

from functools import partial

import pytest
import torch
import torch.distributed as dist
from dmd_engine_fixtures import ADAPTERS, RecordingDMDModel, TinyDiT, make_dmd_engine
from peft import LoraConfig
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.workers.config.diffusion.actor import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.engine.fsdp import dmd_impl
from verl_omni.workers.utils.losses import diffusion_loss

LOSS = partial(
    diffusion_loss,
    config=DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=DiffusionLossConfig(loss_mode="dmd2")),
)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(dmd_impl, "get_device_id", lambda: torch.device("cpu"))
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    try:
        torch.manual_seed(0)
        model = TinyDiT()
        for role in ADAPTERS:
            model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["proj_in", "proj_out"]), adapter_name=role)
        model.set_adapter("default")
        RecordingDMDModel.calls = []
        yield make_dmd_engine(model)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("stage", ["fake_score", "student"])
def test_each_metric_holds_one_float_per_microbatch(engine, stage):
    """verl's worker gathers each metric over ranks and flattens the per-rank lists; a bare float breaks that."""
    for update, update_stage in enumerate(("fake_score", "student")[: 1 + (stage == "student")]):
        data = TensorDict({"prompt_ids": torch.arange(3).view(-1, 1), "global_sample": torch.arange(3)}, batch_size=[3])
        tu.assign_non_tensor(data, dmd_stage=update_stage, dmd_update=update, use_dynamic_bsz=False)
        metrics = engine.train_batch(data, LOSS)["metrics"]

    assert metrics["dmd/nonfinite_fraction"] == [0.0, 0.0, 0.0]
    for key, values in metrics.items():
        if key == "grad_norm":  # the worker pops it before the gather: it is already the same on every rank
            continue
        assert isinstance(values, list) and len(values) == 3, key
        assert all(isinstance(value, float) for value in values), key
