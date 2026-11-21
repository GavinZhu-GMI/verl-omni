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
"""The DMD adapter contract, run against the engine tests' recording adapter and the tiny MiniMax H3 adapter."""

from contextlib import nullcontext

import pytest
import torch
from dmd_adapter_conformance import CHECKS, AdapterCase
from dmd_engine_fixtures import ADAPTERS, RecordingDMDModel, TinyDiT
from peft import LoraConfig
from tensordict import TensorDict

from tests.pipelines.minimax_h3_dmd2.h3_dmd2_fixtures import MiniMaxH3DMD2, micro_batch, tiny_h3, tiny_row_widths


def recording_case() -> AdapterCase:
    torch.manual_seed(0)
    module = TinyDiT()
    lora = LoraConfig(r=2, lora_alpha=2, target_modules=["proj_in", "proj_out"])
    for role in ADAPTERS:
        module.add_adapter(lora, adapter_name=role)
    module.set_adapter("default")
    prompts = TensorDict({"prompt_ids": torch.tensor([[0], [1]])}, batch_size=[2])
    return AdapterCase(RecordingDMDModel, module, prompts, nullcontext)


def h3_case() -> AdapterCase:
    return AdapterCase(MiniMaxH3DMD2, tiny_h3(), micro_batch(), tiny_row_widths)


@pytest.mark.parametrize("check", CHECKS, ids=lambda check: check.__name__)
@pytest.mark.parametrize("make_case", [recording_case, h3_case], ids=["recording", "minimax_h3"])
def test_adapter_keeps_the_dmd_contract(make_case, check):
    torch.manual_seed(1)
    check(make_case())
