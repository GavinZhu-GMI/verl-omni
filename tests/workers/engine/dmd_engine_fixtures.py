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
"""Build a DMD engine around an existing PEFT module, without a model config or a training worker."""

from types import SimpleNamespace

import torch
from verl.workers.config import FSDPOptimizerConfig

from verl_omni.workers.engine.fsdp.dmd_impl import DMDDiffusersFSDPEngine

ADAPTERS = ("default", "fake_score")
STUDENT_LR = 5e-5
CRITIC_LR = 1e-5


def optimizer_config(lr: float) -> FSDPOptimizerConfig:
    """AdamW with beta1 = 0, beta2 = 0.9, no weight decay and gradient clipping at 1, at ``lr``."""
    return FSDPOptimizerConfig(lr=lr, weight_decay=0.0, betas=(0.0, 0.9), clip_grad=1.0)


def make_dmd_engine(module: torch.nn.Module) -> DMDDiffusersFSDPEngine:
    """Return an engine over ``module`` (adapters ``default`` and ``fake_score``) with both optimizers built.

    ``initialize`` would build the module from a model config; here the caller owns the module, plain or FSDP2.
    """
    engine = DMDDiffusersFSDPEngine.__new__(DMDDiffusersFSDPEngine)
    engine.module = module
    engine.rank = 0
    engine.optimizer_config = optimizer_config(STUDENT_LR)
    engine.engine_config = SimpleNamespace(forward_only=False)
    engine.optimizer = engine._build_optimizer(module)
    engine.lr_scheduler = engine._build_lr_scheduler(engine.optimizer)
    engine._init_adapter_stages()
    engine.add_adapter_optimizer("fake_score", optimizer_config(CRITIC_LR))
    return engine
