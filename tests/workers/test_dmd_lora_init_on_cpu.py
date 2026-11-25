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
"""``model.lora_init_weights=true`` reaches peft as the bool ``True`` for both DMD adapters.

peft reads ``True`` as Kaiming-uniform A and zero B, the PDMD reference's init
(https://github.com/ZeamoxWang/pdmd at 6b6e106, ``src/pdmd/adapters.py:18-19,48-49``).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from verl.utils.config import omega_conf_to_dataclass

import verl_omni
from tests.workers.engine.dmd_engine_fixtures import ARCHITECTURE, TinyDiT
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin

CONFIG_DIR = str(Path(verl_omni.__file__).parent / "trainer" / "config")
ADAPTERS = ("default", "fake_score")


def _lora_module(tmp_path: Path, init_weights: str) -> torch.nn.Module:
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": ARCHITECTURE}))
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                f"actor_rollout_ref.model.path={tmp_path}",
                "+actor_rollout_ref.model.load_tokenizer=false",
                "actor_rollout_ref.model.lora_rank=4",
                "actor_rollout_ref.model.target_modules=[proj_in,proj_out]",
                "actor_rollout_ref.model.policy_state_adapters=[default,fake_score]",
                f"actor_rollout_ref.model.lora_init_weights={init_weights}",
            ],
        )
    model_config = omega_conf_to_dataclass(config.actor_rollout_ref.model)
    return LoRAAdapterMixin._build_lora_module(SimpleNamespace(model_config=model_config), TinyDiT())


def test_the_true_override_reaches_both_adapters_as_kaiming_a_and_zero_b(tmp_path):
    """Catches the override turned into the string "True" on its way to peft, or reaching one adapter only."""
    module = _lora_module(tmp_path, "true")
    for adapter in ADAPTERS:
        assert module.peft_config[adapter].init_lora_weights is True
        lora_a = [p for name, p in module.named_parameters() if f".lora_A.{adapter}." in name]
        lora_b = [p for name, p in module.named_parameters() if f".lora_B.{adapter}." in name]
        assert lora_a and lora_b
        for weight in lora_a:
            bound = 1 / weight.shape[1] ** 0.5  # kaiming_uniform(a=sqrt(5)): U(-1/sqrt(fan_in), 1/sqrt(fan_in))
            assert float(weight.abs().max()) <= bound
        assert all(bool((weight == 0).all()) for weight in lora_b)


def test_the_gaussian_setting_still_reaches_peft_unchanged(tmp_path):
    """The control: gaussian stays peft's gaussian init, so the check above reads the override."""
    module = _lora_module(tmp_path, "gaussian")
    assert all(module.peft_config[adapter].init_lora_weights == "gaussian" for adapter in ADAPTERS)
