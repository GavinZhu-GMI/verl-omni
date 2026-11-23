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
"""Build a DMD engine around an existing PEFT module, without loading a model or starting a training worker."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.loaders.peft import PeftAdapterMixin
from peft.tuners.tuners_utils import BaseTunerLayer
from tensordict import TensorDict
from verl.workers.config import FSDPOptimizerConfig

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config.diffusion.dmd import DiffusionDMDConfig
from verl_omni.workers.engine.fsdp import dmd_impl
from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine
from verl_omni.workers.engine.fsdp.dmd_impl import DMDDiffusersFSDPEngine

ADAPTERS = ("default", "fake_score")
ARCHITECTURE = "DMDEngineTestModel"  # the model configs of these engines name it; RecordingDMDModel serves it
STUDENT_LR = 5e-5
CRITIC_LR = 1e-5


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


class _RecordingDMDModel:
    """Stand-in for a ``(architecture, dmd2)`` adapter: one ``image`` modality of shape (B, 2, 3).

    Each prediction scales the latents by a scalar of the module output, so gradients reach the active adapter.
    It records the active adapter, or ``None`` with adapters disabled.
    """

    calls: list[dict] = []

    @classmethod
    def dmd_condition(cls, micro_batch: TensorDict) -> torch.Tensor:
        return micro_batch["prompt_ids"]

    @classmethod
    def dmd_initial_noise(cls, micro_batch: TensorDict, generators: list[torch.Generator]) -> dict[str, torch.Tensor]:
        return {"image": torch.stack([torch.randn(2, 3, generator=generator) for generator in generators])}

    @classmethod
    def dmd_predict(cls, module, latents, sigmas, condition, negative) -> dict[str, torch.Tensor]:
        lora_layers = [layer for layer in module.modules() if isinstance(layer, BaseTunerLayer)]
        (active,) = {layer.active_adapters[0] for layer in lora_layers}
        disabled = all(layer.disable_adapters for layer in lora_layers)
        cls.calls.append(
            {
                "prompts": condition.reshape(-1).tolist(),
                "grad": torch.is_grad_enabled(),
                "adapter": None if disabled else active,
                "latents": latents["image"].detach().clone(),
            }
        )
        scale = module(torch.ones(1, 8)).square().mean()
        return {
            name: value * (0.5 + 0.01 * scale) + 0.1 * sigmas[name].view(-1, 1, 1) for name, value in latents.items()
        }


# The engine tests import this module as ``dmd_engine_fixtures``, the H3 adapter tests as
# ``tests.workers.engine.dmd_engine_fixtures``: both copies must dispatch to the one class whose ``calls`` they read.
RecordingDMDModel = DiffusionModelBase.peek_class(ARCHITECTURE, "dmd2") or DiffusionModelBase.register(
    ARCHITECTURE, algorithm="dmd2"
)(_RecordingDMDModel)


def pin_engine_to_cpu() -> None:
    """Make the DMD engine put microbatches on the CPU, for a torchrun script of CPU FSDP2 runs.

    The engine moves each microbatch to verl's ``get_device_id()``, an accelerator wherever one is visible. An
    in-process test patches it with ``monkeypatch`` instead.
    """
    dmd_impl.get_device_id = lambda: torch.device("cpu")


def optimizer_config(lr: float, lr_warmup_steps: int = -1, **schedule) -> FSDPOptimizerConfig:
    """AdamW with beta1 = 0, beta2 = 0.9, no weight decay and gradient clipping at 1, at ``lr`` over 100 steps."""
    return FSDPOptimizerConfig(
        lr=lr,
        weight_decay=0.0,
        betas=(0.0, 0.9),
        clip_grad=1.0,
        lr_warmup_steps=lr_warmup_steps,
        total_training_steps=100,
        **schedule,
    )


def dmd_config(**overrides) -> DiffusionDMDConfig:
    """Four Euler steps on one ``image`` modality, one critic update per student update, microbatches of one."""
    settings = {
        "num_inference_steps": 4,
        "modality_shifts": {"image": 2.0},
        "modality_weights": {"image": 1.0},
        "score_discrete_steps": 0,
        "cfg_norm": "none",
        "teacher_guidance_scale": 1.0,
        "fake_update_ratio": 1,
        "fake_score_micro_batch_size_per_gpu": 1,
        "student_micro_batch_size_per_gpu": 1,
        "fake_score_optim": optimizer_config(CRITIC_LR),
    }
    return DiffusionDMDConfig(**{**settings, **overrides})


def model_config(architecture: str = ARCHITECTURE) -> SimpleNamespace:
    return SimpleNamespace(architecture=architecture, algorithm="dmd2", external_lib=None, lora_rank=2)


def make_dmd_engine(
    module: torch.nn.Module, config: DiffusionDMDConfig | None = None, architecture: str = ARCHITECTURE
) -> DMDDiffusersFSDPEngine:
    """Return an engine over ``module`` (adapters ``default`` and ``fake_score``) with both optimizers built.

    ``architecture`` selects the ``(architecture, dmd2)`` adapter; the default is ``RecordingDMDModel``.

    The engine runs its own ``__init__`` and ``_build_model_optimizer``. Only the base engine's steps are replaced:
    its ``__init__`` would build a device mesh, and its model build would load the module from a model config;
    here the caller owns the module, plain or FSDP2.
    """

    def base_init(engine, model_config, engine_config, optimizer_config, checkpoint_config):
        engine.model_config = model_config
        engine.engine_config = engine_config
        engine.optimizer_config = optimizer_config
        engine.checkpoint_config = checkpoint_config
        engine.rank = 0
        engine.ulysses_device_mesh = None
        engine.ulysses_sequence_parallel_size = 1
        engine._is_offload_param = engine._is_offload_optimizer = False

    def base_build_model_optimizer(engine):
        engine.module = module
        engine.optimizer = engine._build_optimizer(module)
        engine.lr_scheduler = engine._build_lr_scheduler(engine.optimizer)

    with (
        patch.object(DiffusersFSDPEngine, "__init__", base_init),
        patch.object(DiffusersFSDPEngine, "_build_model_optimizer", base_build_model_optimizer),
    ):
        engine = DMDDiffusersFSDPEngine(
            model_config(architecture),
            SimpleNamespace(strategy="fsdp2", forward_only=False, seed=3),
            optimizer_config(STUDENT_LR),
            None,
            dmd_config=config or dmd_config(),
        )
        engine._build_model_optimizer()
    return engine
