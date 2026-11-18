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
"""Diffusers FSDP2 engine that trains several LoRA adapters of one frozen base, for DMD-style distillation."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
from tensordict import TensorDict
from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.config.optimizer import build_optimizer

from verl_omni.workers.config import DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine
from verl_omni.workers.engine.lora_adapter_mixin import adapter_parameters

__all__ = ["DMDDiffusersFSDPEngine"]


@dataclass
class _AdapterOptimizer:
    config: FSDPOptimizerConfig
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler


class DMDDiffusersFSDPEngine(DiffusersFSDPEngine):
    """Train several LoRA adapters of one frozen base, each with its own optimizer and LR scheduler.

    The ``default`` adapter trains with the engine's optimizer config; ``add_adapter_optimizer`` gives another
    adapter its own. ``train_stage(adapter)`` selects the adapter that one stage's forward, backward and optimizer
    calls train; outside a stage they act on ``default``, as in the single-adapter engine. Inside a stage,
    ``use_adapter`` returns to the stage's adapter on exit, so that backward and its activation-checkpoint
    recompute run under the trained adapter. FSDP2 only: each adapter clips its own gradients.
    """

    def __init__(
        self,
        model_config: DiffusionModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        if engine_config.strategy != "fsdp2":
            raise ValueError(f"{type(self).__name__} needs strategy=fsdp2, got {engine_config.strategy!r}.")
        if model_config.lora_rank <= 0:
            raise ValueError(f"{type(self).__name__} trains LoRA adapters; set lora_rank > 0.")
        if engine_config.forward_only:
            raise ValueError(f"{type(self).__name__} trains; forward_only engines have no optimizer.")
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _build_model_optimizer(self):
        super()._build_model_optimizer()
        self._init_adapter_stages()

    def _init_adapter_stages(self) -> None:
        """Register the ``default`` adapter's optimizer, built by the base engine; no stage is open."""
        assert self.optimizer is not None and self.lr_scheduler is not None  # forward_only is rejected in __init__
        self._adapter_optimizers = {
            "default": _AdapterOptimizer(self.optimizer_config, self.optimizer, self.lr_scheduler)
        }
        self._stage_adapter: Optional[str] = None

    def _build_optimizer(self, module):
        return build_optimizer(adapter_parameters(module, "default"), self.optimizer_config)

    def add_adapter_optimizer(self, adapter: str, optimizer_config: FSDPOptimizerConfig) -> None:
        """Give ``adapter`` its own optimizer and LR scheduler, built from ``optimizer_config``.

        Raises:
            ValueError: ``adapter`` is ``default``, already has an optimizer, or has no parameters.
        """
        if adapter in self._adapter_optimizers:
            raise ValueError(f"Adapter {adapter!r} already has an optimizer.")
        optimizer = build_optimizer(adapter_parameters(self.module, adapter), optimizer_config)
        self._adapter_optimizers[adapter] = _AdapterOptimizer(
            optimizer_config, optimizer, self._lr_scheduler_from(optimizer, optimizer_config)
        )

    @contextmanager
    def train_stage(self, adapter: str) -> Iterator[None]:
        """Train ``adapter`` inside the context; on exit ``default`` is active again.

        Raises:
            RuntimeError: A stage is already open; stages do not nest.
            ValueError: ``adapter`` has no optimizer.
        """
        if self._stage_adapter is not None:
            raise RuntimeError(f"The engine is already in the {self._stage_adapter!r} stage; stages do not nest.")
        if adapter not in self._adapter_optimizers:
            raise ValueError(f"Adapter {adapter!r} has no optimizer; call add_adapter_optimizer first.")
        self._stage_adapter = adapter
        self._set_adapter(adapter)
        try:
            yield
        finally:
            self._stage_adapter = None
            self._set_adapter("default")

    @contextmanager
    def use_adapter(self, name: str):
        """Select ``name`` temporarily; inside a stage, return to the stage's adapter instead of ``default``."""
        stage_adapter = self._stage_adapter
        if stage_adapter is None or name == "reference":
            with super().use_adapter(name):
                yield
            return
        self._set_adapter(name)
        try:
            yield
        finally:
            self._set_adapter(stage_adapter)

    def _stage_optimizer(self) -> _AdapterOptimizer:
        """The optimizer of the open stage's adapter, or of ``default`` outside a stage."""
        return self._adapter_optimizers["default" if self._stage_adapter is None else self._stage_adapter]

    def optimizer_zero_grad(self):
        self._stage_optimizer().optimizer.zero_grad()

    def optimizer_step(self) -> float:
        stage = self._stage_optimizer()
        assert stage.config.clip_grad is not None
        parameters = [param for group in stage.optimizer.param_groups for param in group["params"]]
        return self._clip_and_step(stage.optimizer, parameters, stage.config.clip_grad)

    def lr_scheduler_step(self) -> float:
        return self._step_lr_scheduler(self._stage_optimizer().lr_scheduler)

    def _optimizers(self) -> list[torch.optim.Optimizer]:
        return [state.optimizer for state in self._adapter_optimizers.values()]

    def save_checkpoint(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} cannot checkpoint yet: the checkpoint manager saves one optimizer, and every "
            "other adapter's optimizer state would be lost."
        )

    def load_checkpoint(self, *args, **kwargs) -> None:
        raise NotImplementedError(f"{type(self).__name__} cannot load a checkpoint yet; see save_checkpoint.")

    def forward_backward_batch(
        self, data: TensorDict, loss_function: Callable, forward_only: bool = False
    ) -> list[TensorDict]:
        raise NotImplementedError(f"{type(self).__name__} has no batch forward path yet.")

    def prepare_model_inputs(self, micro_batch: TensorDict, step: int):
        raise NotImplementedError(f"{type(self).__name__} has no batch forward path yet.")

    def prepare_model_outputs(self, output, micro_batch: TensorDict):
        raise NotImplementedError(f"{type(self).__name__} has no batch forward path yet.")

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only, step):
        raise NotImplementedError(f"{type(self).__name__} has no batch forward path yet.")
