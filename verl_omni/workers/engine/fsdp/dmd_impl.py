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

import math
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from tensordict import TensorDict
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.utils.metric import Metric
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.engine import EngineRegistry

from verl_omni.pipelines.dmd2.program import DMD2Program, Trajectory
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionDMDConfig, DiffusionModelConfig
from verl_omni.workers.config.diffusion.dmd import check_constant_learning_rate
from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine
from verl_omni.workers.engine.lora_adapter_mixin import adapter_parameters

__all__ = ["DMD_MODEL_TYPE", "DMDDiffusersFSDPEngine"]

DMD_MODEL_TYPE = "diffusion_dmd_model"  # the engine registry key of the DMD engine

_STAGE_ADAPTERS = {"fake_score": "fake_score", "student": "default"}  # the adapter each DMD stage trains
_ADAPTER_STAGES = {adapter: stage for stage, adapter in _STAGE_ADAPTERS.items()}
_ROLE_ADAPTERS = {**_STAGE_ADAPTERS, "teacher": "reference"}
_NO_CONDITION = object()  # the condition outside a _conditioned context


@dataclass
class _AdapterOptimizer:
    config: FSDPOptimizerConfig
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler


@EngineRegistry.register(model_type=DMD_MODEL_TYPE, backend="fsdp2")  # the registry's default device
class DMDDiffusersFSDPEngine(DiffusersFSDPEngine):
    """Run DMD2 fake_score and student updates on LoRA adapters of one frozen base, one update per ``train_batch``.

    The student trains the ``default`` adapter with the engine's optimizer config; the fake score trains
    ``fake_score`` with ``dmd_config.fake_score_optim``; the teacher is the base with adapters off.
    ``train_stage(adapter)`` selects the adapter that one stage's forward, backward and optimizer calls train;
    outside a stage they act on ``default``, as in the single-adapter engine. Inside a stage, ``use_adapter``
    returns to the stage's adapter on exit, so that backward and its activation-checkpoint recompute run under the
    trained adapter. FSDP2 only: each adapter clips its own gradients.

    The ``(architecture, dmd2)`` adapter registered for the model config supplies ``dmd_condition(micro_batch)``,
    ``dmd_initial_noise(micro_batch, generators)`` and ``dmd_predict(module, latents, sigmas, condition, negative)``.
    """

    def __init__(
        self,
        model_config: DiffusionModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
        *,
        dmd_config: DiffusionDMDConfig,
    ):
        if engine_config.strategy != "fsdp2":
            raise ValueError(f"{type(self).__name__} needs strategy=fsdp2, got {engine_config.strategy!r}.")
        if model_config.lora_rank <= 0:
            raise ValueError(f"{type(self).__name__} trains LoRA adapters; set lora_rank > 0.")
        if engine_config.forward_only:
            raise ValueError(f"{type(self).__name__} trains; forward_only engines have no optimizer.")
        check_constant_learning_rate("student", optimizer_config)
        # the architecture adapter and the DMD program, before the model loads
        self.dmd_config = dmd_config
        self._model_adapter = DiffusionModelBase.get_class(model_config)
        self._program = DMD2Program(dmd_config, self._predict, engine_config.seed)
        self._condition: Any = _NO_CONDITION
        # the last fake_score update's rollout, condition and microbatch, by slot; None until one update keeps them
        self._kept_rollouts: Optional[list[tuple[Trajectory, Any, TensorDict]]] = None
        self._kept_update = -1
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _build_model_optimizer(self):
        """Build the model and the student's optimizer as the base engine does, then the fake score's optimizer."""
        super()._build_model_optimizer()
        assert self.optimizer is not None and self.lr_scheduler is not None  # forward_only is rejected in __init__
        critic_config = self.dmd_config.fake_score_optim
        critic_optimizer = build_optimizer(adapter_parameters(self.module, "fake_score"), critic_config)
        self._adapter_optimizers = {
            "default": _AdapterOptimizer(self.optimizer_config, self.optimizer, self.lr_scheduler),
            "fake_score": _AdapterOptimizer(
                critic_config, critic_optimizer, self._lr_scheduler_from(critic_optimizer, critic_config)
            ),
        }
        self._stage_adapter: Optional[str] = None
        self._last_optimized_adapter: Optional[str] = None

    def _build_optimizer(self, module):
        return build_optimizer(adapter_parameters(module, "default"), self.optimizer_config)

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
            raise ValueError(
                f"Adapter {adapter!r} has no optimizer; expected one of {sorted(self._adapter_optimizers)}."
            )
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

    def _current_adapter(self) -> str:
        """The open stage's adapter, or ``default`` outside a stage."""
        return "default" if self._stage_adapter is None else self._stage_adapter

    def optimizer_zero_grad(self):
        self._adapter_optimizers[self._current_adapter()].optimizer.zero_grad()

    def optimizer_step(self) -> float:
        """Clip and step the stage's optimizer; after a finite norm, also step that adapter's LR scheduler.

        The worker calls ``lr_scheduler_step`` after ``train_batch`` has left the stage, so the scheduler steps here.
        """
        adapter = self._current_adapter()
        adapter_optimizer = self._adapter_optimizers[adapter]
        assert adapter_optimizer.config.clip_grad is not None
        parameters = [param for group in adapter_optimizer.optimizer.param_groups for param in group["params"]]
        grad_norm = self._clip_and_step(adapter_optimizer.optimizer, parameters, adapter_optimizer.config.clip_grad)
        if math.isfinite(grad_norm):
            adapter_optimizer.lr_scheduler.step()
        self._last_optimized_adapter = adapter
        return grad_norm

    def lr_scheduler_step(self) -> float:
        """Return the LR of the adapter of the last ``optimizer_step`` call, without stepping any scheduler.

        Raises:
            RuntimeError: No ``optimizer_step`` has run yet.
        """
        if self._last_optimized_adapter is None:
            raise RuntimeError(
                "lr_scheduler_step reports the LR of the last optimized adapter, but no optimizer_step has run yet"
            )
        return float(self._adapter_optimizers[self._last_optimized_adapter].lr_scheduler.get_last_lr()[0])

    def _optimizers(self) -> list[torch.optim.Optimizer]:
        return [state.optimizer for state in self._adapter_optimizers.values()]

    def _checkpoint_state(self) -> dict[str, Any]:
        """The trained state: each adapter's parameters, each adapter's optimizer and LR scheduler.

        The adapter tensors are detached views of the parameters, so a load writes into the parameters. No RNG
        state is part of it: every DMD draw is keyed by update and sample (``verl_omni.pipelines.dmd2.program``).
        Each optimizer's state is read inside its adapter's stage: PEFT freezes the inactive adapters, and the
        state-dict helpers skip frozen parameters.
        """
        trained = {
            id(param)
            for state in self._adapter_optimizers.values()
            for group in state.optimizer.param_groups
            for param in group["params"]
        }
        optimizers = {}
        for name, state in self._adapter_optimizers.items():
            with self.train_stage(name):
                optimizers[name] = get_optimizer_state_dict(self.module, state.optimizer)
        return {
            "adapters": {
                name: param.detach() for name, param in self.module.named_parameters() if id(param) in trained
            },
            "optimizers": optimizers,
            "lr_schedulers": {
                name: state.lr_scheduler.state_dict() for name, state in self._adapter_optimizers.items()
            },
        }

    def _check_checkpoint_args(self, hdfs_path: Optional[str]) -> None:
        if hdfs_path is not None:
            raise NotImplementedError(f"{type(self).__name__} checkpoints to a local path only; got hdfs_path.")
        if self.is_param_offload_enabled or self.is_optimizer_offload_enabled:
            raise NotImplementedError(f"{type(self).__name__} cannot checkpoint with parameter or optimizer offload.")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Save every adapter's parameters, optimizer and LR scheduler as a sharded checkpoint under ``local_path``.

        The base engine's checkpoint manager holds one optimizer; this engine trains one per adapter. The
        checkpoint holds no RNG state. ``global_step`` is the caller's: the path names it.

        Raises:
            NotImplementedError: ``hdfs_path`` or ``max_ckpt_to_keep`` is set, or parameters or optimizer states
                are offloaded.
        """
        self._check_checkpoint_args(hdfs_path)
        if max_ckpt_to_keep is not None:
            raise NotImplementedError(f"{type(self).__name__} keeps every checkpoint; unset max_ckpt_to_keep.")
        dcp.save(self._checkpoint_state(), checkpoint_id=local_path)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        """Load what :meth:`save_checkpoint` saved into this engine's adapters, optimizers and LR schedulers.

        No global or private RNG changes. Kept rollouts are not part of a checkpoint: the next student update
        rebuilds them (see :meth:`forward_backward_batch`).

        Raises:
            NotImplementedError: ``hdfs_path`` is set, or parameters or optimizer states are offloaded.
        """
        self._check_checkpoint_args(hdfs_path)
        state = self._checkpoint_state()
        dcp.load(state, checkpoint_id=local_path)
        for name, adapter_optimizer in self._adapter_optimizers.items():
            with self.train_stage(name):  # see _checkpoint_state
                set_optimizer_state_dict(self.module, adapter_optimizer.optimizer, state["optimizers"][name])
            adapter_optimizer.lr_scheduler.load_state_dict(state["lr_schedulers"][name])
        self._kept_rollouts, self._kept_update = None, -1
        if del_local_after_load:
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(local_path)

    @contextmanager
    def _conditioned(self, condition: Any) -> Iterator[None]:
        """Run the predictions inside the context on ``condition``."""
        self._condition = condition
        try:
            yield
        finally:
            self._condition = _NO_CONDITION

    def _predict(
        self, role: str, latents: dict[str, torch.Tensor], sigmas: dict[str, torch.Tensor], negative: bool
    ) -> dict[str, torch.Tensor]:
        """Run ``role`` under its adapter on the condition of the open ``_conditioned`` context.

        Raises:
            RuntimeError: No ``_conditioned`` context is open.
        """
        if self._condition is _NO_CONDITION:
            raise RuntimeError("A DMD prediction ran outside a _conditioned context, so it has no condition.")
        with self.use_adapter(_ROLE_ADAPTERS[role]):
            return self._model_adapter.dmd_predict(self.module, latents, sigmas, self._condition, negative)

    @staticmethod
    def _stage_of(data: TensorDict) -> str:
        stage = tu.get_non_tensor_data(data, "dmd_stage", default=None)
        if stage not in _STAGE_ADAPTERS:
            raise ValueError(f"Unknown DMD stage {stage!r}; expected one of {sorted(_STAGE_ADAPTERS)}")
        return stage

    def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
        """Run one optimizer update of the stage that ``data`` names (``dmd_stage``), inside that stage."""
        with self.train_stage(_STAGE_ADAPTERS[self._stage_of(data)]):
            return super().train_batch(data, loss_function)

    def _roll_out(
        self, micro_batches: list[TensorDict], samples: list[tuple[int, ...]], update: int
    ) -> list[tuple[Trajectory, Any, TensorDict]]:
        """Roll out each microbatch on the draw keys of its global ``samples`` in ``update``; return its rollout,
        condition and microbatch."""
        rollouts = []
        for micro_batch, micro_samples in zip(micro_batches, samples, strict=True):
            condition = self._model_adapter.dmd_condition(micro_batch)
            keys = self._program.keys(update, micro_samples)
            with self._conditioned(condition):
                generators = keys.generators("noise", micro_batch.device)
                initial = self._model_adapter.dmd_initial_noise(micro_batch, generators)
                trajectory = self._program.rollout(initial, keys=keys)
            rollouts.append((trajectory, condition, micro_batch))
        return rollouts

    def _student_rollouts(
        self, micro_batches: list[TensorDict], samples: list[tuple[int, ...]], update: int
    ) -> list[tuple[Trajectory, Any, TensorDict]]:
        """Return the rollouts of the fake_score update before ``update``, kept or rebuilt, and forget the kept ones.

        The student reuses the last critic update's rollouts, as the reference implementation does
        (https://github.com/ZeamoxWang/pdmd, ``training/train.py``). An engine that has kept none since it was built,
        as after a resume, rebuilds them from the student batch, which then holds that update's prompts, on that
        update's draw keys.

        Raises:
            RuntimeError: The last fake_score update kept a different number of rollouts, or kept them in another
                update than the one before ``update``.
        """
        if self._kept_rollouts is None:
            rebuilt, self._kept_rollouts = self._roll_out(micro_batches, samples, update - 1), []
            return rebuilt
        if len(self._kept_rollouts) != len(micro_batches):
            raise RuntimeError(
                f"The student update has {len(micro_batches)} microbatches, but the last fake_score update kept "
                f"{len(self._kept_rollouts)} rollouts; each rollout serves one student update"
            )
        if self._kept_update != update - 1:
            raise RuntimeError(
                f"The student update {update} trains on the rollouts of update {update - 1}, but the last "
                f"fake_score update kept those of update {self._kept_update}"
            )
        kept, self._kept_rollouts = self._kept_rollouts, []
        return kept

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only: bool = False) -> dict:
        """Accumulate the open stage's gradients over its microbatches, one slot per microbatch.

        ``train_batch`` opens the stage that ``data`` names; the stage, not ``data``, decides what runs here.
        A fake_score update rolls out each microbatch and, once all its gradients are accumulated, keeps the rollout,
        condition and microbatch of each slot; the next student update trains each slot on them, with one exit index
        for all its microbatches. The student batch then sets only the number of slots; an engine that has kept no
        rollout since it was built rolls them out again from the student batch.

        Every random draw is keyed by ``data``'s ``dmd_update``, the 0-based update, and each row's ``global_sample``,
        its index in the global batch (see :mod:`verl_omni.pipelines.dmd2.program`); a rebuilt rollout uses the keys
        of update ``dmd_update - 1``. A student update reports its exit index as ``dmd/exit_index``.

        Raises:
            ValueError: ``forward_only`` is set, ``data`` is empty, asks for dynamic batch sizes or deferred
                gradient sync, does not split into whole microbatches, or lacks ``dmd_update`` or ``global_sample``.
            RuntimeError: No stage is open, or a student update finds the kept rollouts of another update or
                another number of microbatches.
        """
        if forward_only:
            raise ValueError(f"{type(self).__name__} has no forward-only path.")
        if self._stage_adapter is None:
            raise RuntimeError(f"{type(self).__name__} runs updates inside train_batch, which opens their stage.")
        stage = _ADAPTER_STAGES[self._stage_adapter]
        if stage == "fake_score":
            self._kept_rollouts = []  # a fake_score update that fails, even when rejected here, keeps no rollout
        if len(data) == 0:
            raise ValueError(f"The {stage} batch is empty; each update trains at least one microbatch.")
        if tu.get_non_tensor_data(data, "use_dynamic_bsz", default=True):  # verl's default when unset
            raise ValueError(
                f"{type(self).__name__} splits batches by the DMD microbatch size; set use_dynamic_bsz=False."
            )
        if tu.get_non_tensor_data(data, "use_no_sync_for_gradient_accumulation", default=False):
            raise ValueError(
                f"{type(self).__name__} syncs gradients on every microbatch; "
                "unset use_no_sync_for_gradient_accumulation."
            )
        micro_batch_size = self.dmd_config.student_micro_batch_size_per_gpu  # equal to fake_score's, by the config
        if len(data) % micro_batch_size:
            raise ValueError(f"A {stage} batch of {len(data)} does not split into microbatches of {micro_batch_size}.")
        slots = len(data) // micro_batch_size
        update = tu.get_non_tensor_data(data, "dmd_update", default=None)
        if isinstance(update, bool) or not isinstance(update, int) or update < (1 if stage == "student" else 0):
            raise ValueError(
                f"A {stage} batch needs its 0-based update as dmd_update, after a fake_score update for a student "
                f"update; got {update!r}."
            )
        if "global_sample" not in data.keys():
            raise ValueError(f"A {stage} batch needs global_sample, each row's index in the global batch.")
        chunks = tu.chunk_tensordict(data, slots)
        samples = [tuple(chunk["global_sample"].tolist()) for chunk in chunks]  # on the host, before the move
        micro_batches = [chunk.to(get_device_id()) for chunk in chunks]
        if stage == "student":
            slot_inputs = self._student_rollouts(micro_batches, samples, update)
            exit_index = self._program.draw_exit_index(update)
        elif stage == "fake_score":
            slot_inputs = self._roll_out(micro_batches, samples, update)
            exit_index = None
        else:
            raise ValueError(f"Unknown DMD stage {stage!r}")
        losses, metrics = [], {}
        for (trajectory, condition, micro_batch), micro_samples in zip(slot_inputs, samples, strict=True):
            tu.assign_non_tensor(
                micro_batch, gradient_accumulation_steps=slots, sp_size=self.ulysses_sequence_parallel_size
            )
            with self._conditioned(condition):
                loss, micro_metrics, invalid = self._program.compute_loss(
                    trajectory,
                    stage,
                    micro_batch,
                    loss_function,
                    exit_index=exit_index,
                    keys=self._program.keys(update, micro_samples),
                    dp_group=self.get_data_parallel_group(),
                )
            loss.backward()
            losses.append(loss.detach().item())
            micro_metrics["dmd/nonfinite_fraction"] = float(invalid)
            if exit_index is not None:
                micro_metrics["dmd/exit_index"] = float(exit_index)
            for key, value in micro_metrics.items():
                value = value.aggregate() if isinstance(value, Metric) else value
                metrics.setdefault(key, []).append(float(value))  # one value per microbatch, as verl's engines
        if stage == "fake_score":
            self._kept_rollouts, self._kept_update = slot_inputs, update
        return {"loss": losses, "metrics": metrics, "model_output": {}}

    def prepare_model_inputs(self, micro_batch: TensorDict, step: int):
        raise NotImplementedError(f"{type(self).__name__} runs whole updates in forward_backward_batch.")

    def prepare_model_outputs(self, output, micro_batch: TensorDict):
        raise NotImplementedError(f"{type(self).__name__} runs whole updates in forward_backward_batch.")

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only, step):
        raise NotImplementedError(f"{type(self).__name__} runs whole updates in forward_backward_batch.")
