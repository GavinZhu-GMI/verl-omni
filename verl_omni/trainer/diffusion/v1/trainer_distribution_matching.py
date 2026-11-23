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
"""The v1 ``distribution_matching`` trainer: DMD2 critic and student updates, one update per global step."""

import math
from collections.abc import Iterable, Iterator
from numbers import Real
from typing import Any

import ray
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from verl_omni.trainer.diffusion.v1.trainer_base import register_diffusion_trainer
from verl_omni.utils.dataset.prompt_cache_dataset import PromptCacheDataset, collate_prompt_features
from verl_omni.utils.dataset.rl_dataset import create_rl_sampler
from verl_omni.workers.config import DiffusionDMDConfig
from verl_omni.workers.engine_workers import DMD_MODEL_TYPE, ActorRolloutRefWorker


def _nonfinite_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return the metrics holding a nonfinite number; each value is a number or a list of numbers.

    Raises:
        TypeError: A value is neither, so its finiteness cannot be checked.
    """
    nonfinite = {}
    for key, values in metrics.items():
        for value in values if isinstance(values, list) else [values]:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Metric {key!r} holds a {type(value).__name__}; expected numbers")
            if not math.isfinite(value):
                nonfinite[key] = values
    return nonfinite


def check_dmd_update_config(config) -> None:
    """Refuse a config under which the worker would split one DMD update, send other microbatches, or build a non-DMD
    engine; the trainer sends ``actor.ppo_mini_batch_size`` and ``actor.ppo_epochs`` with each update.

    A fake_score update keeps one rollout per microbatch for the next student update, so it must reach the engine
    whole: one mini-batch, one epoch.

    Raises:
        ValueError: ``actor_rollout_ref.model.model_type`` is not the DMD engine's, ``actor.ppo_epochs`` is not 1,
            ``actor.ppo_mini_batch_size`` differs from ``data.train_batch_size``, or
            ``actor.ppo_micro_batch_size_per_gpu`` differs from ``dmd.student_micro_batch_size_per_gpu``.
    """
    model_type = config.actor_rollout_ref.model.get("model_type")
    if model_type != DMD_MODEL_TYPE:
        raise ValueError(
            f"distribution_matching trains the DMD engine; set actor_rollout_ref.model.model_type="
            f"{DMD_MODEL_TYPE}, got {model_type!r}."
        )
    actor = config.actor_rollout_ref.actor
    if actor.ppo_epochs != 1:
        raise ValueError(f"actor.ppo_epochs={actor.ppo_epochs}: each DMD update is one epoch; set it to 1.")
    if actor.ppo_mini_batch_size != config.data.train_batch_size:
        raise ValueError(
            f"actor.ppo_mini_batch_size={actor.ppo_mini_batch_size} differs from data.train_batch_size="
            f"{config.data.train_batch_size}: each DMD update is one mini-batch; set them equal."
        )
    if actor.ppo_micro_batch_size_per_gpu != config.dmd.student_micro_batch_size_per_gpu:
        raise ValueError(
            f"actor.ppo_micro_batch_size_per_gpu={actor.ppo_micro_batch_size_per_gpu} differs from "
            f"dmd.student_micro_batch_size_per_gpu={config.dmd.student_micro_batch_size_per_gpu}: the DMD engine "
            "trains on DMD microbatches; set them equal."
        )


@register_diffusion_trainer("distribution_matching")
class DistributionMatchingDiffusionTrainerV1:
    """Run ``fake_update_ratio`` critic updates, then one student update on the last critic update's prompts.

    A nonfinite number in an update's metrics stops training with ``FloatingPointError``, as the reference
    implementation does (https://github.com/ZeamoxWang/pdmd, ``training/train.py``). On a nonfinite gradient norm
    the engine has skipped that optimizer step, so counting it would break the critic-to-student ratio.
    """

    actor_rollout_wg: Any  # the worker group of the DMD engine, built by init()
    train_dataloader: Iterable[TensorDict]  # prompt batches, one per critic update, built by init()
    logger: Any  # the per-step metric log, built by init()

    def __init__(self, config):
        self.config = config
        self.dmd_config = omega_conf_to_dataclass(config.dmd, DiffusionDMDConfig)
        self.total_training_steps = config.trainer.total_training_steps
        self.global_steps = 0

    def init(self):
        """Check the config, then build the dataloader, the worker group and the metric log."""
        check_dmd_update_config(self.config)
        self.train_dataloader = self._build_dataloader()
        self.actor_rollout_wg = self._build_worker_group()
        self.logger = self._build_logger()

    def _build_dataloader(self) -> StatefulDataLoader:
        """Batches of ``data.train_batch_size`` precomputed prompts; the trainer loads no text encoder.

        Raises:
            ValueError: ``data.train_files`` is not one cache directory, ``data.latent_meta_buckets`` or ``data.seed``
                is unset, or a microbatch could hold two latent geometries.
        """
        data = self.config.data
        if not isinstance(data.train_files, str):
            raise ValueError(f"data.train_files must be one prompt-cache directory, got {data.train_files!r}.")
        if data.latent_meta_buckets is None:
            raise ValueError("Set data.latent_meta_buckets: the latent geometries a cached prompt draws from.")
        if data.seed is None:
            raise ValueError("Set data.seed: it seeds each prompt's draw from data.latent_meta_buckets.")
        if len(data.latent_meta_buckets) > 1 and self.dmd_config.student_micro_batch_size_per_gpu > 1:
            raise ValueError(
                "Several data.latent_meta_buckets need dmd.student_micro_batch_size_per_gpu=1: a microbatch shares one "
                "latent geometry, and each prompt draws its own."
            )
        dataset = PromptCacheDataset(data.train_files, OmegaConf.to_container(data.latent_meta_buckets), data.seed)
        return StatefulDataLoader(
            dataset=dataset,
            batch_size=data.train_batch_size,
            num_workers=data.dataloader_num_workers,
            drop_last=True,
            collate_fn=collate_prompt_features,
            sampler=create_rl_sampler(data, dataset),
        )

    def _build_worker_group(self) -> RayWorkerGroup:
        """One actor worker per GPU, each holding the DMD engine; no rollout or reference worker."""
        trainer = self.config.trainer
        pool_manager = ResourcePoolManager(
            resource_pool_spec={"global_pool": [trainer.n_gpus_per_node] * trainer.nnodes},
            mapping={Role.Actor: "global_pool"},
        )
        pool_manager.create_resource_pool()
        worker_cls = RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker),
            config=self.config.actor_rollout_ref,
            role="actor",
            dmd_config=self.config.dmd,
        )
        worker_group = RayWorkerGroup(
            resource_pool=pool_manager.get_resource_pool(Role.Actor),
            ray_cls_with_init=worker_cls,
            device_name=trainer.device,
        )
        worker_group.init_model()
        return worker_group

    def _build_logger(self) -> Tracking:
        return Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    def _prompt_batches(self) -> Iterator[TensorDict]:
        """Yield prompt batches without end, one pass over ``train_dataloader`` per epoch."""
        while True:
            empty = True
            for batch in self.train_dataloader:
                empty = False
                yield batch
            if empty:
                raise ValueError("train_dataloader yields no prompt batch")

    def fit(self):
        """Train until ``total_training_steps`` updates have run."""
        stages = self.dmd_config.training_stages()
        actor = self.config.actor_rollout_ref.actor
        batches = self._prompt_batches()
        batch = None
        while self.global_steps < self.total_training_steps:
            position = self.global_steps % len(stages)
            stage = stages[position]
            self.global_steps += 1
            if stage == "fake_score":
                batch = next(batches)
            # one mini-batch and one epoch (check_dmd_update_config): a fake_score update keeps a rollout per
            # microbatch for the student update; the DMD engine splits by its own microbatch size, never dynamically
            tu.assign_non_tensor(
                batch,
                dmd_stage=stage,
                mini_batch_size=actor.ppo_mini_batch_size,
                epochs=actor.ppo_epochs,
                dataloader_kwargs={"shuffle": False},
                use_dynamic_bsz=False,
            )
            output = self.actor_rollout_wg.update_actor(batch)
            metrics = tu.get(output, "metrics")
            nonfinite = _nonfinite_metrics(metrics)
            self.logger.log(
                data={**reduce_metrics(metrics), "dmd/student_step": float(stage == "student")}, step=self.global_steps
            )
            if nonfinite:
                label = (
                    f"fake_score {position + 1}/{self.dmd_config.fake_update_ratio}" if stage == "fake_score" else stage
                )
                raise FloatingPointError(
                    f"Nonfinite metrics {nonfinite} at global_step={self.global_steps}, stage={label}"
                )
