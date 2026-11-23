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
"""The training worker of ``distribution_matching``: the DMD engine, its config, and whole updates."""

import os
from contextlib import contextmanager
from inspect import unwrap
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.engine import EngineRegistry

import verl_omni
from verl_omni.trainer.diffusion.v1.trainer_distribution_matching import (
    DistributionMatchingDiffusionTrainerV1,
    check_dmd_update_config,
)
from verl_omni.workers import engine_workers
from verl_omni.workers.config import DiffusionDMDConfig
from verl_omni.workers.engine.fsdp import DMDDiffusersFSDPEngine
from verl_omni.workers.engine_workers import DMD_MODEL_TYPE, ActorRolloutRefWorker, TrainingWorker

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")
BATCH = 4


def compose_config(model_path, *overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="diffusion_trainer",
            overrides=[
                "trainer.v1.trainer_mode=distribution_matching",
                f"actor_rollout_ref.model.path={model_path}",
                "+actor_rollout_ref.model.architecture=MiniMaxH3Pipeline",
                "actor_rollout_ref.model.algorithm=dmd2",
                f"actor_rollout_ref.model.model_type={DMD_MODEL_TYPE}",
                "+actor_rollout_ref.model.load_tokenizer=false",
                f"data.train_batch_size={BATCH}",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={BATCH}",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
                "dmd.student_micro_batch_size_per_gpu=2",
                "dmd.fake_score_micro_batch_size_per_gpu=2",
                "dmd.fake_update_ratio=5",
                *overrides,
            ],
        )


class RecordingTrainingWorker:
    """Stands in for ``TrainingWorker`` inside ``ActorRolloutRefWorker.init_model``; records what it was built with."""

    built: list["RecordingTrainingWorker"] = []

    def __init__(self, config, dmd_config=None):
        self.config, self.dmd_config = config, dmd_config
        RecordingTrainingWorker.built.append(self)

    def reset(self):
        pass

    def set_loss_fn(self, loss_fn):
        pass

    def get_dispatch_collect(self):
        return {}


def build_actor(monkeypatch, config, dmd_config="from config"):
    """Run ``init_model`` of an actor worker without Ray, a process group or a model; return the engine's builds."""
    RecordingTrainingWorker.built = []
    monkeypatch.setattr(engine_workers, "TrainingWorker", RecordingTrainingWorker)
    monkeypatch.setattr(engine_workers.CheckpointEngineRegistry, "new", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_workers.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(engine_workers, "aggressive_empty_cache", lambda force_sync: None)
    worker = object.__new__(ActorRolloutRefWorker)
    worker.config, worker.role = config.actor_rollout_ref, "actor"
    worker.dmd_config = config.dmd if dmd_config == "from config" else dmd_config
    worker.distillation_config, worker.distillation_enabled, worker.teacher_key = None, False, None
    worker.set_dispatch_collect = lambda mesh_name, **kwargs: None
    ActorRolloutRefWorker.init_model.__wrapped__(worker)
    return RecordingTrainingWorker.built


def test_the_dmd_model_type_resolves_to_the_dmd_engine(monkeypatch):
    monkeypatch.setenv("VERL_ENGINE_DEVICE", "cuda")
    assert EngineRegistry.get_engine_cls(DMD_MODEL_TYPE, "fsdp2") is DMDDiffusersFSDPEngine


def test_the_actor_builds_its_engine_with_the_trainer_dmd_config(monkeypatch, tmp_path):
    config = compose_config(tmp_path, "dmd.fake_score_optim.lr=3e-6", "dmd.projected=true")

    (actor,) = build_actor(monkeypatch, config)

    # the microbatch size the worker sends, which check_dmd_update_config holds equal to the DMD one
    assert actor.config.engine_config.micro_batch_size_per_gpu == config.dmd.student_micro_batch_size_per_gpu == 2

    assert actor.config.model_type == DMD_MODEL_TYPE
    assert actor.dmd_config == omega_conf_to_dataclass(config.dmd, DiffusionDMDConfig)
    assert actor.dmd_config.fake_score_optim.lr == 3e-6 and actor.dmd_config.projected


def test_the_training_worker_hands_the_dmd_config_to_the_engine_and_only_for_dmd(monkeypatch, tmp_path):
    built = []

    def new(*, model_type, backend, **kwargs):
        built.append((model_type, kwargs))
        return SimpleNamespace(get_data_parallel_rank=lambda: 0, is_mp_src_rank_with_outputs=lambda: True)

    monkeypatch.setattr(engine_workers.Worker, "__init__", lambda self: setattr(self, "_rank", 0))
    monkeypatch.setattr(engine_workers, "initialize_global_process_group_ray", lambda timeout_second: None)
    monkeypatch.setattr(engine_workers, "set_numa_affinity", lambda: None)
    monkeypatch.setattr(engine_workers.DistProfilerExtension, "__init__", lambda self, profiler: None)
    monkeypatch.setattr(EngineRegistry, "new", new)
    monkeypatch.setattr(TrainingWorker, "_register_dispatch_collect_info", lambda self, **kwargs: None)
    model_config = SimpleNamespace(get=lambda key, default=None: default, architecture=None, transformer_config=None)

    def worker_config(model_type):
        return SimpleNamespace(
            model_config=model_config,
            engine_config=SimpleNamespace(strategy="fsdp2"),
            optimizer_config=None,
            checkpoint_config=None,
            profiler_config=None,
            model_type=model_type,
        )

    dmd_config = DiffusionDMDConfig()
    TrainingWorker(worker_config(DMD_MODEL_TYPE), dmd_config=dmd_config)
    assert built == [(DMD_MODEL_TYPE, {**built[0][1], "dmd_config": dmd_config})]
    with pytest.raises(ValueError, match="dmd_config is required"):
        TrainingWorker(worker_config(DMD_MODEL_TYPE))
    with pytest.raises(ValueError, match="dmd_config is required"):
        TrainingWorker(worker_config("diffusion_model"), dmd_config=dmd_config)


@pytest.mark.parametrize(
    "override, name",
    [
        ("actor_rollout_ref.actor.ppo_epochs=2", "ppo_epochs"),
        (f"actor_rollout_ref.actor.ppo_mini_batch_size={BATCH // 2}", "ppo_mini_batch_size"),
        ("actor_rollout_ref.model.model_type=diffusion_model", "model_type"),
        ("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1", "ppo_micro_batch_size_per_gpu"),
        ("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null", "ppo_micro_batch_size_per_gpu"),
    ],
)
def test_a_config_that_would_split_an_update_or_send_other_microbatches_is_refused_at_build_time(
    tmp_path, override, name
):
    config = compose_config(tmp_path, override)
    trainer = DistributionMatchingDiffusionTrainerV1(config=config)
    trainer._build_dataloader = trainer._build_worker_group = trainer._build_logger = pytest.fail
    with pytest.raises(ValueError, match=name):
        trainer.init()
    check_dmd_update_config(compose_config(tmp_path))


class UpdateRecorder:
    """The trainer side: keep each update as the worker would receive it."""

    def __init__(self):
        self.updates = []

    def update_actor(self, data):
        self.updates.append(data.clone())
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"loss": [0.0]}})


@pytest.mark.parametrize("data_parallel_size", [1, 2])
def test_each_update_reaches_the_engine_whole_without_dynamic_batching(tmp_path, data_parallel_size):
    """Each rank's shard of the trainer's update (here rank 0's) reaches its engine as one batch, once."""
    trainer = DistributionMatchingDiffusionTrainerV1(config=compose_config(tmp_path))
    trainer.train_dataloader = [TensorDict({"prompt_ids": torch.arange(BATCH)[:, None]}, batch_size=[BATCH])]
    trainer.actor_rollout_wg = UpdateRecorder()
    trainer.logger = SimpleNamespace(log=lambda data, step: None)
    trainer.total_training_steps = 6
    trainer.fit()

    rank_batch = BATCH // data_parallel_size
    for update in trainer.actor_rollout_wg.updates:
        received: list[list[int]] = []

        @contextmanager
        def train_mode(disable_auto_offload):
            yield

        engine = SimpleNamespace(
            get_data_parallel_size=lambda: data_parallel_size,
            get_data_parallel_rank=lambda: 0,
            train_mode=train_mode,
            is_mp_src_rank_with_outputs=lambda: False,
        )
        worker = SimpleNamespace(
            engine=engine,
            train_batch=lambda data, received=received: received.append(data["prompt_ids"].flatten().tolist()),
            profiler=SimpleNamespace(step=lambda: None),
        )
        rank_shard = update[:rank_batch]
        # the DMD engine splits by its own microbatch size and refuses dynamic batch sizes
        assert tu.get_non_tensor_data(rank_shard, "use_dynamic_bsz", default=None) is False
        unwrap(TrainingWorker.train_mini_batch)(worker, rank_shard)
        assert received == [list(range(rank_batch))]


def test_a_dmd_actor_without_a_dmd_config_is_refused_instead_of_training_on_defaults(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="needs dmd_config"):
        build_actor(monkeypatch, compose_config(tmp_path), dmd_config=None)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["data.train_files=[a,b]"], "data.train_files must be one prompt-cache directory"),
        (["data.seed=0"], "Set data.latent_meta_buckets"),
        (["data.latent_meta_buckets=[[8,10,2,4,4,5]]"], "Set data.seed"),
        (
            ["data.seed=0", "data.latent_meta_buckets=[[8,10,2,4,4,5],[16,10,4,4,4,5]]"],
            "Several data.latent_meta_buckets need dmd.student_micro_batch_size_per_gpu",
        ),
    ],
)
def test_a_dataloader_config_the_dmd_engine_cannot_train_on_is_refused_by_name(tmp_path, overrides, message):
    trainer = DistributionMatchingDiffusionTrainerV1(config=compose_config(tmp_path, *overrides))
    with pytest.raises(ValueError, match=message):
        trainer._build_dataloader()
