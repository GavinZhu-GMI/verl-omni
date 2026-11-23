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
"""The v1 ``distribution_matching`` trainer takes its prompts from the PDMD prompt stream, keyed by update and slot,
so a resumed run sends the worker group what the straight run sends.

The trainer runs on a real prompt cache of 14 prompts with a global batch of 2, through a fake worker group. A resume
at 3 falls inside a fake_score block, at 5 right before a student update, at 6 right after one; the 8th critic
update (update 8) starts epoch 1, so every resumed run crosses the epoch boundary.
"""

import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from verl.utils import tensordict_utils as tu

import verl_omni
from tests.utils.dataset._prompt_stream_fixtures import PdmdSampleStream, write_cache

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")
BUCKETS = [[4, 6, 1, 4, 4, 3], [8, 6, 2, 4, 4, 3], [4, 12, 1, 4, 4, 3]]
SEED, PROMPTS, GLOBAL_BATCH, FAKE_UPDATE_RATIO, UPDATES = 5, 14, 2, 5, 12


class RecordingWorkerGroup:
    """Record what each update receives and each checkpoint call; answer with finite metrics."""

    def __init__(self):
        self.updates: list[dict] = []
        self.saved: list[tuple[str, int]] = []
        self.save_limits: list[tuple[str | None, int | None]] = []
        self.loaded: list[str] = []

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        os.makedirs(local_path)
        self.saved.append((local_path, global_step))
        self.save_limits.append((hdfs_path, max_ckpt_to_keep))

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        self.loaded.append(local_path)

    def update_actor(self, data):
        self.updates.append(
            {
                "stage": tu.get(data, "dmd_stage"),
                "update": tu.get(data, "dmd_update"),
                "prompt_index": data["prompt_index"].tolist(),
                "latent_meta": data["latent_meta"].tolist(),
                "global_sample": data["global_sample"].tolist(),
            }
        )
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"grad_norm": [1.0], "loss": [0.5]}})


def make_trainer(
    cache_dir, total_steps=UPDATES, *, checkpoints=None, save_freq=-1, init=True, resume_mode="auto", overrides=()
):
    from verl_omni.trainer.diffusion.v1 import get_diffusion_trainer_cls

    buckets = "[" + ",".join("[" + ",".join(map(str, bucket)) + "]" for bucket in BUCKETS) + "]"
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                "trainer.v1.trainer_mode=distribution_matching",
                f"trainer.total_training_steps={total_steps}",
                "actor_rollout_ref.model.algorithm=dmd2",
                "actor_rollout_ref.model.model_type=diffusion_dmd_model",
                f"data.train_files={cache_dir}",
                f"data.latent_meta_buckets={buckets}",
                f"data.seed={SEED}",
                f"actor_rollout_ref.actor.fsdp_config.seed={SEED}",  # one seed, as the PDMD reference has
                f"data.train_batch_size={GLOBAL_BATCH}",
                f"actor_rollout_ref.actor.ppo_mini_batch_size={GLOBAL_BATCH}",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                f"dmd.fake_update_ratio={FAKE_UPDATE_RATIO}",
                f"trainer.save_freq={save_freq}",
                f"trainer.default_local_dir={checkpoints or cache_dir / 'no-checkpoints'}",
                f"trainer.resume_mode={resume_mode}",
                *overrides,
            ],
        )
    trainer = get_diffusion_trainer_cls("distribution_matching")(config=config)
    trainer._build_worker_group = RecordingWorkerGroup
    trainer._build_logger = lambda: SimpleNamespace(log=lambda data, step: None)
    if init:
        trainer.init()
    return trainer


def test_each_critic_update_takes_pdmds_prompts_and_a_student_update_its_critic_update_before(tmp_path):
    write_cache(tmp_path, [[2, 3, 1, 4, 2, 3, 1], [1, 2, 3, 2, 4, 1, 2]])
    trainer = make_trainer(tmp_path)
    reference = PdmdSampleStream(PROMPTS, GLOBAL_BATCH, FAKE_UPDATE_RATIO, BUCKETS, SEED)

    trainer.fit()

    updates = trainer.actor_rollout_wg.updates
    assert [record["update"] for record in updates] == list(range(UPDATES))
    assert [record["stage"] for record in updates] == (["fake_score"] * 5 + ["student"]) * 2
    for update, record in enumerate(updates):
        assert record["global_sample"] == list(range(GLOBAL_BATCH))
        critic_update = update if record["stage"] == "fake_score" else update - 1
        assert record["prompt_index"] == [reference.index(critic_update, slot) for slot in range(GLOBAL_BATCH)]
        assert record["latent_meta"] == [reference.bucket(critic_update, slot) for slot in range(GLOBAL_BATCH)]
    # the 7th critic update (update 7) ends epoch 0, the 8th (update 8) starts epoch 1
    critic_prompts = [
        index for record in updates if record["stage"] == "fake_score" for index in record["prompt_index"]
    ]
    assert sorted(critic_prompts[:PROMPTS]) == list(range(PROMPTS))


def global_rng_states():
    return torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate()


@pytest.mark.parametrize("point", [3, 5, 6])
def test_a_resumed_run_sends_each_update_what_the_straight_run_sends(tmp_path, point):
    """After a resume at ``point``, each update gets the straight run's stage, update and prompts, and
    loading the checkpoint leaves every global RNG as the test seeded it."""
    write_cache(tmp_path, [[2, 3, 1, 4, 2, 3, 1], [1, 2, 3, 2, 4, 1, 2]])
    straight = make_trainer(tmp_path)
    straight.fit()

    checkpoints = tmp_path / "checkpoints"
    first = make_trainer(tmp_path, total_steps=point, checkpoints=checkpoints, save_freq=point)
    first.fit()
    assert [step for _, step in first.actor_rollout_wg.saved] == [point]
    resumed = make_trainer(tmp_path, checkpoints=checkpoints, init=False)
    load_checkpoint = resumed._load_checkpoint
    rng = {}

    def reseeded_load():
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)
        rng["before"] = global_rng_states()
        load_checkpoint()
        rng["after"] = global_rng_states()

    resumed._load_checkpoint = reseeded_load
    resumed.init()
    resumed.fit()

    assert resumed.actor_rollout_wg.loaded == [str(checkpoints / f"global_step_{point}" / "actor")]
    assert resumed.actor_rollout_wg.updates == straight.actor_rollout_wg.updates[point:]
    assert resumed.global_steps == UPDATES
    before, after = rng["before"], rng["after"]
    assert torch.equal(before[0], after[0]) and np.array_equal(before[1], after[1]) and before[2] == after[2]


def test_a_run_without_a_checkpoint_starts_at_update_0(tmp_path):
    write_cache(tmp_path, [[2, 3, 1, 4, 2, 3, 1], [1, 2, 3, 2, 4, 1, 2]])
    trainer = make_trainer(tmp_path, total_steps=2, checkpoints=tmp_path / "empty")
    trainer.fit()
    assert trainer.actor_rollout_wg.loaded == []
    assert [record["update"] for record in trainer.actor_rollout_wg.updates] == [0, 1]


def test_a_data_seed_other_than_the_engine_seed_is_refused(tmp_path):
    write_cache(tmp_path, [[2, 3]])
    with pytest.raises(ValueError, match="one seed"):
        make_trainer(tmp_path, overrides=[f"actor_rollout_ref.actor.fsdp_config.seed={SEED + 1}"])


def test_a_checkpoint_hands_the_hdfs_dir_and_the_keep_limit_to_the_worker_group(tmp_path):
    """The DMD engine refuses both; the trainer passes them on instead of dropping them."""
    write_cache(tmp_path, [[2, 3]])
    trainer = make_trainer(
        tmp_path,
        total_steps=1,
        checkpoints=tmp_path / "checkpoints",
        save_freq=1,
        overrides=[f"trainer.default_hdfs_dir={tmp_path / 'hdfs'}", "trainer.max_actor_ckpt_to_keep=2"],
    )
    trainer.fit()
    assert trainer.actor_rollout_wg.save_limits == [(str(tmp_path / "hdfs" / "global_step_1" / "actor"), 2)]


@pytest.mark.parametrize("resume_from_path", ["null", "[a]"])
def test_a_resume_path_that_is_not_one_string_is_refused_by_name(tmp_path, resume_from_path):
    write_cache(tmp_path, [[2, 3]])
    with pytest.raises(ValueError, match="trainer.resume_from_path"):
        make_trainer(tmp_path, resume_mode="resume_path", overrides=[f"trainer.resume_from_path={resume_from_path}"])
