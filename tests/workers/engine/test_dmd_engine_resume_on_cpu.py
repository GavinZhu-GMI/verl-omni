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
"""A resumed DMD engine run equals the straight run bit for bit, and an update depends on its keys alone.

Two-rank CPU FSDP2 on a tiny MiniMax H3 (see dmd_engine_resume_run.py). A resume at 3 falls inside a fake_score
block, at 5 right before a student update, at 6 right after one.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from dmd_engine_resume_run import FAKE_UPDATE_RATIO, RESUME_POINTS, UPDATES, stage_of

from tests.utils.dataset._prompt_stream_fixtures import pdmd_stream_seed

REPO_ROOT = Path(__file__).resolve().parents[3]
RANKS = 2
NUM_INFERENCE_STEPS = 4  # dmd_engine_fixtures.dmd_config


def _run(out: Path, *args: str) -> None:
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join(filter(None, (str(REPO_ROOT), str(Path(__file__).parent), env.get("PYTHONPATH")))),
        TORCH_COMPILE_DISABLE="1",
        TORCHINDUCTOR_DISABLE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={RANKS}",
            str(Path(__file__).with_name("dmd_engine_resume_run.py")),
            str(out),
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    """Each mode's evidence per rank: ``evidence[mode][rank]``, modes ``straight`` and ``resumed-<k>``."""
    out = tmp_path_factory.mktemp("dmd-engine-resume")
    _run(out, "straight")
    for point in RESUME_POINTS:
        _run(out, "resumed", "--point", str(point))
    modes = ["straight", *(f"resumed-{point}" for point in RESUME_POINTS)]
    return {
        mode: [torch.load(out / f"{mode}-rank{rank}.pt", weights_only=False) for rank in range(RANKS)] for mode in modes
    }


def assert_equal_tree(actual, expected, where: str) -> None:
    """Equal bit for bit: tensors by ``torch.equal`` with equal dtypes, containers key by key."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype, where
        assert torch.equal(actual, expected), where
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), where
        for key in expected:
            assert_equal_tree(actual[key], expected[key], f"{where}/{key}")
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected), where
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_equal_tree(left, right, f"{where}[{index}]")
    else:
        assert actual == expected, where


def test_the_schedule_resumes_inside_a_block_before_and_after_a_student_update():
    stages = [stage_of(update) for update in range(UPDATES)]
    assert stages == (["fake_score"] * FAKE_UPDATE_RATIO + ["student"]) * 2
    assert (stages[2], stages[3]) == ("fake_score", "fake_score")  # k = 3
    assert (stages[4], stages[5]) == ("fake_score", "student")  # k = 5
    assert (stages[5], stages[6]) == ("student", "fake_score")  # k = 6


@pytest.mark.parametrize("point", RESUME_POINTS)
def test_a_resumed_run_ends_with_the_straight_runs_adapters_and_optimizers(evidence, point):
    """After all updates, both adapters and both optimizer states equal the straight run's, on every rank."""
    for rank in range(RANKS):
        straight, resumed = evidence["straight"][rank], evidence[f"resumed-{point}"][rank]
        assert_equal_tree(resumed["final_adapters"], straight["final_adapters"], f"rank {rank} adapters")
        assert_equal_tree(resumed["final_optimizers"], straight["final_optimizers"], f"rank {rank} optimizers")
        assert set(straight["final_optimizers"]) == {"default", "fake_score"}
        assert all(straight["final_optimizers"][name]["state"] for name in ("default", "fake_score"))


@pytest.mark.parametrize("point", RESUME_POINTS)
def test_a_load_restores_the_adapters_and_optimizers_the_save_held(evidence, point):
    for rank in range(RANKS):
        saved, loaded = evidence["straight"][rank]["saved"][point], evidence[f"resumed-{point}"][rank]["loaded"]
        assert_equal_tree(loaded["adapters"], saved["adapters"], f"rank {rank} adapters")
        assert_equal_tree(loaded["optimizers"], saved["optimizers"], f"rank {rank} optimizers")


@pytest.mark.parametrize("point", RESUME_POINTS)
def test_every_resumed_update_has_the_straight_runs_loss(evidence, point):
    """Every resumed update has the straight run's loss and adapters. At k = 5 the first resumed update is a student
    update, which runs on rebuilt rollouts the loss the straight run computed on its kept ones."""
    for rank in range(RANKS):
        straight, resumed = evidence["straight"][rank], evidence[f"resumed-{point}"][rank]
        assert len(resumed["records"]) == UPDATES - point
        for update, record in zip(range(point, UPDATES), resumed["records"], strict=True):
            assert record["loss"] == straight["records"][update]["loss"], f"rank {rank} update {update}"
            assert_equal_tree(
                record["adapters_after"], straight["records"][update]["adapters_after"], f"rank {rank} u{update}"
            )
    assert stage_of(5) == "student"


@pytest.mark.parametrize("point", RESUME_POINTS)
def test_loading_a_checkpoint_restores_no_global_rng(evidence, point):
    """Every global RNG holds after the load what the test seeded right before it."""
    assert all(evidence[f"resumed-{point}"][rank]["load_kept_rng"] for rank in range(RANKS))


@pytest.mark.parametrize("update", range(UPDATES))
def test_an_update_alone_on_a_fresh_engine_reproduces_the_straight_update(evidence, update):
    """With the straight run's adapters and optimizer states copied in, other global RNG seeds and, for a student
    update, no kept rollout, the update computes the straight run's loss and adapters."""
    for rank in range(RANKS):
        straight = evidence["straight"][rank]
        isolated, expected = straight["isolated"][update], straight["records"][update]
        assert isolated["loss"] == expected["loss"], f"rank {rank}"
        assert_equal_tree(isolated["adapters_after"], expected["adapters_after"], f"rank {rank}")


@pytest.mark.parametrize("update", [update for update in range(UPDATES) if stage_of(update) == "student"])
def test_a_student_update_uses_pdmds_keyed_exit_index(evidence, update):
    """The exit index is pdmd's draw keyed by (seed, "exit", update), the same on every rank and microbatch."""
    seed = evidence["straight"][0]["seed"]
    generator = torch.Generator().manual_seed(pdmd_stream_seed(seed, "exit", update))
    expected = float(int(torch.randint(NUM_INFERENCE_STEPS, (1,), generator=generator)))
    for rank in range(RANKS):
        assert evidence["straight"][rank]["records"][update]["exit_index"] == [expected, expected]
