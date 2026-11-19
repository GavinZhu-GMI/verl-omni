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
"""DMD engine contracts on a tiny MiniMax H3 under two-rank CPU FSDP2 (see dmd_engine_fsdp2_contracts.py)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from dmd_engine_fixtures import ADAPTERS

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    """Run the two-rank script once and load rank 0's evidence."""
    path = tmp_path_factory.mktemp("dmd-engine-fsdp2") / "evidence.json"
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join(filter(None, (str(REPO_ROOT), env.get("PYTHONPATH")))),
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
            "--nproc_per_node=2",
            str(Path(__file__).with_name("dmd_engine_fsdp2_contracts.py")),
            str(path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(path.read_text())


@pytest.mark.parametrize("role", ADAPTERS)
def test_trained_adapters_start_equal_to_the_teacher(evidence, role):
    """DMD starts the critic and the student at the teacher: with lora_B at 0 both reproduce it exactly."""
    assert evidence["init_gap"][role] == 0.0


@pytest.mark.parametrize("role", ADAPTERS)
def test_checkpoint_recompute_runs_under_the_trained_adapter(evidence, role):
    """Checkpointed gradients equal plain ones, and the recompute is observed, not assumed."""
    result = evidence["recompute"][role]
    assert result["checkpointed_block_calls"] == result["plain_block_calls"] + 1
    assert result["gradient_count"] > 0 and result["same_names"]
    assert result["max_gradient_norm"] > 0
    assert result["max_gradient_gap"] <= 1e-6


@pytest.mark.parametrize("role", ADAPTERS)
def test_stage_updates_only_its_adapter(evidence, role):
    """A stage's backward and step touch only the trained adapter's parameters."""
    result = evidence["isolation"][role]
    assert result["stray_gradients"] == []
    assert result["changed_other"] == 0
    assert result["changed_trained"] == result["trained_parameters"] > 0
    assert result["grad_norm"] > 0


@pytest.mark.parametrize("role", ADAPTERS)
def test_nonfinite_gradient_on_one_rank_skips_the_step_on_every_rank(evidence, role):
    """Every rank sees the same nonfinite norm, so the driver can abort without leaving a rank in a collective."""
    result = evidence["nonfinite"][role]
    assert result["poisoned"][0] is None and result["poisoned"][1] is not None
    assert result["grad_norms"] == ["nan", "nan"]
    assert result["changed"] == 0


def test_default_is_active_after_the_stages(evidence):
    """Stages and restoring contexts leave the student adapter active."""
    assert evidence["active_after"] == ["default"]
