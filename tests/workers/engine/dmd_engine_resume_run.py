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
"""Two-rank CPU FSDP2 runs of the DMD engine on a tiny MiniMax H3 for the resume checks; each rank saves its evidence.

  straight    UPDATES updates, ``FAKE_UPDATE_RATIO`` fake_score updates per student update; a checkpoint after each
              update in RESUME_POINTS. Then, for each update u, a fresh engine receives the straight run's adapters and
              optimizer states from before u, copied in directly, every global RNG is reseeded apart, and u runs alone.
  resumed k   a new process reseeds every global RNG apart, loads the checkpoint of k and runs the remaining updates.

Every random setting of the DMD program is on: the renoise sampler, fresh critic noise and fresh score-query noise.
"""

import argparse
import copy
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from dmd_engine_fixtures import ADAPTERS, dmd_config, make_dmd_engine, pin_engine_to_cpu
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from verl.utils import tensordict_utils as tu

from tests.pipelines.minimax_h3_dmd2.h3_dmd2_fixtures import ARCHITECTURE, micro_batch, tiny_h3, tiny_row_widths
from verl_omni.workers.config.diffusion.actor import DiffusionActorConfig, DiffusionLossConfig
from verl_omni.workers.utils.losses import diffusion_loss

UPDATES, FAKE_UPDATE_RATIO, RESUME_POINTS = 12, 5, (3, 5, 6)
TEXT_LENGTHS = (3, 2, 2, 3)  # the global batch: two rows per rank, one per microbatch
STRAIGHT_SEED, OTHER_SEED = 0, 12345


def stage_of(update: int) -> str:
    return "student" if update % (FAKE_UPDATE_RATIO + 1) == FAKE_UPDATE_RATIO else "fake_score"


def update_batch(update: int):
    """This rank's rows of the update's global batch; a student update gets the prompts of the update before."""
    critic_update = update - 1 if stage_of(update) == "student" else update
    rows = micro_batch(TEXT_LENGTHS, seed=critic_update)
    rows["global_sample"] = torch.arange(len(TEXT_LENGTHS))
    per_rank = len(TEXT_LENGTHS) // dist.get_world_size()
    data = rows[dist.get_rank() * per_rank : (dist.get_rank() + 1) * per_rank]
    tu.assign_non_tensor(data, dmd_stage=stage_of(update), dmd_update=update, use_dynamic_bsz=False)
    return data


LOSS = partial(
    diffusion_loss,
    config=DiffusionActorConfig(strategy="fsdp2", rollout_n=1, diffusion_loss=DiffusionLossConfig(loss_mode="dmd2")),
)


def build_engine(mesh):
    module = tiny_h3(ADAPTERS)
    module.train()
    for block in (*module.token_refiner.refiner_blocks, *module.transformer_blocks):
        fully_shard(block, mesh=mesh)
    fully_shard(module, mesh=mesh)
    config = dmd_config(
        modality_shifts={"video": 12.0, "audio": 3.0},
        modality_weights={"video": 1.0, "audio": 1.0},
        critic_time_lead_modality="video",
        fake_update_ratio=FAKE_UPDATE_RATIO,
        sampler="renoise",
        critic_noise="fresh",
        score_query_noise="fresh",
    )
    return make_dmd_engine(module, config, architecture=ARCHITECTURE)


def _local(value):
    """``value`` with every DTensor replaced by a clone of its local shard."""
    if isinstance(value, DTensor):
        return value.to_local().detach().clone()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _local(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_local(item) for item in value)
    return value


def adapters_of(engine) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in engine.module.named_parameters() if ".lora_" in name}


def optimizers_of(engine) -> dict[str, dict]:
    return {name: copy.deepcopy(state.optimizer.state_dict()) for name, state in engine._adapter_optimizers.items()}


def reseed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def rng_states():
    return torch.get_rng_state(), np.random.get_state(), random.getstate()


def same_rng_states(before, after) -> bool:
    torch_before, numpy_before, python_before = before
    torch_after, numpy_after, python_after = after
    return (
        torch.equal(torch_before, torch_after)
        and all(
            np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
            for left, right in zip(numpy_before, numpy_after, strict=True)
        )
        and python_before == python_after
    )


def run_update(engine, update: int) -> dict:
    output = engine.train_batch(update_batch(update), LOSS)
    return {"loss": list(output["loss"]), "exit_index": output["metrics"].get("dmd/exit_index")}


def straight(mesh, out: Path) -> dict:
    reseed(STRAIGHT_SEED)
    engine = build_engine(mesh)
    before_each = []  # (adapters, optimizers) before update u
    records, saved = [], {}
    for update in range(UPDATES):
        before_each.append((adapters_of(engine), optimizers_of(engine)))
        records.append(run_update(engine, update) | {"adapters_after": _local(adapters_of(engine))})
        if update + 1 in RESUME_POINTS:
            engine.save_checkpoint(str(out / f"checkpoint-{update + 1}"), None, update + 1)
            saved[update + 1] = {"adapters": _local(adapters_of(engine)), "optimizers": _local(optimizers_of(engine))}
    evidence = {
        "records": records,
        "final_adapters": _local(adapters_of(engine)),
        "final_optimizers": _local(optimizers_of(engine)),
        "saved": saved,
        "seed": engine._program.seed,
        "isolated": [],
    }
    for update in range(UPDATES):
        fresh = build_engine(mesh)
        adapters, optimizers = before_each[update]
        with torch.no_grad():
            for name, param in fresh.module.named_parameters():
                if name in adapters:
                    param.copy_(adapters[name])
        for name, state in fresh._adapter_optimizers.items():
            state.optimizer.load_state_dict(copy.deepcopy(optimizers[name]))
        reseed(OTHER_SEED + update)
        evidence["isolated"].append(run_update(fresh, update) | {"adapters_after": _local(adapters_of(fresh))})
    return evidence


def resumed(mesh, out: Path, point: int) -> dict:
    reseed(STRAIGHT_SEED)
    engine = build_engine(mesh)
    reseed(OTHER_SEED)
    before = rng_states()
    engine.load_checkpoint(str(out / f"checkpoint-{point}"), None, del_local_after_load=False)
    load_kept_rng = same_rng_states(before, rng_states())
    loaded = {"adapters": _local(adapters_of(engine)), "optimizers": _local(optimizers_of(engine))}
    records = [
        run_update(engine, update) | {"adapters_after": _local(adapters_of(engine))} for update in range(point, UPDATES)
    ]
    return {
        "records": records,
        "final_adapters": _local(adapters_of(engine)),
        "final_optimizers": _local(optimizers_of(engine)),
        "load_kept_rng": load_kept_rng,
        "loaded": loaded,
    }


def main() -> None:
    """Run one mode and save each rank's evidence as ``<out>/<mode>-rank<r>.pt``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("mode", choices=["straight", "resumed"])
    parser.add_argument("--point", type=int)
    args = parser.parse_args()
    pin_engine_to_cpu()
    dist.init_process_group("gloo")
    try:
        mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("fsdp",))
        with tiny_row_widths():
            if args.mode == "straight":
                evidence, name = straight(mesh, args.out), "straight"
            else:
                evidence, name = resumed(mesh, args.out, args.point), f"resumed-{args.point}"
        torch.save(evidence, args.out / f"{name}-rank{dist.get_rank()}.pt")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
