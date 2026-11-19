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
"""Two-rank CPU FSDP2 run of the DMD engine on a tiny MiniMax H3; rank 0 writes the evidence as JSON.

Checked on the H3 LoRA targets, sharded per block as the engine shards:
  init       the critic and the student reproduce the teacher (adapters off) bit for bit while lora_B is 0
  isolation  a stage's backward and step change only its adapter
  recompute  with activation checkpointing, a stage's gradients equal those without it, although the stage runs
             forwards under the other adapter and under the teacher between its forward and its backward
  nonfinite  a NaN gradient on rank 1 only gives every rank the same nonfinite norm, and no rank steps
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from dmd_engine_fixtures import ADAPTERS, make_dmd_engine
from peft import LoraConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from tests.model_merger.model_fixtures import forward_inputs, tiny_transformer
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import H3_LORA_TARGETS
from verl_omni.workers.engine.lora_adapter_mixin import adapter_parameters

ARCHITECTURE = "MiniMaxH3Pipeline"


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().clone()


def _outputs(module, inputs) -> tuple[torch.Tensor, ...]:
    return tuple(value for value in module(**inputs) if isinstance(value, torch.Tensor))


def _loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack([value.float().square().mean() for value in outputs]).sum()


def _build_model(mesh):
    model = tiny_transformer(ARCHITECTURE)
    lora = LoraConfig(r=4, lora_alpha=4, target_modules=sorted(H3_LORA_TARGETS))
    for role in ADAPTERS:
        model.add_adapter(lora, adapter_name=role)
    model.set_adapter("default")
    model.set_attention_backend("native")
    for block in (*model.token_refiner.refiner_blocks, *model.transformer_blocks):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def _check_init(engine, inputs) -> dict[str, float]:
    with torch.no_grad():
        with engine.use_adapter("reference"):
            teacher = _outputs(engine.module, inputs)
        gaps = {}
        for role in ADAPTERS:
            with engine.use_adapter(role):
                outputs = _outputs(engine.module, inputs)
            gaps[role] = max((left - right).abs().max().item() for left, right in zip(outputs, teacher, strict=True))
    return gaps


def _randomize_lora_b(module) -> None:
    generator = torch.Generator().manual_seed(1000 + dist.get_rank())
    with torch.no_grad():
        for name, param in module.named_parameters():
            if ".lora_B." in name:
                local = param.to_local() if isinstance(param, DTensor) else param
                local.copy_(torch.randn(local.shape, generator=generator, dtype=local.dtype) * 0.1)


def _stage_gradients(engine, inputs, trained: str, other: str) -> dict[str, torch.Tensor]:
    with engine.train_stage(trained):
        engine.optimizer_zero_grad()
        loss = _loss(_outputs(engine.module, inputs))
        with torch.no_grad():
            with engine.use_adapter(other):
                _outputs(engine.module, inputs)
            with engine.use_adapter("reference"):
                _outputs(engine.module, inputs)
        loss.backward()
        names = {id(param) for param in adapter_parameters(engine.module, trained)}
        gradients = {
            name: _full(param.grad)
            for name, param in engine.module.named_parameters()
            if id(param) in names and param.grad is not None
        }
        engine.optimizer_zero_grad()
    return gradients


def _check_recompute(engine, inputs) -> dict[str, dict]:
    calls = []
    hook = engine.module.transformer_blocks[0].register_forward_pre_hook(lambda module, args: calls.append(1))
    result = {}
    try:
        for trained, other in (ADAPTERS, ADAPTERS[::-1]):
            engine.module.disable_gradient_checkpointing()
            calls.clear()
            plain = _stage_gradients(engine, inputs, trained, other)
            plain_calls = len(calls)
            engine.module.enable_gradient_checkpointing()
            calls.clear()
            checkpointed = _stage_gradients(engine, inputs, trained, other)
            result[trained] = {
                "plain_block_calls": plain_calls,
                "checkpointed_block_calls": len(calls),
                "gradient_count": len(plain),
                "same_names": sorted(plain) == sorted(checkpointed),
                "max_gradient_gap": max(
                    ((plain[name] - checkpointed[name]).abs().max().item() for name in plain if name in checkpointed),
                    default=float("inf"),
                ),
                "max_gradient_norm": max((value.norm().item() for value in plain.values()), default=0.0),
            }
    finally:
        hook.remove()
        engine.module.disable_gradient_checkpointing()
    return result


def _check_isolation(engine, inputs) -> dict[str, dict]:
    result = {}
    for trained, other in (ADAPTERS, ADAPTERS[::-1]):
        before = {name: _full(param) for name, param in engine.module.named_parameters()}
        trained_ids = {id(param) for param in adapter_parameters(engine.module, trained)}
        with engine.train_stage(trained):
            engine.optimizer_zero_grad()
            loss = _loss(_outputs(engine.module, inputs))
            with torch.no_grad(), engine.use_adapter(other):
                _outputs(engine.module, inputs)
            loss.backward()
            stray_gradients = [
                name
                for name, param in engine.module.named_parameters()
                if id(param) not in trained_ids and param.grad is not None
            ]
            grad_norm = engine.optimizer_step()
            engine.optimizer_zero_grad()
        changed_trained = changed_other = 0
        for name, param in engine.module.named_parameters():
            changed = not torch.equal(_full(param), before[name])
            if id(param) in trained_ids:
                changed_trained += changed
            else:
                changed_other += changed
        result[trained] = {
            "grad_norm": grad_norm,
            "stray_gradients": stray_gradients,
            "trained_parameters": len(trained_ids),
            "changed_trained": changed_trained,
            "changed_other": changed_other,
        }
    return result


def _poison_one_gradient(module, trained: str) -> str:
    """Set one element of a local gradient shard of ``trained`` to NaN, and return the parameter's name."""
    names = {id(param): name for name, param in module.named_parameters()}
    for param in adapter_parameters(module, trained):
        assert param.grad is not None, f"adapter {trained!r} parameter {names[id(param)]} has no gradient"
        local = param.grad.to_local()
        if local.numel() > 0:
            local.view(-1)[0] = float("nan")
            return names[id(param)]
    raise AssertionError(f"rank {dist.get_rank()} holds no gradient element of adapter {trained!r}")


def _check_nonfinite(engine, inputs) -> dict[str, dict]:
    result = {}
    for trained in ADAPTERS:
        before = {name: _full(param) for name, param in engine.module.named_parameters()}
        with engine.train_stage(trained):
            engine.optimizer_zero_grad()
            _loss(_outputs(engine.module, inputs)).backward()
            poisoned = _poison_one_gradient(engine.module, trained) if dist.get_rank() == 1 else None
            grad_norm = engine.optimizer_step()
            engine.optimizer_zero_grad()
        grad_norms: list[float | None] = [None] * dist.get_world_size()
        dist.all_gather_object(grad_norms, grad_norm)
        poisoned_names: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(poisoned_names, poisoned)
        result[trained] = {
            "grad_norms": [str(value) for value in grad_norms],
            "poisoned": poisoned_names,
            "changed": sum(
                not torch.equal(_full(param), before[name]) for name, param in engine.module.named_parameters()
            ),
        }
    return result


def main() -> None:
    """Run the four checks in one process group and write rank 0's evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    try:
        mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("fsdp",))
        model = _build_model(mesh)
        model.train()
        engine = make_dmd_engine(model)
        inputs = forward_inputs(ARCHITECTURE)
        evidence: dict[str, object] = {"init_gap": _check_init(engine, inputs)}
        _randomize_lora_b(model)
        evidence["recompute"] = _check_recompute(engine, inputs)
        evidence["isolation"] = _check_isolation(engine, inputs)
        evidence["nonfinite"] = _check_nonfinite(engine, inputs)
        evidence["active_after"] = engine.module.active_adapters()
        if dist.get_rank() == 0:
            args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
