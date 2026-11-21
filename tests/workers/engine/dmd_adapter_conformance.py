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
"""The contract every ``(architecture, dmd2)`` adapter keeps with the DMD engine, as checks over one adapter case.

An adapter supplies ``dmd_condition(micro_batch)``, ``dmd_initial_noise(micro_batch)`` and
``dmd_predict(module, latents, sigmas, condition, negative)``, where ``sigmas`` maps each modality to a per-sample
``(B,)`` tensor.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable

import torch
from tensordict import TensorDict


@dataclass
class AdapterCase:
    """One adapter with a module and a microbatch of at least two samples it can predict on.

    ``context`` opens around every adapter call, for settings the case's module needs.
    """

    adapter: type
    module: torch.nn.Module
    micro_batch: TensorDict
    context: Callable[[], AbstractContextManager]


def _sigmas(names, batch: int) -> dict[str, torch.Tensor]:
    """Distinct sigmas per sample and per modality, inside (0, 1)."""
    return {
        name: torch.linspace(0.2, 0.8, batch) * (0.9 - 0.1 * index) + 0.05 * index for index, name in enumerate(names)
    }


def _predict(case: AdapterCase, micro_batch: TensorDict, latents, sigmas) -> dict[str, torch.Tensor]:
    condition = case.adapter.dmd_condition(micro_batch)
    with torch.no_grad():
        return case.adapter.dmd_predict(case.module, latents, sigmas, condition, False)


def check_initial_noise(case: AdapterCase) -> None:
    """``dmd_initial_noise`` returns one tensor of batch size B per modality."""
    batch = len(case.micro_batch)
    with case.context():
        noise = case.adapter.dmd_initial_noise(case.micro_batch)
    assert noise
    for name, value in noise.items():
        assert isinstance(value, torch.Tensor), name
        assert value.shape[0] == batch, name


def check_predict_keys_and_shapes(case: AdapterCase) -> None:
    """``dmd_predict`` returns the latents' keys and shapes."""
    with case.context():
        latents = case.adapter.dmd_initial_noise(case.micro_batch)
        prediction = _predict(case, case.micro_batch, latents, _sigmas(latents, len(case.micro_batch)))
    assert prediction.keys() == latents.keys()
    for name, value in latents.items():
        assert prediction[name].shape == value.shape, name


def check_predict_is_repeatable(case: AdapterCase) -> None:
    """The same ``dmd_predict`` call twice returns equal outputs."""
    with case.context():
        latents = case.adapter.dmd_initial_noise(case.micro_batch)
        sigmas = _sigmas(latents, len(case.micro_batch))
        first = _predict(case, case.micro_batch, latents, sigmas)
        second = _predict(case, case.micro_batch, latents, sigmas)
    for name in latents:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def check_predict_is_per_sample(case: AdapterCase) -> None:
    """In a microbatch whose samples carry different sigmas, each sample's output equals that sample's alone."""
    with case.context():
        latents = case.adapter.dmd_initial_noise(case.micro_batch)
        sigmas = _sigmas(latents, len(case.micro_batch))
        batched = _predict(case, case.micro_batch, latents, sigmas)
        for index in range(len(case.micro_batch)):
            alone = _predict(
                case,
                case.micro_batch[index : index + 1],
                {name: value[index : index + 1] for name, value in latents.items()},
                {name: value[index : index + 1] for name, value in sigmas.items()},
            )
            for name in latents:
                torch.testing.assert_close(batched[name][index : index + 1], alone[name])


CHECKS = (check_initial_noise, check_predict_keys_and_shapes, check_predict_is_repeatable, check_predict_is_per_sample)
