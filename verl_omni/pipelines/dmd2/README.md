# DMD2 computation core

`DMD2Program` implements continuous-flow distribution matching using the registered
`DMDLoss` and public `verl_omni.workers.utils.losses.diffusion_loss` wrapper.
The objective excludes adversarial loss; model construction and training lifecycle
remain caller-owned.

The caller supplies a velocity callback:

```python
def predict(role, latents, sigmas, negative):
    # Return flow velocities with the same modality keys and shapes as latents.
    ...
```

`role` is `student`, `fake_score`, or `teacher`; `negative=True` selects unconditional
teacher inputs for CFG. Latents are floating `(B, ...)` tensors sharing a batch size
and device. The callback owns models, conditions, precision, and forward contexts.

Teacher `cfg_norm="layer_norm"` with guidance other than 1 requires packed
`(B, tokens, features)` latents and velocities. It rescales last-axis vectors by
their L2 norms, not `torch.nn.LayerNorm`. The program checks rank, while the caller
must ensure the feature axis is last. CFG on unpacked latents requires `"none"` or
`"scalar"` normalization. Fake-score updates and inference have no CFG layout restriction.

A minimal configuration for packed image latents:

```python
from verl_omni.pipelines.dmd2.program import DMD2Program
from verl_omni.workers.config.diffusion.dmd import DiffusionDMDConfig

config = DiffusionDMDConfig(
    score_discrete_steps=0,
    modality_shifts={"image": 3.0},
    modality_weights={"image": 1.0},
)
program = DMD2Program(config, predict, seed=42)
```

The shared YAML defaults to `score_discrete_steps=1000`, so callers must override it
to `0` and supply both modality dictionaries. Hydra needs `+` for new dictionary
keys: `+dmd.modality_shifts.image=3.0` and `+dmd.modality_weights.image=1.0`.

| Fields | Meaning in this program |
| --- | --- |
| `num_inference_steps` | Positive integer number of generation steps. |
| `modality_shifts` | Per-modality rational shifts applied to the base-time knots. |
| `modality_weights` | Per-modality loss weights; keys match the shifts. |
| `base_time_knots` | `num_inference_steps + 1` knots from 1 to 0 before the shifts; `null` = uniform. |
| `sampler` | `euler` (along the velocity) or `renoise` (to x0, then fresh noise at the next knot). |
| `score_query_noise` | `student_estimate`, `trajectory` (initial noise, Euler only) or `fresh`. |
| `critic_noise` | `fresh` or `trajectory` (initial noise, Euler only). |
| `critic_time` | `uniform` with the lead-modality rule, or `query_matched`. |
| `critic_time_lead_modality/threshold`, `critic_time_follower_floor` | Lead-modality rule; the lead is required with several modalities. |
| `score_discrete_steps` | Must be `0`; discrete score sampling is unsupported here. |
| `rollout_timestep_shift`, `score_timestep_shift`, `score_sigma_min/max` | Upstream fields, unused here; keep their defaults. |

Changing an unused field raises an error.

`rollout(initial)` runs the student for every step without autograd and returns a
`Trajectory`: all `num_inference_steps + 1` FP32 states and each modality's grid, from
fresh standard Gaussian `initial` latents. The grid shifts the knots in FP64, then casts
to FP32. `sample(initial)` returns the rollout endpoint.

`stage_outputs(trajectory, stage, exit_index)` builds the loss inputs from a stored
rollout, so a student update can reuse the trajectory of the fake-score update before it.

- `fake_score`: re-noise the endpoint `states[-1]` at per-sample critic sigmas with
  `critic_noise`; the target is `noise - endpoint`. `uniform` draws each sigma from
  U(0, 1), and when the lead modality's sigma exceeds the threshold, every other modality
  redraws from U(floor, 1). `query_matched` draws a step `k` and ratio `r` per sample and
  maps them onto every modality's grid.
- `student`: re-run the student with autograd at `states[exit_index]` and take its x0.
  The query sigma is `grid[k + 1] + r * (grid[k] - grid[k + 1])` with one `r` per sample
  shared by all modalities. The fake score and teacher see the same detached query and
  sigmas. `draw_exit_index()` draws `k` once per update for every microbatch and rank.

`predict` receives per-sample FP32 `(B,)` sigmas. CFG applies only to the teacher.

`compute_loss(trajectory, stage, data, loss_function, exit_index=..., dp_group=...)` returns weighted
loss, per-modality metrics, and a nonfinite flag. Bind the public loss wrapper with
`loss_mode="dmd2"`; supply positive integer `gradient_accumulation_steps` and
`sp_size` in the `TensorDict`. The wrapper applies these scales once. Use `sp_size=1`
when gathering whole-sample outputs already scales backward.

`training_stages()` returns `fake_update_ratio` fake-score attempts followed by one
student attempt. The caller executes backward and optimizer updates, owns clipping,
schedulers, EMA and checkpoints, and handles reduction and skipped updates across ranks.
For resume, each rank saves these RNG states at the same training boundary:

| State | Owner and use |
| --- | --- |
| `program.state_dict()` | Private CPU generator for exit indices only. |
| Global CPU RNG | `torch.get_rng_state()` / `torch.set_rng_state()`; score ratios and critic times, plus any CPU tensor noise. |
| Used accelerator RNGs | Device backend's `get_rng_state` / `set_rng_state` (or `*_all`); initial noise, renoise sampling, fresh query and critic noise, and model randomness. |

`seed` and `load_state_dict()` affect only exit indices. Restore caller-owned CPU,
device, data-loader and other RNG states after model loading, before sampling the
next initial latents. Coordinate streams within model-parallel groups and use
independent streams across data-parallel replicas. Critic times are drawn in `modality_shifts` order and
noise in the order of the latent dict, so a resumed run must keep both orders. RNG restoration does not ensure
deterministic backend kernels or reductions.

With the repository's normal dependencies installed, run the CPU tests with:

```bash
pytest -q tests/pipelines/test_dmd2_program_on_cpu.py \
  tests/trainer/diffusion/test_dmd_config_on_cpu.py \
  tests/trainer/diffusion/test_dmd_loss_on_cpu.py \
  tests/trainer/diffusion/test_distillation_utils_on_cpu.py
```

CPU CI collects these files via the `_on_cpu.py` suffix. They test the computation
core without production models, engines, or distributed execution.
