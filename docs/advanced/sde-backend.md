---
title: SDE Step Backend
description: Train-side SDE dynamics for Flow-GRPO — interface, rollout alignment, and related algorithms.
---
Flow-matching RL algorithms fall into two paradigms:

- **Coupled (Flow-GRPO)** — training re-scores the same `(x_t → x_{t+1})`
  transitions that rollout produced; needs tractable log-probabilities → uses
  `SdeStepBackend`.
- **Decoupled (DiffusionNFT, …)** — training samples its own timesteps from the
  final image; rollout dynamics are irrelevant → does **not** use this backend.

## 1. Overview

Flow-GRPO training (`--loss-type policy_loss`, the default) calls
`SdeStepBackend.sde_step_logprob` on each recorded transition. The backend
mirrors sglang-diffusion's rollout stepping so train-side log-probs match
rollout-side values.

| Objective | Miles flag | Uses `SdeStepBackend`? | Paper |
|---|---|---|---|
| Flow-GRPO | `--loss-type policy_loss` | **Yes** | [Flow-GRPO](https://arxiv.org/abs/2505.05470) |
| DiffusionNFT | `--loss-type nft` | **No** | [DiffusionNFT](https://arxiv.org/abs/2509.16117) |

Canonical Flow-GRPO recipe: `scripts/run_diffusion_grpo_sd3_ocr_sglang.py` — see
[SD3 model guide](/models/sd3/sd3) and [Quick Start](/getting-started/quick-start)
for full training setup.

## 2. SDE step computation

Each denoising step is a stochastic transition from latent `x_t` to `x_{t+1}`.
The backend computes three quantities from the DiT velocity prediction `v_θ`:

| Symbol | Meaning |
|---|---|
| `prev_mean` | Mean of the Gaussian transition `p(x_{t+1} \| x_t, v_θ)` |
| `noise_std` | Std of that Gaussian (drives `log_prob`) |
| `std_dev_t` | Diffusion scale factor (KL denominator in Flow-GRPO) |

### Per-step pipeline

Both rollout (SAMPLE mode) and training (SCORE mode) share the same kernel:

```
(timestep, next_timestep, x_t, v_θ, η)
    → resolve_sigmas        → (σ, σ_prev)
    → prev_sample_mean_and_std → (prev_mean, noise_std, std_dev_t)
```

**SAMPLE (rollout).** Draw `ε ~ N(0, I)` and set `x_{t+1} = prev_mean + noise_std · ε`.
Store `x_{t+1}` and the step's `log_prob`.

**SCORE (train).** `x_{t+1}` is already recorded as `next_latent` from rollout.
Re-run the DiT forward with updated weights, recompute `(prev_mean, noise_std)`,
then score the fixed residual:

```
log_prob = mean_dims( −‖x_{t+1} − prev_mean‖² / (2·noise_std²) − log(noise_std) − log√(2π) )
```

Training never resamples noise — it only evaluates how likely the rollout action
was under the current policy. `flow_grpo_loss_formula` compares this
`log_prob_new` against rollout's `log_prob_old`.

### Flow-SDE kernel (`DiffusersSdeStepBackend`)

Default for `--diffusion-sde-type sde`. Implements the Flow-GRPO noise schedule
σ_t ∝ η√(t/(1−t)) ([paper](https://arxiv.org/abs/2505.05470)).

**1. Resolve σ.** Look up `(σ, σ_prev)` from the diffusers scheduler using the
**actual rollout timestep values** (not positional indices). `dt = σ_prev − σ`
(is negative as denoising progresses).

**2. Diffusion scale.**

```
std_dev_t = √(σ / (1 − σ)) · η        (η = --diffusion-noise-level)
```

**3. Transition mean** (flow-matching SDE drift):

```
prev_mean = x_t · (1 + std_dev_t² / (2σ) · dt)
          + v_θ · (1 + std_dev_t² · (1 − σ) / (2σ)) · dt
```

**4. Gaussian std** of the transition:

```
noise_std = std_dev_t · √(−dt)
```

At rollout, `x_{t+1} ~ N(prev_mean, noise_std² I)`. At train time the same
formulas score the stored `next_latent`.

### CPS kernel (`CpsSdeStepBackend`)

For `--diffusion-sde-type cps` ([FlowCPS](https://arxiv.org/abs/2509.05952)).
σ is resolved directly as `timestep / divisor` — no scheduler table.

```
std_dev_t = σ_prev · sin(ηπ / 2)
pred_x0   = x_t − σ · v_θ
noise_est = x_t + v_θ · (1 − σ)
prev_mean = pred_x0 · (1 − σ_prev) + noise_est · √(σ_prev² − std_dev_t²)
noise_std = std_dev_t
```

CPS sets `noise_std == std_dev_t`. Both trainer and engine drop Gaussian
constants in `log_prob` (`diffusion_log_prob_no_const=True`).

### What each train pair carries

`expand_samples_to_train_pairs` (`miles/ray/data_conversion_hub/flow_grpo.py`)
produces one row per selected SDE step:

| Field | Role in SDE computation |
|---|---|
| `latent` | `x_t` input to the kernel |
| `next_latent` | Recorded `x_{t+1}` — fixed target for SCORE mode |
| `timestep`, `next_timestep` | Resolve `(σ, σ_prev)` |
| `log_prob_old` | Rollout log-prob at sampling time |

The FSDP actor forward-passes the DiT to get `v_θ`, then calls
`sde_step_logprob(..., prev_sample=next_latent)` to obtain `log_prob_new`.

## 3. Dynamics types

| SDE formulation | `--diffusion-sde-type` | Train backend | Noise schedule | Paper |
|---|---|---|---|---|
| Flow-SDE | `sde` | `DiffusersSdeStepBackend` | η√(t/(1−t)) via scheduler σ | [Flow-GRPO](https://arxiv.org/abs/2505.05470) |
| Dance-SDE | `sde` + tune `--diffusion-noise-level` | `DiffusersSdeStepBackend` | constant η (engine-side) | [DanceGRPO](https://arxiv.org/abs/2505.07818) |
| CPS | `cps` | `CpsSdeStepBackend` | σₜ₋₁ sin(ηπ/2) | [FlowCPS](https://arxiv.org/abs/2509.05952) |
| ODE | `ode` | `DiffusersSdeStepBackend` (no noise) | deterministic | — (rollout only for NFT) |

Auto-selection in `arguments.py` (unless `--sde-step-backend-path` is set):

```python
sde_step_backends = {
    "sde": "...DiffusersSdeStepBackend",
    "cps": "...CpsSdeStepBackend",
    "ode": "...DiffusersSdeStepBackend",
}
```

Custom backends: pass any import path via `--sde-step-backend-path`.

Partial SDE windows ([MixGRPO](https://arxiv.org/abs/2507.21802),
[TempFlow-GRPO](https://arxiv.org/abs/2508.04324)) select which steps enter
§2's computation — see §6.

## 4. Interface

Defined in `miles/backends/fsdp_utils/sde_step_backend.py`:

```python
class SdeStepBackend(abc.ABC):
    def resolve_sigmas(timesteps, next_timesteps, *, ndim) -> (sigma, sigma_prev)
    def prev_sample_mean_and_std(model_output, sample, sigma, sigma_prev, *, noise_level)
        -> (prev_mean, noise_std, std_dev_t)
    def log_prob(prev_sample, prev_mean, noise_std) -> Tensor
    def sde_step_logprob(...) -> (prev_sample, log_prob, prev_mean, std_dev_t)
```

| Method | Role |
|---|---|
| `resolve_sigmas` | Map rollout `(timestep, next_timestep)` → `(σ, σ_prev)` |
| `prev_sample_mean_and_std` | Dynamics kernel (§2) |
| `log_prob` | Gaussian score of `prev_sample − prev_mean` |
| `sde_step_logprob` | Full SCORE-mode composition used by `flow_grpo_loss_formula` |

Constructed in the FSDP actor and passed to `DiffusionLossContext.sde_backend`.

## 5. Rollout-side SDE parameters

These CLI flags configure sglang-diffusion rollout stepping and must match §2–§3
on the trainer:

| CLI flag | Default | Sent to engine |
|---|---|---|
| `--diffusion-sde-type` | `sde` | `rollout_sde_type` |
| `--diffusion-noise-level` | 0.7 | `rollout_noise_level` |
| `--diffusion-num-sde-steps` | 0 | Step strategy input |
| `--diffusion-sde-window-range` | None | e.g. `0,10` |
| `--diffusion-step-strategy-path` | None | Custom step selection |
| `--diffusion-num-steps` | model default | Denoise schedule length |

DiffusionNFT uses ODE rollout (`--diffusion-sde-type ode`,
`--diffusion-noise-level 0.0`) but does not consume SDE log-probs at train time.

## 6. Step strategies

Functions in `miles/rollout/step_strategy_hub.py` select which denoising steps
run through the SDE kernel during rollout and enter §2's train-pair expansion.
Signature: `(args, sample, num_steps, seed) -> (sde, ret)`.

### `sde_window`

Random contiguous window of `--diffusion-num-sde-steps` steps within
`--diffusion-sde-window-range`:

```bash
--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window \
--diffusion-num-sde-steps 10 \
--diffusion-sde-window-range 0,10
```

sglang-d returns the full trajectory; only the window steps are scored in training.

### `epoch_global_random_choice`

Per-epoch global random SDE step subset via `--diffusion-sde-candidate-steps`
(e.g. `"1,2,3"`).

## 7. Train / rollout alignment

The train-side kernel must reproduce rollout dynamics exactly. Mismatched σ
resolution or noise level causes `train/log_prob_mean_abs_diff` to diverge.

Checklist:

1. Match `--diffusion-sde-type` and `--diffusion-noise-level` on train and rollout.
2. Use the same `--diffusion-num-steps` schedule.
3. Match input dtypes between FSDP forward and the rollout engine when running
   fp32 end-to-end.
4. Monitor `train/log_prob_mean_abs_diff` — should stay near zero before the
   first optimizer step.

## 8. Future: unified sgl-d backend

The `SdeStepBackend` ABC is designed so a future version can import the same
dynamics kernel sglang-diffusion uses in SAMPLE mode (rollout) and SCORE mode
(trainer). Today the two paths are separate but structurally mirrored.
`--sde-step-backend-path` will be the integration point when sgl-d exports its
stepping kernel for import.

## 9. References

| Algorithm / component | miles entry point | Paper |
|---|---|---|
| Flow-GRPO | `--loss-type policy_loss` + SDE backend | [Flow-GRPO (2025)](https://arxiv.org/abs/2505.05470) |
| DanceGRPO dynamics | `--diffusion-sde-type sde`, tune `--diffusion-noise-level` | [DanceGRPO (2025)](https://arxiv.org/abs/2505.07818) |
| MixGRPO / partial SDE windows | `sde_window` step strategy | [MixGRPO (2025)](https://arxiv.org/abs/2507.21802), [TempFlow-GRPO (2025)](https://arxiv.org/abs/2508.04324) |
| FlowCPS | `--diffusion-sde-type cps` | [FlowCPS (2025)](https://arxiv.org/abs/2509.05952) |
| DiffusionNFT | `--loss-type nft` (decoupled; no SDE backend) | [DiffusionNFT (2025)](https://arxiv.org/abs/2509.16117) |
