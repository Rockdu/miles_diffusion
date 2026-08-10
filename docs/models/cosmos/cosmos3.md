---
title: Cosmos3-Nano
description: 16B MoT (8B UND + 8B GEN) omni model — token-level conditioning, packed single-sample forward, VideoAlign reward.
---
<Warning>
Cosmos3 support is **pending merge**:
[#25](https://github.com/radixark/miles_diffusion/pull/25) (train pipeline
config, VideoAlign reward, T2V recipe) plus a companion sglang-diffusion
branch (`feat/cosmos3-rl-rollout`: rollout SDE-Euler on the serving sigma
grid, trajectory sigmas, fused-param weight-sync fix). Everything below
describes that PR, not `main`.
</Warning>

## 1. Model introduction

[Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) is a 16 B
Mixture-of-Transformers (MoT) omni model: an 8 B **UND** (understanding) tower
and an 8 B **GEN** (generation) tower over a joint text+vision packed
sequence. It reuses the Wan2.2 VAE (4× temporal compression).

**Key highlights for RL training:**

- **No separate text encoder.** Conditioning is token-level: `CondKwargs`
  carries `text_ids` / `text_mask` / `fps` verbatim, which eliminates the
  text-replay-consistency failure class other families guard against.
- **UND tower frozen inside the training graph.** The UND tower participates
  in the packed forward, so it is frozen by parameter-name fragments rather
  than dropped; LoRA targets are GEN attention only
  (`add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out`).
- **Packed single-sample forward.** The transformer consumes one packed
  text+vision sequence per forward — one request cannot batch multiple
  outputs, so recipes run `--diffusion-microgroup-size 1` and CFG batching is
  disabled by construction.
- **Karras flow-sigma grid.** The checkpoint ships a non-uniform sigma grid;
  SDE candidate steps must be derived from it (see §5.2).

## 2. Supported variants

| Model | Composition | HF ID |
|---|---|---|
| Cosmos3-Nano | 8 B UND + 8 B GEN (MoT) | [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) |

Family detection matches `("cosmos3", "cosmos-3")`. Cosmos3 requires
`--update-weight-target-module transformer` (validated at startup).

## 3. Family config

From `miles/backends/fsdp_utils/configs/cosmos3.py` (on the PR branch):

| Property | Value | Why |
|---|---|---|
| Timestep dtype | fp32, no scaling | The Karras grid is non-integer and sgl-d conditions on exact fp32 values — bf16 rounds 993.25 → 992 |
| Cond dtype | pass-through | mRoPE position ids sit at ~15000 where bf16 spacing is 128; a boundary cast scrambles rotary phases |
| CFG batching | Off (asserted) | Packed forward is single-sample |
| LoRA targets | GEN attention (`add_*_proj`, `to_add_out`) | UND tower and unused sound/action heads stay frozen |
| Frozen params | Name-fragment allowlist (`_GEN_PARAM_FRAGMENTS`) | UND sits inside the graph and cannot be detached |

## 4. Launch

Recipes on the PR branch:

| Recipe | Layout | Reward |
|---|---|---|
| `scripts/run-diffusion-grpo-cosmos3-pickscore-t2i-5gpu.sh` | 4 colocate + 1 reward GPU, T2I (832×480, 1 frame) | PickScore |
| `run-diffusion-grpo-cosmos3-videoalign-4gpu.sh` | 3 colocate + 1 reward GPU, T2V (17 frames, 832×480) | VideoAlign |

```bash
export SGLANG_DISABLE_COSMOS3_GUARDRAILS=1   # RL scores raw samples; skip serving-side guardrail models
bash scripts/run-diffusion-grpo-cosmos3-pickscore-t2i-5gpu.sh
```

## 5. Recipe configuration (T2I PickScore)

### 5.1 Batch and algorithm

| Setting | Value |
|---|---|
| Batch | 48 prompts × 16 samples, `num_steps_per_rollout=2` → 96 items/rank on 4 GPUs |
| Microgroup | `--diffusion-microgroup-size 1` (packed forward, see §1) |
| Guidance | `1.0` — CFG-free training; merged LoRA sampled at g=4 still beats base at g=4 |
| SDE | Flow-SDE, `--diffusion-noise-level 0.7`, 16 steps (eval 35) |
| KL | `--diffusion-kl-beta 1e-3`, global reward std, per-prompt mean |
| LoRA | r=64, alpha=128, init gaussian |
| Optimizer | lr 3e-4, `--adam-beta2 0.95`, weight decay 1e-4, `--clip-grad 2e-3` |
| Clipping | `--diffusion-clip-range 1e-3` |

### 5.2 SDE schedule on a Karras grid

`epoch_global_window` draws a 2-step window per rollout from
`--diffusion-sde-candidate-steps 4-15`.

<Warning>
The Cosmos3 checkpoint's Karras flow-sigma grid puts head steps 1–3 at
`sigma > 0.96` with `|dt| < 0.02` — they train nothing. Step numbers are
**not transferable across sigma-grid families**: re-derive candidates from
`|dt|` when changing model or grid.
</Warning>

### 5.3 Ratio stability

Two choices specific to this recipe:

- **`--diffusion-recompute-old-log-prob`** — the trainer recomputes old
  log-probs at rollout ingestion so the PPO ratio is
  implementation-self-consistent (rollout FA kernels vs train SDPA would
  otherwise leak into the ratio).
- **`--adam-beta2 0.95` + `--clip-grad 2e-3`** — absorb Adam-preconditioner
  spikes after quiet stretches (single-step policy jumps the PPO loss clip
  cannot stop).

## 6. VideoAlign reward (T2V)

The T2V recipe scores with **VideoAlign**
([KlingTeam/VideoReward](https://huggingface.co/KlingTeam/VideoReward)): the
z-scored sum of Visual Quality (VQ), Motion Quality (MQ), and Text Alignment
(TA). Because it needs transformers 4.45.x, the worker runs in a **pinned
interpreter** via Ray `runtime_env.py_executable`.

Per-dimension scores are logged on a rolling basis: **TA collapse is the
canonical reward-hacking mode** and is invisible in the summed Overall score.

## 7. Validation status (from the PR)

- T2I pipeline smoke (3 rollouts): `ratio_abs_minus_1` stable at 1–2.5e-5
  (10× below clip range); cross-engine weight-sync checksums equal.
- T2V e2e on wandb (`miles-diffusion-grpo/diffusion_grpo_cosmos3_videoalign_*`):
  768×17f rollout in ~15 min on 3 engines; long-run reward trend still being
  monitored.
- Batched multi-sample generation per request is deliberately deferred.

## 8. Pairs well with

- [SDE Step Backend](/advanced/sde-backend) — the SCORE-mode step Cosmos3
  feeds with fp32 timesteps.
- [LoRA Training and Weight Sync](/advanced/lora) — GEN-tower LoRA sync.
- [Rewards](/user-guide/rewards) — custom reward workers; VideoAlign follows
  the same actor-pool pattern as PickScore.
