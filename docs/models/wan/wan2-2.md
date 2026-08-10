---
title: Wan2.2-T2V-A14B
description: Dual-expert MoE video model — Flow-GRPO + PickScore recipe, LoRA SFT recipe, and the high/low-noise expert boundary.
---
## 1. Model introduction

[Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) is a
text-to-video model with a **dual-expert MoE DiT**: a high-noise expert
(`transformer`) denoises timesteps `t ≥ boundary` and a low-noise expert
(`transformer_2`) handles the rest. Conditioning comes from a UMT5 text
encoder; latents go through the Wan VAE (4× temporal compression).

Training support landed in
[#8](https://github.com/radixark/miles_diffusion/pull/8); the rollout side is
enabled by sglang-diffusion's per-request scheduler switch
([sglang#30036](https://github.com/sgl-project/sglang/pull/30036)) and Wan
multi-output conditioning
([sglang#27223](https://github.com/sgl-project/sglang/pull/27223),
[sglang#31233](https://github.com/sgl-project/sglang/pull/31233)).

**Key highlights for RL training:**

- **Two experts, one boundary.** `boundary_ratio = 0.875` — which expert a
  train pair updates depends only on its timestep. Weight sync must push both:
  `--update-weight-target-module transformer,transformer_2`.
- **Two guidance scales.** Rollout denoises low-noise steps with
  `guidance_scale_2` and there is **no fallback** — training asserts
  `--diffusion-guidance-scale-2` is set explicitly, because a silent mismatch
  against rollout would corrupt the ratio.
- **Flow shift override.** The sgl-d serving default is `flow_shift = 12.0`;
  recipes override it to `3.0` via `--diffusion-flow-shift`, which repositions
  the expert boundary on the timestep grid.
- **USP-ready.** Wan was the first family enabled for Ulysses × Ring sequence
  parallelism ([#21](https://github.com/radixark/miles_diffusion/pull/21)).

## 2. Supported variants

| Model | HF ID | Notes |
|---|---|---|
| Wan2.2-T2V-A14B | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | Default in both Wan recipes |

Family detection matches checkpoint names against `("wan2.2", "wan-2.2")`
(`miles/backends/fsdp_utils/configs/wan2_2.py`). Override with
`MILES_DIFFUSION_MODEL_FAMILY=wan2_2`.

## 3. Family config

Registered in `miles/backends/fsdp_utils/configs/wan2_2.py`:

| Property | Value | Notes |
|---|---|---|
| Expert routing | `t ≥ 0.875 × num_train_timesteps` → `transformer`, else `transformer_2` | `component_for_timestep` |
| Guidance routing | high-noise → `--diffusion-guidance-scale`, low-noise → `--diffusion-guidance-scale-2` (required) | `select_guidance_scale` |
| Timestep scaling | None — Wan DiT takes raw scheduler timesteps (0..1000) | |
| Condition inputs | `encoder_hidden_states` only (fixed-length UMT5 embeds) | Concat-collate, no padding needed |
| CFG combine | `neg + scale × (pos − neg)` | Standard |
| CFG batching | Off | |
| fp32 param islands | `scale_shift_table`, `time_embedder`, `norm2` kept fp32 under FSDP mixed precision | `models/diffusers/wan2_2/parallel_plan.py` |

## 4. Launch

### 4.1 Flow-GRPO + PickScore (4 train GPUs + 1 reward GPU)

Canonical recipe: `scripts/run_diffusion_grpo_wan22_pickscore_5gpu.py`

```bash
python3 scripts/run_diffusion_grpo_wan22_pickscore_5gpu.py
```

The launcher downloads the `flowgrpo_pickscore` subset of
[`rockdu/miles-diffusion-datasets`](https://huggingface.co/datasets/rockdu/miles-diffusion-datasets)
and starts a colocated train+rollout job on GPUs 0–3 with PickScore on GPU 4.

### 4.2 LoRA SFT on (video, prompt) pairs (4 GPUs, no rollout engines)

Recipe: `scripts/run_diffusion_sft_wan22.py`

```bash
MILES_SCRIPT_DATA_JSONL=/abs/data.jsonl python3 scripts/run_diffusion_sft_wan22.py
```

Dataset rows: `{"prompt": "...", "metadata": {"video": "/abs/path.mp4"}}`.
`--loss-type sft_loss --train-only` runs no sglang engines; the
`sft_rollout` plugin encodes cache misses through a colocated encoder pool
(UMT5 + Wan VAE, `miles/rollout/encoder_hub/wan2_2.py`) and writes
content-addressed files into `.sft_cache/` — epoch 2 onward is all cache hits.
Frame counts must satisfy `4k + 1` (Wan VAE temporal stride).

## 5. Recipe configuration (GRPO + PickScore)

### 5.1 Batch sizing

| Flag | Value | Effect |
|---|---|---|
| `--rollout-batch-size` | 48 | Prompts per rollout |
| `--n-samples-per-prompt` | 16 | 768 samples per rollout |
| `--num-steps-per-rollout` | 2 | 384 samples per optimizer step over 4 train GPUs |
| `--diffusion-microgroup-size` | 8 | 8 outputs per rollout request ([multi-gen](/advanced/single-prompt-multi-gen)) |
| `--micro-batch-size` | 2 | Keeps every micro-batch **phase-pure**: one expert, one CFG scale. 4 OOMs on H200 |

### 5.2 SDE schedule and the expert boundary

```bash
--diffusion-num-steps 10
--diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice
--diffusion-num-sde-steps 1
--diffusion-sde-candidate-steps 1,2,3
--diffusion-noise-level 0.9
--diffusion-flow-shift 3.0
```

`epoch_global_random_choice` draws **one** SDE step per rollout, shared across
the batch, from candidate steps 1, 2, 3. At `flow_shift = 3.0` the dual-expert
boundary sits at `t = 875`, so steps 1–2 train `transformer` (high-noise) and
step 3 trains `transformer_2` (low-noise): both experts receive gradient
stochastically across rollouts, and weight sync pushes both every iteration.

<Warning>
Candidate step indices are schedule-dependent. If you change
`--diffusion-num-steps` or `--diffusion-flow-shift`, re-derive which steps land
on which side of the boundary before reusing `--diffusion-sde-candidate-steps`.
</Warning>

### 5.3 Guidance

```bash
--diffusion-guidance-scale 4.0     # high-noise expert
--diffusion-guidance-scale-2 3.0   # low-noise expert — required, no fallback
```

### 5.4 LoRA and weight sync

```bash
--use-lora --lora-ipc-weight-sync \
--lora-rank 64 --lora-alpha 128 --lora-init-weights gaussian \
--lora-target-modules attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 \
  attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0 \
  ffn.net.0.proj ffn.net.2 \
--update-weight-buffer-size 2147483648 \
--update-weight-target-module transformer,transformer_2
```

Targets cover self-attention (`attn1`), cross-attention (`attn2`), and the
FFN. IPC merge internals: [LoRA Training and Weight Sync](/advanced/lora).

### 5.5 Precision and known limits

```bash
--train-backend fsdp --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 \
--diffusion-forward-dtype bf16
```

Gradient checkpointing stays **off**: Wan2.2 under FSDP2 mixed precision hits
a `torch.utils.checkpoint` `CheckpointError` on the fp32 RoPE frequency
buffers. If you OOM, lower `--rollout-batch-size`,
`--n-samples-per-prompt`, or `--diffusion-microgroup-size` instead.

## 6. USP sequence parallelism

Wan is the reference family for USP (Ulysses × Ring), added in
[#21](https://github.com/radixark/miles_diffusion/pull/21). The plan is built
from the model's diffusers `_cp_plan` boundaries at wrap time
(`miles/backends/fsdp_utils/model_backend.py`):

```bash
--sequence-parallel-size 4      # sp ranks per replica
--ulysses-degree 0              # 0 = auto: ulysses fills sp, ring = sp // ulysses
```

Ring degrees > 1 use torch's experimental ring-attention implementation and
require torch ≥ 2.11.

## 7. Pairs well with

- [Single-Prompt Multi-Generation](/advanced/single-prompt-multi-gen) — the
  microgroup mechanics behind `--diffusion-microgroup-size 8`.
- [LoRA Training and Weight Sync](/advanced/lora) — IPC merge used by the GRPO
  recipe.
- [SDE Step Backend](/advanced/sde-backend) — how the trained SDE step is
  scored train-side.
- [Rewards](/user-guide/rewards) — PickScore worker pool configuration.
