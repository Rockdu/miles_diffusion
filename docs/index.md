---
title: Miles-Diffusion Documentation
---
Miles-diffusion is a reinforcement learning (RL) post-training framework for
**diffusion models** — text-to-image and text-to-video. It couples
[sglang-diffusion](https://github.com/sgl-project/sglang) for high-throughput
rollout with **FSDP2 + diffusers** for training, and inherits the modular,
minimal-core design of [Miles](https://github.com/radixark/miles).

*"A journey of a thousand miles begins with a single rollout."* — For DiT
models the rollout is a full denoising trajectory, and miles-diffusion focuses
on the system work that makes trajectory-level RL stable, efficient, and
reproducible.

## Core features

- **Fast and stable support for the latest diffusion models.** Launch-ready
  recipes for Wan2.2-T2V-A14B (dual-expert MoE video), Qwen-Image, LTX-2.3,
  the Cosmos3 MoT omni family, and SD3.5. A per-family
  `TrainPipelineConfig` isolates model quirks — timestep scaling, CFG combine,
  conditioning collation, LoRA targets — so new architectures plug in without
  touching the trainer.
- **LoRA training with zero-copy weight sync.** PEFT LoRA on the FSDP2 actor;
  each iteration ships only `lora_A`/`lora_B` pairs to the rollout engines
  over CUDA IPC and merges them engine-side — no full-weight transfer, no
  separate merge or conversion step. See
  [LoRA Training and Weight Sync](/advanced/lora).
- **Deterministic mode.** Seeded, collision-free rollout combined with
  batch-invariant train-side attention gives bitwise reproducible runs —
  repeatable experiments you can audit and resume. See
  [Deterministic Training](/advanced/deterministic).
- **SFT, DiffusionNFT, and Flow-GRPO under one trainer.** Objectives are
  `--loss-type` plugins (`sft_loss`, `nft`, `policy_loss`) sharing the same
  rollout / reward / weight-sync loop, so switching algorithms never means
  switching systems. See [Core Concepts](/user-guide/concepts).
- **sglang native.** Rollout runs on the sglang-diffusion serving engine
  itself, not a forked inference stack. RL support lives engine-side —
  trajectory capture with SDE/CPS log-probs, per-request scheduler switching,
  [single-prompt multi-generation](/advanced/single-prompt-multi-gen),
  sleep/wake for colocation, CUDA-IPC weight updates, and msgpack tensor
  transport — while a small set of monkey patches (RMSNorm, QK-norm + RoPE,
  LayerNorm scale-shift, fused mul-add) keeps engine kernels numerically
  aligned with the training-side diffusers forward.

- **USP sequence parallelism (Ulysses × Ring).** Shard long video sequences
  across GPUs through each family's diffusers `_cp_plan` — no model rewrite
  required.

## Supported models

Each model name links to its recipe page.

| Model | Task | Canonical recipes |
|---|---|---|
| [Stable Diffusion 3.5](/models/sd3/sd3) | T2I | Flow-GRPO + OCR, DiffusionNFT + PickScore |
| [Qwen-Image](/models/qwen-image/qwen-image) | T2I | Flow-GRPO + PickScore (flow_grpo-aligned) |
| [Wan2.2-T2V-A14B](/models/wan/wan2-2) | T2V | Flow-GRPO + PickScore, LoRA SFT |
| [LTX-2.3](/models/ltx/ltx2) | T2V | Flow-GRPO + PickScore |
| [Cosmos3 (Edge / Nano / Super)](/models/cosmos/cosmos3) | T2I / T2V | Flow-GRPO + PickScore |

## Feature support matrix

Objectives are model-agnostic plugins; ✓ marks combinations exercised by a
canonical recipe in `scripts/`, except the USP row, which reflects what the
family's code path and tests enable (no shipped recipe turns it on yet).

| | SD3.5 | Qwen-Image | Wan2.2 | LTX-2.3 | Cosmos3 |
|---|---|---|---|---|---|
| Flow-GRPO (`policy_loss`) | ✓ | ✓ | ✓ | ✓ | ✓ |
| DiffusionNFT (`nft`) | ✓ | — | — | — | — |
| SFT (`sft_loss`, `--train-only`) | — | — | ✓ | — | — |
| LoRA + IPC weight sync | ✓ | ✓ | ✓ | train-side merge | ✓ |
| Single-prompt multi-gen (microgroup > 1) | ✓ | ✓ | ✓ | — | — (packed forward) |
| USP sequence parallelism | via `_cp_plan` | via `_cp_plan` | ✓ | via `_cp_plan` | — |
| Deterministic mode | ✓ | ✓ | — | ✓ | — |

## Start here

1. **[Installation](/getting-started/installation)** — Docker image, pinned
   dependency versions, bare-metal setup.
2. **[Quick Start](/getting-started/quick-start)** — a working Flow-GRPO run
   on SD3.5 with 2 GPUs.
3. **[Core Concepts](/user-guide/concepts)** — the five objects in every
   miles-diffusion job and the loop that connects them.
4. **[Training Script Walkthrough](/user-guide/training-script-walkthrough)** —
   every argument group in a launch script, annotated.
5. **[Rewards](/user-guide/rewards)** — built-in reward models and custom
   reward hooks.
6. **Model guides** — per-model config and recipes, starting from the
   [supported models](#supported-models) table above.

## Contribute

- GitHub: [github.com/radixark/miles_diffusion](https://github.com/radixark/miles_diffusion)
- Miles (LLM RL): [github.com/radixark/miles](https://github.com/radixark/miles)
