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

- **Flow-GRPO, DiffusionNFT, and SFT under one trainer.** Objectives are
  plugins selected by `--loss-type` (`policy_loss`, `nft`, `sft_loss`); the
  rollout / reward / weight-sync loop stays identical. See
  [Core Concepts](/user-guide/concepts).
- **sglang-diffusion rollout with trajectory capture.** The rollout engine
  returns per-step latents and SDE log-probs
  ([sglang#21204](https://github.com/sgl-project/sglang/pull/21204),
  [sglang#23151](https://github.com/sgl-project/sglang/pull/23151)), supports
  per-request scheduler switching for RL
  ([sglang#30036](https://github.com/sgl-project/sglang/pull/30036)), and
  sleep/wake for colocation
  ([sglang#22659](https://github.com/sgl-project/sglang/pull/22659)).
- **Single-prompt multi-generation.** One request produces N outputs for one
  prompt — conditioning is encoded once and expanded engine-side
  ([sglang#31233](https://github.com/sgl-project/sglang/pull/31233)). See
  [Single-Prompt Multi-Generation](/advanced/single-prompt-multi-gen).
- **LoRA training with CUDA-IPC weight sync.** Only `lora_A`/`lora_B` pairs
  cross the process boundary; the rollout engine merges locally
  ([sglang#31029](https://github.com/sgl-project/sglang/pull/31029)). See
  [LoRA Training and Weight Sync](/advanced/lora).
- **Streaming rollout transfer and async rewards.** msgpack raw-bytes
  transport, a Ray parser-actor pool, and per-microgroup reward scoring keep
  deserialization and reward off the rollout critical path
  ([#44](https://github.com/radixark/miles_diffusion/pull/44)). See
  [Streaming Reward and Deserialization](/advanced/streaming-reward).
- **USP sequence parallelism (Ulysses × Ring).** Shard long video sequences
  across GPUs via each family's diffusers `_cp_plan`
  ([#21](https://github.com/radixark/miles_diffusion/pull/21)).
- **Deterministic training.** Batch-invariant attention and seeded rollout for
  bitwise reproducible runs
  ([#14](https://github.com/radixark/miles_diffusion/pull/14)). See
  [Deterministic Training](/advanced/deterministic).
- **Per-family `TrainPipelineConfig`.** New model families plug in timestep
  scaling, CFG combine, conditioning collation, and LoRA targets without
  touching the trainer — see the model pages below.

## Supported models

Each model name links to its recipe page.

| Model | Task | Canonical recipes | Enabling PR |
|---|---|---|---|
| [Stable Diffusion 3.5](/models/sd3/sd3) | T2I | Flow-GRPO + OCR, DiffusionNFT + PickScore | [#4](https://github.com/radixark/miles_diffusion/pull/4) |
| [Qwen-Image](/models/qwen-image/qwen-image) | T2I | Flow-GRPO + PickScore (flow_grpo-aligned) | initial release; attention backend fix in [#48](https://github.com/radixark/miles_diffusion/pull/48) |
| [Wan2.2-T2V-A14B](/models/wan/wan2-2) | T2V | Flow-GRPO + PickScore, LoRA SFT | [#8](https://github.com/radixark/miles_diffusion/pull/8) |
| [LTX-2.3](/models/ltx/ltx2) | T2V (AV) | Flow-GRPO + PickScore | [#38](https://github.com/radixark/miles_diffusion/pull/38) |
| [Cosmos3-Nano](/models/cosmos/cosmos3) | T2I / T2V | Flow-GRPO + PickScore / VideoAlign | [#25](https://github.com/radixark/miles_diffusion/pull/25) *(in review)* |

## Feature support matrix

Objectives are model-agnostic plugins; ✓ marks combinations exercised by a
canonical recipe on `main` (see `scripts/`).

| | SD3.5 | Qwen-Image | Wan2.2 | LTX-2.3 | Cosmos3* |
|---|---|---|---|---|---|
| Flow-GRPO (`policy_loss`) | ✓ | ✓ | ✓ | ✓ | ✓ |
| DiffusionNFT (`nft`) | ✓ | — | — | — | — |
| SFT (`sft_loss`, `--train-only`) | — | — | ✓ | — | — |
| LoRA + IPC weight sync | ✓ | ✓ | ✓ | train-side merge | ✓ |
| Single-prompt multi-gen (microgroup > 1) | ✓ | ✓ | ✓ | — | — (packed forward) |
| USP sequence parallelism | via `_cp_plan` | via `_cp_plan` | ✓ | via `_cp_plan` | — |
| Deterministic mode | ✓ | ✓ | — | — | — |

*\*Cosmos3 support is pending merge of
[#25](https://github.com/radixark/miles_diffusion/pull/25).*

## Start here

1. **[Installation](/getting-started/installation)** — Docker image, pinned
   dependency versions, bare-metal setup.
2. **[Quick Start](/getting-started/quick-start)** — a working Flow-GRPO run
   on SD3.5 with 2 GPUs.
3. **[Core Concepts](/user-guide/concepts)** — the five objects in every
   miles-diffusion job and the loop that connects them.
4. **[Training Script Walkthrough](/user-guide/training-script-walkthrough)** —
   every argument group in a launch script, annotated.
5. **[CLI Reference](/user-guide/cli-reference)** — every flag, grouped and
   fully cataloged.

## Contribute

- GitHub: [github.com/radixark/miles_diffusion](https://github.com/radixark/miles_diffusion)
- Contributing: [developer guide](/developer/contributor-guide)
- Miles (LLM RL): [github.com/radixark/miles](https://github.com/radixark/miles)
