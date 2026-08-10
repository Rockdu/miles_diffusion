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
`TrainPipelineConfig` isolates model quirks so new architectures plug in without
touching the trainer.
- **LoRA training with ipc-handle weight sync.** PEFT LoRA on the FSDP2 actor;
each iteration ships only `lora_A`/`lora_B` pairs to the rollout engines
over CUDA IPC and merges them engine-side — no full-weight transfer, no
separate merge or conversion step. See
[LoRA Training and Weight Sync](/advanced/lora).
- **Quality control on three fronts.** Deterministic mode (seeded,
collision-free rollout + batch-invariant train-side attention) makes runs
bitwise reproducible and backs the CI e2e regression suite; sglang-side
monkey patches manage train/rollout alignment; and an FSDP2 param-dtype
patch manages precision — per-parameter fp32 islands (e.g. Wan's
`scale_shift_table`) under the mixed-precision policy. See
[Deterministic Training](/advanced/deterministic) and
[Dtype Control](/advanced/dtype-control).
- **SFT, DiffusionNFT, and Flow-GRPO under one trainer — and easy to
extend.** Objectives are `--loss-type` plugins sharing the same
rollout / reward / weight-sync loop, and every stage of that loop is
swappable through customized-function flags: `--custom-loss-function-path`,
`--custom-prepare-train-batch-path`,
`--custom-convert-samples-to-train-data-path`, `--rollout-function-path`,
`--custom-rm-path`, `--diffusion-step-strategy-path`, and more. SFT itself
ships as exactly this set of plugins — supporting a new algorithm means
writing a few functions, not forking the trainer. See
[Core Concepts](/user-guide/concepts).
- **Sglang native.** Two commitments: rollout runs **on the inference engine
itself** — the sglang-diffusion serving stack, with RL support living
engine-side (trajectory capture with SDE/CPS log-probs, per-request
scheduler switching,
[single-prompt multi-generation](/advanced/single-prompt-multi-gen),
sleep/wake for colocation, CUDA-IPC weight updates, msgpack tensor
transport); and **train-inference consistency is managed on sglang** through
a curated set of monkey patches (RMSNorm, QK-norm + RoPE, LayerNorm
scale-shift, fused mul-add) that pin engine kernels to the numerics of the
training-side diffusers forward.
- **Multiple parallelism.** The rollout engines scale with tensor and
sequence parallelism (`--sglang-tp-size` × `--sglang-sp-degree` per engine,
plus CFG parallel); training scales with USP (Ulysses × Ring), built from
each family's diffusers `_cp_plan` — no model rewrite required.



## Supported models

Each model name links to its recipe page.


| Model                                                   | Task      | Canonical recipes                         |
| ------------------------------------------------------- | --------- | ----------------------------------------- |
| [Stable Diffusion 3.5](/models/sd3/sd3)                 | T2I       | Flow-GRPO + OCR, DiffusionNFT + PickScore |
| [Qwen-Image](/models/qwen-image/qwen-image)             | T2I       | Flow-GRPO + PickScore (flow_grpo-aligned) |
| [Wan2.2-T2V-A14B](/models/wan/wan2-2)                   | T2V       | Flow-GRPO + PickScore, LoRA SFT           |
| [LTX-2.3](/models/ltx/ltx2)                             | T2V       | Flow-GRPO + PickScore                     |
| [Cosmos3 (Edge / Nano / Super)](/models/cosmos/cosmos3) | T2I / T2V | Flow-GRPO + PickScore                     |




## Feature support matrix

Objectives are model-agnostic plugins; ✓ marks combinations exercised by a
canonical recipe in `scripts/`, except the USP row, which reflects what the
family's code path and tests enable (no shipped recipe turns it on yet).


|                                          | SD3.5          | Qwen-Image     | Wan2.2 | LTX-2.3          | Cosmos3            |
| ---------------------------------------- | -------------- | -------------- | ------ | ---------------- | ------------------ |
| Flow-GRPO (`policy_loss`)                | ✓              | ✓              | ✓      | ✓                | ✓                  |
| DiffusionNFT (`nft`)                     | ✓              | —              | —      | —                | —                  |
| SFT (`sft_loss`, `--train-only`)         | —              | —              | ✓      | —                | —                  |
| LoRA + IPC weight sync                   | ✓              | ✓              | ✓      | train-side merge | ✓                  |
| Single-prompt multi-gen (microgroup > 1) | ✓              | ✓              | ✓      | —                | — (packed forward) |
| USP sequence parallelism                 | via `_cp_plan` | via `_cp_plan` | ✓      | via `_cp_plan`   | —                  |
| Deterministic mode                       | ✓              | ✓              | —      | ✓                | —                  |




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

