#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=1,2
# Prepare OCR prompts into JSONL expected by Miles data loader.
python "${ROOT_DIR}/tools/prepare_ocr_jsonl.py"

# Minimal diffusion GRPO run, aligned with flow_grpo single-node settings.
python "${ROOT_DIR}/train.py" \
  --train-backend fsdp \
  --diffusion-train \
  --rollout-function-path miles.rollout.diffusion_rollout.generate_rollout \
  --hf-checkpoint gpt2 \
  --prompt-data "${ROOT_DIR}/data/ocr/train.jsonl" \
  --input-key input \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 16 \
  --num-rollout 1 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 4 \
  --colocate \
  --diffusion-model stabilityai/stable-diffusion-3.5-medium \
  --diffusion-num-steps 10 \
  --diffusion-guidance-scale 4.5 \
  --diffusion-noise-level 0.7 \
  --diffusion-height 512 \
  --diffusion-width 512 \
  --diffusion-reward pickscore \
  --sglang-disable-cuda-graph \
  --sglang-mem-fraction-static 0.7 \
  --sglang-cuda-graph-max-bs 16
