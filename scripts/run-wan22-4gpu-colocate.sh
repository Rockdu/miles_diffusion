#!/usr/bin/env bash
# Wan2.2-T2V-A14B GRPO on 4 GPUs, reward colocated.
#
# Derived from the 16-GPU run `wan22_16gpu_bs24_ns1`
# (wandb kangrdu/miles-diffusion-grpo/9rqt4j58), which was shaped by 40GB cards.
# Same trajectory scale -- 24 prompts x 8 samples = 192 items per rollout, one
# optimizer step per rollout -- packed onto 4 H200s with the reward model
# sharing the training GPUs instead of owning one.
#
# What the 40GB build had to give up, and what 140GB buys back:
#   gradient checkpointing  ON  -> OFF   (recompute was a memory tax; the DiT
#                                         activations fit, and Wan2.2 under
#                                         FSDP2 mixed precision is the config
#                                         the 5-GPU script warns aborts in
#                                         torch.utils.checkpoint)
#   update-weight buffer   512M -> 2G    (fewer IPC bucket round-trips)
#   4 GPUs per engine       ->  1        (sequence parallelism only pays when a
#                                         single card cannot hold the model)
#   micro-batch-size         1  -> 2     (matched to the rollout microgroup, see
#                                         below)
#
# Batch matching is deliberate, not cosmetic. --diffusion-microgroup-size is the
# rollout's forward batch (generate_microgroup sends one request per group) and
# --micro-batch-size is the trainer's. When they differ, cuBLAS picks a
# different split-k for Wan's skinny 5120->64 proj_out -- the one op measured
# NOT to be batch-invariant -- and the training forward stops reproducing the
# rollout bitwise. Equal values keep the policy that generated the trajectory
# byte-identical to the policy being differentiated.
#
# Resolution is 832x480, Wan's own 480P, rather than the square 480x480.
#
# Layout: all 4 GPUs run train + sgld colocate; --colocate-reward puts pickscore
# in the same placement group (train 0.7 + rollout 0.25 + reward 0.05).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
RUN_NAME="wan22_4gpu_colocate_bs24_ns1_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${ROOT_DIR}/logs/${RUN_NAME}/ckpt"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project miles-diffusion-grpo
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
    --diffusion-log-images 8
    --diffusion-log-image-interval 10
    --disable-wandb-random-suffix
  )
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASETS_DIR="/root/datasets/miles-diffusion-datasets"
if [[ ! -f "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" ]]; then
  hf download --repo-type dataset rockdu/miles-diffusion-datasets \
    --include "flowgrpo_pickscore/**" \
    --local-dir "${DATASETS_DIR}"
fi

WAN_LORA_TARGET_MODULES=(
  attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0
  attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0
  ffn.net.0.proj ffn.net.2
)

"${PYTHON_BIN}" -u "${ROOT_DIR}/train_diffusion.py" \
  --train-backend fsdp \
  --rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout \
  --hf-checkpoint Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --diffusion-model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt-data "${DATASETS_DIR}/flowgrpo_pickscore/train.jsonl" \
  --input-key input \
  --rollout-batch-size 24 \
  --n-samples-per-prompt 8 \
  --num-rollout 10000 \
  --num-steps-per-rollout 1 \
  --sglang-attention-backend torch_sdpa \
  --diffusion-microgroup-size 2 \
  --micro-batch-size-sample 2 \
  --micro-batch-size-tstep 1 \
  --diffusion-train-iter-order sample_major \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 1 \
  --num-gpus-per-node 4 \
  --colocate \
  --gradient-checkpointing \
  --use-lora \
  --lora-ipc-weight-sync \
  --lora-rank 64 \
  --lora-alpha 128 \
  --lora-target-modules "${WAN_LORA_TARGET_MODULES[@]}" \
  --diffusion-init-lora-weight gaussian \
  --lr 1e-4 \
  --adam-beta2 0.999 \
  --diffusion-clip-range 1e-4 \
  --weight-decay 1e-4 \
  --clip-grad 1.0 \
  --use-miles-router \
  --sglang-server-concurrency 8 \
  --rollout-health-check-timeout 120 \
  --miles-router-health-check-failure-threshold 20 \
  --update-weight-buffer-size 2147483648 \
  --update-weight-target-module transformer,transformer_2 \
  --diffusion-reward pickscore:1.0 \
  --advantage-estimator grpo \
  --rm-type pickscore \
  --pickscore-num-workers 1 \
  --colocate-reward \
  --pickscore-batch-size 8 \
  --pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --pickscore-model-path yuvalkirstain/PickScore_v1 \
  --fsdp-master-dtype fp32 \
  --fsdp-reduce-dtype fp32 \
  --diffusion-forward-dtype bf16 \
  --diffusion-num-steps 10 \
  --diffusion-eval-num-steps 28 \
  --diffusion-output-num-frames 21 \
  --diffusion-guidance-scale 4.0 \
  --diffusion-guidance-scale-2 3.0 \
  --diffusion-noise-level 0.9 \
  --diffusion-height 480 \
  --diffusion-width 832 \
  --diffusion-flow-shift 3.0 \
  --diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice \
  --diffusion-num-sde-steps 1 \
  --diffusion-sde-candidate-steps 1,2,3 \
  --diffusion-recompute-old-log-prob \
  --use-miles-dashboard \
  --miles-dashboard-workspace /root/miles_dashboard \
  --diffusion-debug-mode \
  --save "${SAVE_DIR}" \
  --save-interval 10 \
  --eval-prompt-data pickscore_test "${DATASETS_DIR}/flowgrpo_pickscore/test.jsonl" \
  --eval-interval 30 \
  --skip-eval-before-train \
  "${WANDB_ARGS[@]}"
