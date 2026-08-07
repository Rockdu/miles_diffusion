"""SD3.5-medium OCR GRPO through the sglang-diffusion /rollout/generate path.

2-GPU colocate: FSDP DP=2 plus 2 rollout engines time-multiplexed on the same GPUs.

SD3.5 is a gated model, so HF_TOKEN must be set even when the weights are cached:
sglang still fetches model_index.json from the hub at startup.

Usage:
    python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py
    MILES_SCRIPT_DEBUG_ALIGNMENT=1 python3 scripts/run_diffusion_grpo_sd3_ocr_sglang.py
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "stabilityai/stable-diffusion-3.5-medium"
DATASET = "rockdu/miles-diffusion-datasets"
WANDB_PROJECT = "miles-diffusion-grpo"

# master_sglang carries native SD3 /rollout/generate support; prepending it to
# PYTHONPATH shadows the editable install at /sgl-workspace/sglang.
MASTER_SGLANG_PYTHON = "/sgl-workspace/master_sglang/sglang/python"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    cuda_visible_devices: str = "6,7"
    num_rollout: int = 600
    data_dir: str = "/root/datasets"
    debug_alignment: bool = False
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.hf_download_dataset(DATASET, include="flowgrpo_ocr/**", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    data_dir = f"{args.data_dir}/miles-diffusion-datasets/flowgrpo_ocr"
    run_name = f"diffusion_grpo_sd3_ocr_sglang_{U.create_run_id()}"

    ckpt_args = f"--hf-checkpoint {MODEL} " f"--save {args.output_dir}/{run_name}/ckpt "

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 16 "
        f"--num-rollout {args.num_rollout} "
        "--global-batch-size 64 "
        "--diffusion-microgroup-size 8 "
        "--micro-batch-size-sample 16 "
        "--micro-batch-size-tstep 5 "
    )

    diffusion_args = (
        f"--diffusion-model {MODEL} "
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 4.5 "
        "--diffusion-noise-level 0.7 "
        "--diffusion-height 512 "
        "--diffusion-width 512 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window "
        "--diffusion-num-sde-steps 10 "
        "--diffusion-sde-window-range 0,10 "
    )

    eval_args = "--diffusion-eval-num-steps 40 "

    grpo_args = (
        "--advantage-estimator grpo "
        "--globalize-reward-std "
        "--diffusion-clip-range 1e-4 "
        "--diffusion-kl-beta 0.04 "
    )

    optimizer_args = "--lr 3e-4 " "--adam-beta2 0.999 " "--weight-decay 1e-4 "

    lora_args = (
        "--use-lora "
        "--lora-ipc-weight-sync "
        "--lora-rank 32 "
        "--lora-alpha 64 "
        "--diffusion-init-lora-weight gaussian "
    )

    reward_args = "--rm-type ocr "

    wandb_args = U.get_default_wandb_args(WANDB_PROJECT, run_name)

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 8 "
        "--sglang-dit-precision fp16 "
        "--sglang-vae-slicing "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = "--train-backend fsdp " "--diffusion-forward-dtype fp16 "

    perf_args = "--gradient-checkpointing " "--deterministic-mode "

    misc_args = (
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 2 "
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 2 "
        "--colocate "
    )

    debug_args = "--diffusion-debug-mode --debug-skip-optimizer-step " if args.debug_alignment else ""

    U.execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{diffusion_args} "
            f"{eval_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{lora_args} "
            f"{reward_args} "
            f"{wandb_args} "
            f"{sglang_args} "
            f"{train_backend_args} "
            f"{perf_args} "
            f"{misc_args} "
            f"{debug_args} "
            f"{args.extra_args} "
        ),
        config=args,
        run_name=run_name,
        extra_env_vars={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": os.pathsep.join(
                [MASTER_SGLANG_PYTHON, *([p] if (p := os.environ.get("PYTHONPATH")) else [])]
            ),
            **({"MILES_VERIFY_WEIGHT_SYNC": "1"} if args.debug_alignment else {}),
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
