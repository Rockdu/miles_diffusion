"""Krea-2 Flow-GRPO train/rollout alignment probe: no optimizer step, compare rollout precisions.

With --debug-skip-optimizer-step the trainer and the rollout hold the same weights, so
log_prob_mean_abs_diff, ratio_abs_minus_1 and clipfrac measure only the numeric gap between the
bf16 trainer and the rollout engine. Run once with the default bf16 rollout and once with
--extra-args "--sglang-quantization fp8".
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "krea/Krea-2-Raw"
DATASET = "rockdu/miles-diffusion-datasets"
SUBSET = "flowgrpo_ocr"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    tag: str = "align"
    num_rollout: int = 2
    data_dir: str = "/root/datasets"
    wandb_project: str = "async-nft-ema"
    extra_args: str = ""


def execute(args: ScriptArgs, data_dir: str) -> None:
    run_name = f"krea2_grpo_align_{args.tag}_{U.create_run_id()}"
    train_args = (
        f"--hf-checkpoint {MODEL} "
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl --input-key input "
        f"--num-rollout {args.num_rollout} --rollout-batch-size 2 --n-samples-per-prompt 8 --global-batch-size 16 "
        "--rollout-microgroup-size 1 "
        "--diffusion-num-steps 10 --diffusion-guidance-scale 1.0 --diffusion-noise-level 0.7 "
        "--diffusion-height 1024 --diffusion-width 1024 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window "
        "--diffusion-num-sde-steps 10 --diffusion-sde-window-range 0,10 --rollout-return-full-trajectory "
        "--advantage-estimator grpo --globalize-reward-std --diffusion-clip-range 1e-4 --diffusion-kl-beta 0.0 "
        "--lr 3e-4 --adam-beta2 0.999 --weight-decay 1e-4 "
        "--use-lora --lora-ipc-weight-sync --lora-rank 32 --lora-alpha 64 --lora-init-weights gaussian "
        "--rm-type ocr "
        "--use-miles-router --sglang-server-concurrency 8 --sglang-dit-precision bf16 --sglang-vae-slicing "
        "--update-weight-buffer-size 2147483648 "
        "--train-backend fsdp --diffusion-forward-dtype bf16 "
        "--gradient-checkpointing --micro-batch-size-sample 2 --micro-batch-size-tstep 2 "
        "--actor-num-gpus-per-node 8 --rollout-num-gpus 8 --rollout-num-gpus-per-engine 1 --num-gpus-per-node 8 "
        "--colocate --deterministic-mode --diffusion-debug-mode --debug-skip-optimizer-step "
    )
    wandb_args = U.get_default_wandb_args(__file__, run_id=run_name, project=args.wandb_project)
    U.execute_train(
        train_args=f"{train_args} {wandb_args} {args.extra_args}",
        num_gpus_per_node=8,
        config=args,
        extra_env_vars={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    local_dir = U.hf_download_dataset(DATASET, include=f"{SUBSET}/**", data_dir=args.data_dir)
    execute(args, f"{local_dir}/{SUBSET}")


if __name__ == "__main__":
    typer.run(main)
