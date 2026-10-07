"""Krea-2 OCR DiffusionNFT: sync vs one-step async under different EMA settings.

Both modes share the async recipe's batch (6 prompts x 8 samples) and NFT/optimizer/LoRA
settings, on one 8-GPU node. sync colocates train+rollout on all 8 GPUs; async uses 6 train +
2 rollout GPUs as in scripts/run_diffusion_nft_krea2_async.py.

Usage:
    python3 scripts/exp_nft_krea2_ema.py --mode async --tag A0
    python3 scripts/exp_nft_krea2_ema.py --mode async --tag A1 --extra-args "--nft-async-old-ema current"
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
    mode: str = "async"  # sync | async
    tag: str = "run"
    num_rollout: int = 100
    data_dir: str = "/root/datasets"
    wandb_project: str = "async-nft-ema"
    ema_decay_init: float = 0.001
    ema_decay_ramp: float = 0.001
    ema_decay_max: float = 0.5
    ema_decay_flat_steps: int = 0
    extra_args: str = ""


def execute(args: ScriptArgs, data_dir: str) -> None:
    is_async = args.mode == "async"
    run_name = f"krea2_ocr_{args.tag}_{args.mode}_{U.create_run_id()}"

    common_args = (
        f"--hf-checkpoint {MODEL} --save {args.output_dir}/{run_name}/ckpt --save-interval 50 "
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl --input-key input "
        f"--num-rollout {args.num_rollout} --num-steps-per-rollout 1 "
        "--diffusion-num-steps 10 --diffusion-guidance-scale 1.0 "
        "--diffusion-noise-level 0.0 --diffusion-sde-type ode "
        "--diffusion-height 1024 --diffusion-width 1024 --diffusion-debug-mode "
        "--rollout-microgroup-size 1 --rollout-batch-size 6 --n-samples-per-prompt 8 "
        "--diffusion-eval-num-steps 52 --skip-eval-before-train "
        f"--eval-prompt-data ocr_test {data_dir}/test.jsonl --eval-interval 50 "
        "--loss-type nft --diffusion-nft-beta 1.0 --diffusion-nft-timestep-fraction 0.99 "
        "--advantage-estimator grpo --globalize-reward-std "
        "--use-ema --rollout-weights ema "
        f"--ema-decay-init {args.ema_decay_init} --ema-decay-ramp {args.ema_decay_ramp} "
        f"--ema-decay-max {args.ema_decay_max} --ema-decay-flat-steps {args.ema_decay_flat_steps} "
        "--lr 3e-4 --adam-beta2 0.999 --weight-decay 1e-4 --clip-grad 1.0 "
        "--use-lora --lora-ipc-weight-sync --lora-rank 32 --lora-alpha 64 --lora-init-weights gaussian "
        "--rm-type ocr "
        "--use-miles-router --sglang-server-concurrency 8 --sglang-dit-precision bf16 --sglang-vae-slicing "
        "--update-weight-buffer-size 2147483648 "
        "--train-backend fsdp --diffusion-forward-dtype bf16 --deterministic-mode "
        "--rollout-num-gpus-per-engine 1 --num-gpus-per-node 8 "
    )
    if is_async:
        layout_args = "--actor-num-gpus-per-node 6 --rollout-num-gpus 2 --micro-batch-size 4 "
    else:
        layout_args = (
            "--actor-num-gpus-per-node 8 --rollout-num-gpus 8 --colocate "
            "--micro-batch-size 2 --gradient-checkpointing "
        )

    wandb_args = U.get_default_wandb_args(
        __file__, run_id=run_name, project=args.wandb_project, wandb_log_num_images=8, wandb_log_image_interval=10
    )

    U.execute_train(
        train_args=f"{common_args} {layout_args} {wandb_args} {args.extra_args}",
        num_gpus_per_node=8,
        train_script="train_diffusion_async.py" if is_async else "train_diffusion.py",
        config=args,
        extra_env_vars={
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            **({} if is_async else {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    assert args.mode in ("sync", "async"), args.mode
    local_dir = U.hf_download_dataset(DATASET, include=f"{SUBSET}/**", data_dir=args.data_dir)
    execute(args, f"{local_dir}/{SUBSET}")


if __name__ == "__main__":
    typer.run(main)
