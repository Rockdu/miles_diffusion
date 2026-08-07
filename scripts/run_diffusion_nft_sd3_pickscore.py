"""SD3.5-medium DiffusionNFT training with PickScore.

Batch shape follows the UniRL 100-rollout override: 8 prompts x 8 samples, micro=4,
on 2 train GPUs plus a dedicated reward GPU.

NFT needs a reference model, supplied here by the EMA shadow (--ref-mode ema), and
samples under pi_old via --ema-rollout-policy ema. noise_level=0 with sde_type=ode
means the rollout is deterministic, which NFT requires.

Smoke mode swaps in the tiny OCR dataset and a 2-GPU no-dedicated-reward layout, for
checking the pipeline end to end without a real run.

Usage:
    python3 scripts/run_diffusion_nft_sd3_pickscore.py
    MILES_SCRIPT_SMOKE=1 python3 scripts/run_diffusion_nft_sd3_pickscore.py
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "stabilityai/stable-diffusion-3.5-medium"
DATASET = "rockdu/miles-diffusion-datasets"
WANDB_PROJECT = "miles-diffusion-nft"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    cuda_visible_devices: str = "4,5,2"
    num_rollout: int = 0  # 0 = pick the smoke/full default
    data_dir: str = "/root/datasets"
    smoke: bool = False
    extra_args: str = ""


def prepare(args: ScriptArgs):
    subset = "flowgrpo_ocr" if args.smoke else "flowgrpo_pickscore"
    U.hf_download_dataset(DATASET, include=f"{subset}/**", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    subset = "flowgrpo_ocr" if args.smoke else "flowgrpo_pickscore"
    data_dir = f"{args.data_dir}/miles-diffusion-datasets/{subset}"
    run_name = f"diffusion_nft_sd3_pickscore_{U.create_run_id()}"
    num_rollout = args.num_rollout or (1 if args.smoke else 100)

    ckpt_args = (
        f"--hf-checkpoint {MODEL} " f"--save {args.output_dir}/{run_name}/ckpt " "--save-interval 20 "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        f"--num-rollout {num_rollout} "
        "--num-steps-per-rollout 1 "
    ) + (
        (
            "--rollout-batch-size 2 "
            "--n-samples-per-prompt 2 "
            "--micro-batch-size 2 "
            "--diffusion-microgroup-size 2 "
        )
        if args.smoke
        else (
            "--rollout-batch-size 8 "
            "--n-samples-per-prompt 8 "
            "--micro-batch-size 4 "
            "--diffusion-microgroup-size 8 "
        )
    )

    diffusion_args = (
        f"--diffusion-model {MODEL} "
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-noise-level 0.0 "
        "--diffusion-sde-type ode "
        "--diffusion-height 512 "
        "--diffusion-width 512 "
    )

    eval_args = "--diffusion-eval-num-steps 50 --skip-eval-before-train " + (
        "" if args.smoke else f"--eval-prompt-data pickscore_test {data_dir}/test.jsonl --eval-interval 30 "
    )

    nft_args = (
        "--loss-type nft "
        "--diffusion-nft-beta 1.0 "
        "--diffusion-nft-adv-clip-max 5.0 "
        "--diffusion-nft-timestep-fraction 0.99 "
        "--advantage-estimator grpo "
        "--globalize-reward-std "
    )

    ema_args = (
        "--ref-mode ema "
        "--ema-shadow "
        "--ema-rollout-policy ema "
        "--ema-decay 0.001 "
        "--ema-uprate 0.001 "
        "--ema-uphold 0.5 "
        "--ema-flat-steps 0 "
    )

    optimizer_args = "--lr 3e-4 " "--adam-beta2 0.999 " "--weight-decay 1e-4 " "--clip-grad 1.0 "

    lora_args = (
        "--use-lora "
        "--lora-ipc-weight-sync "
        "--lora-rank 32 "
        "--lora-alpha 64 "
        "--diffusion-init-lora-weight gaussian "
    )

    reward_args = (
        "--rm-type ocr "
        if args.smoke
        else (
            "--rm-type pickscore "
            "--pickscore-num-workers 1 "
            "--pickscore-num-gpus-per-worker 1.0 "
            "--pickscore-batch-size 8 "
            "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
            "--pickscore-model-path yuvalkirstain/PickScore_v1 "
        )
    )

    wandb_args = U.get_default_wandb_args(WANDB_PROJECT, run_name)

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 8 "
        "--sglang-dit-precision fp16 "
        "--sglang-vae-slicing "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = "--train-backend fsdp " "--diffusion-forward-dtype fp16 "

    perf_args = "--gradient-checkpointing "

    misc_args = (
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 2 "
        "--rollout-num-gpus-per-engine 1 "
        f"--num-gpus-per-node {2 if args.smoke else 3} "
        "--colocate "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{diffusion_args} "
            f"{eval_args} "
            f"{nft_args} "
            f"{ema_args} "
            f"{optimizer_args} "
            f"{lora_args} "
            f"{reward_args} "
            f"{wandb_args} "
            f"{sglang_args} "
            f"{train_backend_args} "
            f"{perf_args} "
            f"{misc_args} "
            f"{args.extra_args} "
        ),
        config=args,
        run_name=run_name,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
