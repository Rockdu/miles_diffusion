"""MiniMax H3 t2va PickScore DiffusionNFT: 2-GPU FSDP train + sglang rollout, reward colocated.

Same topology and H3 settings as the Flow-GRPO 2-GPU recipe; the training signal is what
differs. NFT learns from the clean x0 at resampled sigmas, so the rollout runs as ODE with
no noise injection and no SDE window, and the engine returns x0 alone instead of a trajectory.

H3's packed forward takes one sample at a time, so micro-batch-size-sample stays 1 and the
K (x0, t) pairs NFT expands per sample are trained one by one.
"""

from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "MiniMaxAI/MiniMax-H3"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
WANDB_PROJECT = "miles-diffusion-nft"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 30
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 8
    num_steps_per_rollout: int = 2
    eval_interval: int = 10
    # miles has no eval-sample cap (verl used val_max_samples=64), so a fixed slice of
    # the test split stands in: all 2048 prompts would take >60h at 20 steps here.
    eval_size: int = 16
    save_interval: int = 10
    data_dir: str = "/root/datasets"
    extra_args: str = ""


def prepare(args: ScriptArgs) -> str:
    local_dir = U.hf_download_dataset(DATASET, include=f"{DATASET_SUBSET}/**", data_dir=args.data_dir)
    return f"{local_dir}/{DATASET_SUBSET}"


def _eval_slice(prompt_dir: str, eval_size: int) -> str:
    eval_data = Path(prompt_dir) / f"val_{eval_size}.jsonl"
    if not eval_data.exists():
        with open(Path(prompt_dir) / "test.jsonl") as f:
            lines = [next(f) for _ in range(eval_size)]
        eval_data.write_text("".join(lines))
    return str(eval_data)


def execute(args: ScriptArgs, prompt_dir: str) -> None:
    run_name = f"diffusion_nft_h3_t2va_{U.create_run_id()}"

    ckpt_args = (
        f"--hf-checkpoint {MODEL} "
        f"--save {args.output_dir}/{run_name}/ckpt "
        f"--save-interval {args.save_interval} "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {prompt_dir}/train.jsonl "
        "--input-key input "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--num-steps-per-rollout {args.num_steps_per_rollout} "
        f"--num-rollout {args.num_rollout} "
        "--rollout-microgroup-size 1 "
        "--micro-batch-size-sample 1 "
        "--micro-batch-size-tstep 1 "
        "--diffusion-train-iter-order sample_major "
    )

    diffusion_args = (
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-noise-level 0.0 "
        "--diffusion-sde-type ode "
        "--diffusion-h3-aspect-ratio 16:9 "
        "--diffusion-h3-duration-seconds 4 "
        "--diffusion-audio-flow-shift 3.0 "
        "--diffusion-flow-shift 12.0 "
    )

    eval_args = ""
    if args.eval_interval:
        eval_args = (
            f"--eval-interval {args.eval_interval} "
            f"--eval-prompt-data pickscore_val {_eval_slice(prompt_dir, args.eval_size)} "
            "--n-samples-per-eval-prompt 1 "
            "--diffusion-eval-num-steps 20 "
        )

    nft_args = (
        "--loss-type nft "
        "--diffusion-nft-beta 1.0 "
        "--diffusion-nft-timestep-fraction 0.99 "
        "--advantage-estimator grpo --globalize-reward-std "
    )

    optimizer_args = "--lr 1e-4 --weight-decay 1e-4 --adam-eps 1e-15 "

    # H3's rollout DiT renames modules and fuses Q/K/V, so weights only reach the engine
    # through the LoRA IPC path's layer grouper; the family rejects any other sync mode.
    lora_args = "--use-lora --lora-ipc-weight-sync --lora-rank 64 --lora-alpha 128 "

    reward_args = (
        "--rm-type pickscore "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
        "--pickscore-num-frames 8 "
        "--pickscore-num-workers 1 "
        "--pickscore-num-gpus-per-worker 0 "
        "--pickscore-batch-size 4 "
        "--rollout-parser-num-workers 2 "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=4)

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 2 "
        "--sglang-tp-size 2 "
        "--sglang-sp-degree 1 "
        "--sglang-ulysses-degree 1 "
        "--sglang-ring-degree 1 "
        "--sglang-dit-precision bf16 "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--fsdp-master-dtype bf16 "
        "--fsdp-reduce-dtype bf16 "
        "--diffusion-forward-dtype bf16 "
    )

    perf_args = "--gradient-checkpointing "

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 2 "
        "--rollout-num-gpus-per-engine 2 "
        "--num-gpus-per-node 2 "
        "--colocate "
        "--colocate-reward "
        "--deterministic-mode "
        "--rollout-health-check-interval 60 "
        "--miles-router-health-check-failure-threshold 30 "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {diffusion_args} {eval_args} {nft_args} {optimizer_args} "
            f"{lora_args} {reward_args} {wandb_args} {sglang_args} {train_backend_args} {perf_args} "
            f"{misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=2,
        config=args,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    prompt_dir = prepare(args)
    execute(args, prompt_dir)


if __name__ == "__main__":
    typer.run(main)
