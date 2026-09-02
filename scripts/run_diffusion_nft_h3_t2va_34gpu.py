"""MiniMax H3 t2va PickScore DiffusionNFT: 4 nodes x 8 GPU train+rollout + 2 reward GPUs.

Same cluster, topology, and H3 sampling settings as run_diffusion_grpo_h3_t2va_34gpu.py --
read that file for the multi-node launch contract, the hostfile format, and why tp_size is
2, why microgroups stay at 1, and why gradient checkpointing is required. Only the training
signal differs:

  * NFT learns from the clean x0 at resampled sigmas, so the rollout is ODE with no noise
    injection and no SDE window, and the engine returns x0 alone instead of a trajectory.
  * The reference model is the EMA copy, which also serves as the rollout policy.
  * One optimizer step per rollout: NFT has no importance-ratio clipping to keep a second
    inner step on-policy.

The NFT knobs follow UniRL's own MiniMax-H3 recipe (beta, lr, grad clip, EMA ramp, and the
full sigma grid), not the SD3 one. Resolution and batch stay this repo's.

Usage (on the head node):

    MASTER_ADDR=<head_ip> python3 scripts/run_diffusion_nft_h3_t2va_34gpu.py
"""

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "MiniMaxAI/MiniMax-H3"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
WANDB_PROJECT = "miles-diffusion-nft"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    hostfile: str = "/root/h3_hostfile"
    num_rollout: int = 10000
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 16
    num_steps_per_rollout: int = 1
    eval_interval: int = 10
    eval_size: int = 16
    save_interval: int = 10
    data_dir: str = "/root/datasets"
    # Multi-node DCP: every rank writes its own shard, so this must be a
    # filesystem both nodes share, or the checkpoint is split and unloadable.
    output_dir: str = "/cluster-storage/shared/miles_diffusion/logs"
    dashboard_workspace: str = "/root/miles_dashboard"
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
    master_addr = os.environ.get("MASTER_ADDR")
    external_ray = U.get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY")
    if not external_ray:
        assert master_addr, (
            "one-command launch builds the whole cluster from this node: run with "
            "MASTER_ADDR=<this node ip> and a hostfile of 'ip [num_gpus]' lines "
            "(see --hostfile), or bring your own cluster with MILES_SCRIPT_EXTERNAL_RAY=1."
        )
    run_name = f"diffusion_nft_h3_t2va_34gpu_{U.create_run_id()}"

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

    # beta and the full sigma grid are H3's own, not SD3's: UniRL runs beta 1.0 on SD3 and
    # Qwen-Image but 0.1 here, and keeps every transition. At fraction 0.99 the one sigma
    # dropped is the smallest, which on H3's shift-12 grid is the only mid-range point.
    nft_args = (
        "--loss-type nft "
        "--diffusion-nft-beta 0.1 "
        "--diffusion-nft-timestep-fraction 1.0 "
        "--advantage-estimator grpo --globalize-reward-std "
    )

    # The reference model NFT needs, and the pi_old the rollout samples under. Decay 0 for
    # the first 75 refreshes makes the shadow a hard copy, then it ramps.
    ema_args = (
        "--ref-mode ema "
        "--use-ema "
        "--ema-rollout-policy ema "
        "--ema-decay-init 0.0 "
        "--ema-decay-ramp 0.0075 "
        "--ema-decay-max 0.999 "
        "--ema-decay-flat-steps 75 "
    )

    optimizer_args = "--lr 3e-4 --weight-decay 1e-4 --clip-grad 1.0 "

    # H3's rollout DiT renames modules and fuses Q/K/V, so weights only reach the engine
    # through the LoRA IPC path's layer grouper; the family rejects any other sync mode.
    lora_args = "--use-lora --lora-ipc-weight-sync --lora-rank 64 --lora-alpha 128 "

    reward_args = (
        "--rm-type pickscore "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
        "--pickscore-num-frames 8 "
        "--pickscore-num-workers 8 "
        "--pickscore-num-gpus-per-worker 0.25 "
        "--pickscore-batch-size 8 "
        "--rollout-parser-num-workers 32 "
        "--rollout-fetch-in-parser "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=run_name, project=WANDB_PROJECT, wandb_log_num_images=8)

    dashboard_args = f"--use-miles-dashboard --miles-dashboard-workspace {args.dashboard_workspace} "

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 2 "
        "--sglang-tp-size 2 "
        "--sglang-sp-degree 1 "
        "--sglang-ulysses-degree 1 "
        "--sglang-ring-degree 1 "
        "--sglang-dit-precision bf16 "
        "--update-weight-buffer-size 4294967296 "
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--fsdp-master-dtype bf16 "
        "--fsdp-reduce-dtype bf16 "
        "--diffusion-forward-dtype bf16 "
        "--dp-replicate-size 4 "
    )

    perf_args = "--gradient-checkpointing "

    misc_args = (
        "--actor-num-nodes 4 "
        "--actor-num-gpus-per-node 8 "
        "--rollout-num-gpus 32 "
        "--rollout-num-gpus-per-engine 2 "
        "--num-gpus-per-node 8 "
        "--colocate "
        "--wandb-log-image-interval 10 "
        "--rollout-health-check-interval 60 "
        "--miles-router-health-check-failure-threshold 30 "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {diffusion_args} {eval_args} {nft_args} {ema_args} {optimizer_args} "
            f"{lora_args} {reward_args} {wandb_args} {dashboard_args} {sglang_args} {train_backend_args} "
            f"{perf_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=8,
        config=args,
        before_ray_job_submit=(
            None if external_ray else partial(U.ssh_start_ray_workers, master_addr, 8, args.hostfile)
        ),
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    prompt_dir = prepare(args)
    execute(args, prompt_dir)


if __name__ == "__main__":
    typer.run(main)
