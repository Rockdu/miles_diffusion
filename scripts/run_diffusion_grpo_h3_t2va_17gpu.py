"""MiniMax H3 t2va PickScore Flow-GRPO: 2 nodes x 8 GPU train+rollout + 1 reward GPU.

Scaled from run_diffusion_grpo_h3_t2va_2gpu.py; every sampling knob there is kept, so
read that file for why the SDE window starts at step 1, why short_edge is 1344x768 /
107 frames, and why the router health-check window is widened. Topology, batch, and
adam_eps differ: the 2-GPU recipe pins eps to verl's 1e-15, this one takes miles'
1e-8 default, as every other family recipe does.

Multi-node: submits into an EXISTING ray cluster (16 colocated train/rollout GPUs plus
one reward GPU on a separate node) -- see docs/user-guide/launch-script.md (Multi-node
training) for the bring-up, then run with MILES_SCRIPT_EXTERNAL_RAY=1. The reward
worker is default-scheduled, so it lands on the only GPU the engines do not occupy.

Topology, and why:
  tp_size 2          8 engines over 16 GPUs. tp 4 would halve the engine count for at
                     best the same throughput, and H3's TP output error is ~1e-2, which
                     widens the train-vs-rollout gap the recipe is validated on.
  dp_replicate 2     FSDP shards over 8 and replicates over 2; H3 has no train-side SP.
  recompute          --gradient-checkpointing is required: the backward OOMs
                     without it even at micro_batch_size_sample 1.

Rollout throughput: the rollout phase runs at 23.8% GPU utilization -- half of a
sampled window has the GPUs below 5% -- while the train phase sits at 98.9%. So every
bubble is on the rollout side. Engine occupancy is ~59s per sample of which only ~14s is
GPU; the rest is the ~1GB raw-float response being made contiguous, copied over PCIe,
re-serialized, and written out. --sglang-server-concurrency exists to overlap that
handoff with the next sample's denoise: at 1 the engine simply idles through it. But 4
stretched per-sample latency from ~70s to ~235s and held train_wait at 1061-1254s, so
most of that concurrency queues inside the engine instead of overlapping. 2 keeps the
overlap without the latency and in-flight-memory blowup. The parser and reward pools are
sized to the slots, since Ray actors are single-threaded and were queueing several deep.
Microgroups stay at 1 -- see below.

Batch: 8 prompts x 16 samples = 128 videos per rollout, split into 2 optimizer steps
(global_batch_size 64, which must stay divisible by dp_size 16). Two steps, not more:
--diffusion-clip-range is 1e-4, so inner steps past the second are mostly clipped -- the
data is already off-policy by then. This matches the verified 2-GPU recipe.

Usage (after the launch-script.md bring-up, on the head node):
    MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/run_diffusion_grpo_h3_t2va_17gpu.py

    # rollout-only smoke: sglang rollout + reward, no FSDP train
    MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/run_diffusion_grpo_h3_t2va_17gpu.py \
        --num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 1 \
        --num-steps-per-rollout 1 --eval-interval 0 --extra-args "--debug-rollout-only"

    # train/rollout alignment diagnostic: freeze the weights so log_prob_mean_abs_diff
    # measures pure train-vs-rollout deviation rather than parameter drift
    MILES_SCRIPT_EXTERNAL_RAY=1 python3 scripts/run_diffusion_grpo_h3_t2va_17gpu.py \
        --num-rollout 2 --eval-interval 0 --extra-args "--debug-skip-optimizer-step"
"""

from dataclasses import dataclass
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "MiniMaxAI/MiniMax-H3"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
WANDB_PROJECT = "miles-diffusion-grpo"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_rollout: int = 10000
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 16
    num_steps_per_rollout: int = 2
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
    assert U.get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY"), (
        "this recipe needs a 17-GPU ray cluster it does not bring up; see "
        "docs/user-guide/launch-script.md (Multi-node training), then run with MILES_SCRIPT_EXTERNAL_RAY=1."
    )
    run_name = f"diffusion_grpo_h3_t2va_17gpu_{U.create_run_id()}"

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
        # H3 returns exactly one video per /rollout/generate: its request carries a single
        # `target` spec, with no sample-count axis. microgroup > 1 hands the parser more
        # samples than bodies and dies in apply_raw's strict zip.
        "--rollout-microgroup-size 1 "
        "--micro-batch-size-sample 1 "
        "--micro-batch-size-tstep 1 "
        "--diffusion-train-iter-order sample_major "
    )

    diffusion_args = (
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 1.0 "
        "--diffusion-noise-level 0.7 "
        "--diffusion-sde-type sde "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window "
        "--diffusion-num-sde-steps 2 "
        "--diffusion-sde-window-range 1,4 "
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

    grpo_args = (
        "--advantage-estimator grpo --globalize-reward-std --diffusion-clip-range 1e-4 --diffusion-kl-beta 0.0 "
    )

    optimizer_args = "--lr 1e-4 --weight-decay 1e-4 "

    # H3's rollout DiT renames modules and fuses Q/K/V, so weights only reach the engine
    # through the LoRA IPC path's layer grouper; the family rejects any other sync mode.
    lora_args = "--use-lora --lora-ipc-weight-sync --lora-rank 64 --lora-alpha 128 "

    reward_args = (
        "--rm-type pickscore "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
        "--pickscore-num-frames 8 "
        "--pickscore-num-workers 4 "
        "--pickscore-num-gpus-per-worker 0.25 "
        "--pickscore-batch-size 8 "
        "--rollout-parser-num-workers 32 "
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
        "--dp-replicate-size 2 "
    )

    perf_args = "--gradient-checkpointing "

    misc_args = (
        "--actor-num-nodes 2 "
        "--actor-num-gpus-per-node 8 "
        "--rollout-num-gpus 16 "
        "--rollout-num-gpus-per-engine 2 "
        "--num-gpus-per-node 8 "
        "--colocate "
        "--wandb-log-image-interval 10 "
        "--rollout-health-check-interval 60 "
        "--miles-router-health-check-failure-threshold 30 "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {diffusion_args} {eval_args} {grpo_args} {optimizer_args} "
            f"{lora_args} {reward_args} {wandb_args} {dashboard_args} {sglang_args} {train_backend_args} "
            f"{perf_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=8,
        config=args,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    prompt_dir = prepare(args)
    execute(args, prompt_dir)


if __name__ == "__main__":
    typer.run(main)
