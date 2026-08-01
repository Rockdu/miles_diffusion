#!/usr/bin/env python3
"""
Wan2.2-T2V-A14B 16-GPU (2x8) GRPO LoRA — miles-LLM-style one-command multi-pod launcher.

Same ergonomics as miles core's run_*.py (subcommands + a config dataclass + a single
command that stands the whole multi-node run up). Stdlib-only (argparse), so it runs on
the control host where `rx` is authed — RadixArk rx pods have no inter-pod ssh / MPI
hostfile that the core .sh recipes assume, so the multi-node bootstrap is driven from the
control host via `rx devbox run`: one command starts Ray across all pods and launches
train_diffusion.py on the head.

Layout "16-GPU colocate": 16 GPU across 2 IB pods, all colocate FSDP train + sglang
rollout; PickScore reward colocated (no separate reward pod). Parallelism: train
dp_replicate=2 x sequence_parallel=8 (ulysses=8), rollout TP=4; recompute_logprob on.

NOTE: rollout TP>1 needs the two-line SGL-D fix in
miles/backends/sglang_diffusion_utils/sglang_diffusion_engine.py (set server_args
num_gpus=rollout_num_gpus_per_engine + pin all tp GPUs in _pin_to_assigned_gpu),
tracked as a separate framework PR. With rollout_tp=1 the launcher needs no patch.

Usage (from the control host):
  python3 run_wan22_16gpu_lora.py prepare  --hf-token ...       # model+dataset, per pod
  python3 run_wan22_16gpu_lora.py train    --wandb-key ... --hf-token ...
  python3 run_wan22_16gpu_lora.py status
  python3 run_wan22_16gpu_lora.py down
"""
import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass

WAN = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
REPO = "/root/miles_diffusion"
DS = "/root/datasets/miles-diffusion-datasets/flowgrpo_pickscore"
ENGINE = f"{REPO}/miles/backends/sglang_diffusion_utils/sglang_diffusion_engine.py"
LORA_TARGETS = ("attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 "
                "attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0 ffn.net.0.proj ffn.net.2")


@dataclass
class Cfg:
    nodes: tuple = ("kangrui-h200-new", "kangrui-h200-ltx")  # nodes[0] = Ray head
    reward_node: str = "kangrui-h200-reward"  # dedicated 1-GPU pod for the pickscore reward
    gpus_per_node: int = 8
    ray_port: int = 6379
    # train parallelism
    dp_replicate: int = 2
    sp: int = 8
    ulysses: int = 8
    # rollout
    rollout_tp: int = 4
    # batch / schedule
    rollout_batch_size: int = 24
    n_samples_per_prompt: int = 8
    num_steps_per_rollout: int = 1  # 1 optim step/rollout: fully on-policy. =2 collapses reward (off-policy 2nd step under 1e-4 clip)
    microgroup_size: int = 2        # tuned down (with gradient ckpt) for <=40GB train peak
    micro_batch_size: int = 1       # tuned down from 2 for <=40GB
    num_rollout: int = 10000
    lora_rank: int = 64
    lora_alpha: int = 128

    @property
    def world(self):
        return len(self.nodes) * self.gpus_per_node


def rx(node, cmd, timeout=600):
    p = subprocess.run(f"rx devbox run {node} -- bash -lc {shlex.quote(cmd)}",
                       shell=True, capture_output=True, text=True, timeout=timeout)
    return (p.stdout or "") + (p.stderr or "")


def node_ip(node):
    return rx(node, "hostname -i | awk '{print $1}'").strip().split("\n")[-1].strip()


def cmd_prepare(c: Cfg, hf_token: str):
    for n in c.nodes:
        print(f"[prepare] {n}")
        print(rx(n, f"export HF_TOKEN={shlex.quote(hf_token)} HF_HUB_ENABLE_HF_TRANSFER=1; "
                   f"hf download {WAN} >/dev/null 2>&1 || true; "
                   f"test -f {DS}/train.jsonl || hf download --repo-type dataset rockdu/miles-diffusion-datasets "
                   f"--include 'flowgrpo_pickscore/**' --local-dir {'/'.join(DS.split('/')[:-1])} >/dev/null 2>&1; "
                   f"echo prepared", timeout=3600).strip()[-120:])


def ray_up(c: Cfg):
    head_ip = node_ip(c.nodes[0])
    cvd = ",".join(str(i) for i in range(c.gpus_per_node))
    clean = ('pkill -9 -f "sgl_diffusio[n]" 2>/dev/null; ray stop --force >/dev/null 2>&1; sleep 2; '
             'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader|tr -d " "); do kill -9 $p 2>/dev/null; done; sleep 1')
    # every ray-actor (trainer + sglang engine) must inherit expandable_segments; a shell export
    # on the driver alone does NOT reach them, so set it in each node's ray-start env.
    alloc = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    rx(c.nodes[0], f"{clean}; export CUDA_VISIBLE_DEVICES={cvd} {alloc}; ray start --head --node-ip-address {head_ip} "
                   f"--port {c.ray_port} --num-gpus {c.gpus_per_node} --disable-usage-stats >/dev/null 2>&1; echo head-up", timeout=120)
    for w in c.nodes[1:]:
        rx(w, f"{clean}; export CUDA_VISIBLE_DEVICES={cvd} {alloc}; ray start --address {head_ip}:{c.ray_port} "
              f"--num-gpus {c.gpus_per_node} --disable-usage-stats >/dev/null 2>&1; echo worker-up", timeout=120)
    if c.reward_node:
        # dedicated 1-GPU reward worker; PACK keeps the 16 train bundles on the full nodes,
        # so the non-colocate pickscore actor is the only thing that can land here.
        rx(c.reward_node, f"{clean}; export CUDA_VISIBLE_DEVICES=0 {alloc} HF_HOME=/cluster-storage/models/hf; "
              f"ray start --address {head_ip}:{c.ray_port} --num-gpus 1 --disable-usage-stats >/dev/null 2>&1; echo reward-up", timeout=120)
    print(rx(c.nodes[0], "ray status 2>&1 | grep -E 'GPU|node_' | head"))
    return head_ip


def train_args(c: Cfg, wandb_key: str) -> str:
    # ckpt on shared cluster-storage (/personal survives devbox release), NOT node-local /root
    save = f"/personal/wan22_{c.world}gpu_lora/ckpt"
    a = [
        "--train-backend fsdp",
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout",
        f"--hf-checkpoint {WAN} --diffusion-model {WAN}",
        f"--prompt-data {DS}/train.jsonl --input-key input",
        f"--rollout-batch-size {c.rollout_batch_size} --n-samples-per-prompt {c.n_samples_per_prompt}",
        f"--num-rollout {c.num_rollout} --num-steps-per-rollout {c.num_steps_per_rollout}",
        f"--diffusion-microgroup-size {c.microgroup_size} --micro-batch-size {c.micro_batch_size}",
        f"--actor-num-nodes {len(c.nodes)} --actor-num-gpus-per-node {c.gpus_per_node}",
        f"--num-gpus-per-node {c.gpus_per_node} --rollout-num-gpus {c.world}",
        f"--rollout-num-gpus-per-engine {c.rollout_tp} --colocate",
        f"--dp-replicate-size {c.dp_replicate} --sequence-parallel-size {c.sp} --ulysses-degree {c.ulysses}",
        # colocate memory: free VAE + text-encoder off the shared GPU during denoise
        "--sglang-vae-cpu-offload --sglang-text-encoder-cpu-offload",
        # explicit DiT layerwise offload (was implicitly on via performance_mode=auto for
        # wan2.2-a14b); pin it so the rollout-phase <=40GB does not depend on the auto heuristic
        "--sglang-dit-layerwise-offload --sglang-dit-offload-prefetch-size 2",
        f"--use-lora --lora-rank {c.lora_rank} --lora-alpha {c.lora_alpha}",
        f"--lora-target-modules {LORA_TARGETS} --diffusion-init-lora-weight gaussian",
        "--lr 1e-4 --adam-beta2 0.999 --diffusion-clip-range 1e-4 --weight-decay 1e-4",
        "--diffusion-recompute-old-log-prob",
        # gradient checkpointing: caps the train-phase activation spike to keep peak <=40GB
        "--gradient-checkpointing",
        # LoRA IPC weight sync: only lora_A/B go to the engine via CUDA IPC (no full 2x14B gather)
        "--lora-ipc-weight-sync",
        # raise router health-check threshold: a busy engine (heavy eval) else misses /health -> false DEAD
        "--use-miles-router --miles-router-health-check-failure-threshold 30 --sglang-server-concurrency 8 --update-weight-buffer-size 536870912",
        "--update-weight-target-module transformer,transformer_2",
        "--diffusion-reward pickscore:1.0 --advantage-estimator grpo --rm-type pickscore",
        # reward on its own GPU (reward_node): non-colocate, 1 worker * full GPU -> lands on the 1-GPU pod
        "--pickscore-num-workers 1 --pickscore-num-gpus-per-worker 1 --pickscore-batch-size 8",
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K --pickscore-model-path yuvalkirstain/PickScore_v1",
        "--fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16",
        "--diffusion-num-steps 10 --diffusion-eval-num-steps 28 --diffusion-output-num-frames 21",
        "--diffusion-guidance-scale 4.0 --diffusion-guidance-scale-2 3.0 --diffusion-noise-level 0.9",
        "--diffusion-height 480 --diffusion-width 480 --diffusion-flow-shift 3.0",
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice",
        "--diffusion-num-sde-steps 1 --diffusion-sde-candidate-steps 1,2,3 --diffusion-debug-mode",
        f"--save {save} --save-interval 10",
        f"--eval-prompt-data pickscore_test {DS}/test.jsonl --eval-interval 100 --skip-eval-before-train",
        f"--use-miles-dashboard --miles-dashboard-workspace {REPO}/miles_dashboard",
    ]
    if wandb_key:
        a.append("--use-wandb --wandb-project miles-diffusion-grpo --wandb-group wan22_16gpu_lora "
                 "--disable-wandb-random-suffix --diffusion-log-images 8 --diffusion-log-image-interval 10 "
                 f"--wandb-key {wandb_key}")
    return " ".join(a)


def cmd_train(c: Cfg, wandb_key: str, hf_token: str):
    assert c.world % (c.dp_replicate * c.sp) == 0, "world not divisible by dp_replicate*sp"
    print(f"[train] nodes={c.nodes} world={c.world} dp_rep={c.dp_replicate} sp={c.sp} rollout_tp={c.rollout_tp} "
          f"micro_batch={c.micro_batch_size} microgroup={c.microgroup_size}")
    ray_up(c)
    env = (f"export RAY_ADDRESS=auto PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
           f"HF_TOKEN={shlex.quote(hf_token)} WANDB_API_KEY={shlex.quote(wandb_key)}; ")
    launch = (f"cd {REPO} && {env} setsid bash -c 'python -u train_diffusion.py {train_args(c, wandb_key)} "
              f"> /root/train_{c.world}gpu.log 2>&1' < /dev/null & disown; sleep 4; "
              f"echo launched pid $(pgrep -f train_diffusio[n] | head -1)")
    print(rx(c.nodes[0], launch, timeout=120))


def cmd_status(c: Cfg):
    print(rx(c.nodes[0], f"grep -vE 'GET /health|POST /add_worker|server_args:' /root/train_{c.world}gpu.log 2>/dev/null | tail -20"))
    for n in c.nodes:
        print(f"[{n}] " + rx(n, "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | paste -sd' ' -").strip())


def cmd_down(c: Cfg):
    for n in c.nodes:
        rx(n, 'pkill -9 -f "train_diffusio[n]" 2>/dev/null; pkill -9 -f "sgl_diffusio[n]" 2>/dev/null; ray stop --force >/dev/null 2>&1; echo down')
    print("cluster torn down")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "train", "status", "down"):
        sp = sub.add_parser(name)
        sp.add_argument("--nodes", default="kangrui-h200-new,kangrui-h200-ltx")
        if name in ("prepare", "train"):
            sp.add_argument("--hf-token", default="")
        if name == "train":
            sp.add_argument("--wandb-key", default="")
            sp.add_argument("--dp-replicate", type=int, default=2)
            sp.add_argument("--sp", type=int, default=8)
            sp.add_argument("--ulysses", type=int, default=8)
            sp.add_argument("--rollout-tp", type=int, default=4)
            sp.add_argument("--micro-batch-size", type=int, default=1)
            sp.add_argument("--microgroup-size", type=int, default=4)
            sp.add_argument("--rollout-batch-size", type=int, default=48)
    args = p.parse_args()
    nodes = tuple(n.strip() for n in args.nodes.split(",") if n.strip())
    if args.cmd == "prepare":
        cmd_prepare(Cfg(nodes=nodes), args.hf_token)
    elif args.cmd == "train":
        c = Cfg(nodes=nodes, dp_replicate=args.dp_replicate, sp=args.sp, ulysses=args.ulysses,
                rollout_tp=args.rollout_tp, micro_batch_size=args.micro_batch_size,
                microgroup_size=args.microgroup_size, rollout_batch_size=args.rollout_batch_size)
        cmd_train(c, args.wandb_key, args.hf_token)
    elif args.cmd == "status":
        cmd_status(Cfg(nodes=nodes))
    elif args.cmd == "down":
        cmd_down(Cfg(nodes=nodes))


if __name__ == "__main__":
    main()
