#!/usr/bin/env python3
"""
Wan2.2-T2V-A14B 16-GPU (2x8) GRPO LoRA -- miles-LLM-style one-command multi-pod launcher.

Same ergonomics as miles core's run_*.py (subcommands + a config dataclass + a single
command that stands the whole multi-node run up). Stdlib-only (argparse). The multi-node
bootstrap is driven from the control host over ssh, exactly like core's .sh recipes
(`ssh <user>@<node>` + `ray start` per node): one command starts Ray across all pods and
launches train_diffusion.py on the head. --nodes are ssh targets (raw IPs, or Host aliases
from ~/.ssh/config); nodes[0] is the Ray head.

On RadixArk rx the pod containers run no sshd (a raw-IP ssh hits the node host, not the
container), so run `rx devbox ssh-config <node>` once per pod -- it installs an ssh alias
(ProxyCommand tunnels through `rx devbox run`) -- then pass those alias names as --nodes.

Layout: 16 train GPU across 2 IB pods colocate FSDP train + sglang rollout; PickScore
reward on its own 1-GPU pod (reward_node, non-colocate). Parallelism: train
dp_replicate=2 x sequence_parallel=8 (ulysses=8), rollout TP=4; recompute_logprob on.
Reproduces wandb run wan22_16gpu_bs24_ns1 (bs24, num-steps-per-rollout=1, peak <=40GB).

Usage (from the control host):
  python3 run_wan22_16gpu_lora.py prepare  --hf-token ...       # model+dataset, per pod
  python3 run_wan22_16gpu_lora.py train    --wandb-key ... --hf-token ...
  python3 run_wan22_16gpu_lora.py status
  python3 run_wan22_16gpu_lora.py down
"""
import argparse
import shlex
import subprocess
from dataclasses import dataclass

REPO = "/root/miles_diffusion"
MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"
DATA_ROOT = "/root/datasets/miles-diffusion-datasets"
DS = f"{DATA_ROOT}/{DATASET_SUBSET}"
WANDB_PROJECT = "miles-diffusion-grpo"

LORA_TARGET_MODULES = (
    "attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 "
    "attn2.to_q attn2.to_k attn2.to_v attn2.to_out.0 "
    "ffn.net.0.proj ffn.net.2"
)


@dataclass
class Cfg:
    nodes: tuple = ("kangrui-h200-new", "kangrui-h200-ltx")  # nodes[0] = Ray head
    reward_node: str = "kangrui-h200-reward"
    gpus_per_node: int = 8
    ray_port: int = 6379
    dp_replicate: int = 2
    sp: int = 8
    ulysses: int = 8
    rollout_tp: int = 4
    rollout_batch_size: int = 24
    n_samples_per_prompt: int = 8
    # =2 collapses reward: the off-policy 2nd step under a 1e-4 clip biases the gradient
    num_steps_per_rollout: int = 1
    microgroup_size: int = 2  # microgroup/micro_batch tuned down for <=40GB train peak
    micro_batch_size: int = 1
    num_rollout: int = 10000
    lora_rank: int = 64
    lora_alpha: int = 128

    @property
    def world(self):
        return len(self.nodes) * self.gpus_per_node


SSH_USER = "root"   # set from --ssh-user in main(); empty -> use node as-is (rely on ssh config)


def run_on(node, cmd, timeout=600):
    tgt = f"{SSH_USER}@{node}" if SSH_USER else node
    # ssh flattens argv after the host into one remote string, so hand it a SINGLE token
    # (else `bash -lc <cmd>` loses cmd's quoting remotely and only the first word runs).
    remote = "bash -lc " + shlex.quote(cmd)
    full = f"ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new {tgt} {shlex.quote(remote)}"
    p = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.stdout + p.stderr


def node_ip(node):
    return run_on(node, "hostname -i | awk '{print $1}'").strip().split("\n")[-1].strip()


def cmd_prepare(c: Cfg, hf_token: str):
    for n in c.nodes:
        print(f"[prepare] {n}")
        print(run_on(n, f"export HF_TOKEN={shlex.quote(hf_token)} HF_HUB_ENABLE_HF_TRANSFER=1; "
                   f"hf download {MODEL} >/dev/null 2>&1 || true; "
                   f"test -f {DS}/train.jsonl || hf download --repo-type dataset {DATASET} "
                   f"--include '{DATASET_SUBSET}/**' --local-dir {DATA_ROOT} >/dev/null 2>&1; "
                   f"echo prepared", timeout=3600).strip()[-120:])


def ray_up(c: Cfg):
    head_ip = node_ip(c.nodes[0])
    cvd = ",".join(str(i) for i in range(c.gpus_per_node))
    clean = ('pkill -9 -f "sgl_diffusio[n]" 2>/dev/null; ray stop --force >/dev/null 2>&1; sleep 2; '
             'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader|tr -d " "); do kill -9 $p 2>/dev/null; done; sleep 1')
    # every ray-actor (trainer + sglang engine) must inherit expandable_segments; a shell export
    # on the driver alone does NOT reach them, so set it in each node's ray-start env.
    alloc = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    run_on(c.nodes[0], f"{clean}; export CUDA_VISIBLE_DEVICES={cvd} {alloc}; ray start --head --node-ip-address {head_ip} "
                   f"--port {c.ray_port} --num-gpus {c.gpus_per_node} --disable-usage-stats >/dev/null 2>&1; echo head-up", timeout=120)
    for w in c.nodes[1:]:
        run_on(w, f"{clean}; export CUDA_VISIBLE_DEVICES={cvd} {alloc}; ray start --address {head_ip}:{c.ray_port} "
              f"--num-gpus {c.gpus_per_node} --disable-usage-stats >/dev/null 2>&1; echo worker-up", timeout=120)
    if c.reward_node:
        # dedicated 1-GPU reward worker; PACK keeps the 16 train bundles on the full nodes,
        # so the non-colocate pickscore actor is the only thing that can land here.
        run_on(c.reward_node, f"{clean}; export CUDA_VISIBLE_DEVICES=0 {alloc} HF_HOME=/cluster-storage/models/hf; "
              f"ray start --address {head_ip}:{c.ray_port} --num-gpus 1 --disable-usage-stats >/dev/null 2>&1; echo reward-up", timeout=120)
    print(run_on(c.nodes[0], "ray status 2>&1 | grep -E 'GPU|node_' | head"))
    return head_ip


def train_args(c: Cfg, wandb_key: str) -> str:
    # /personal is shared cluster-storage that survives a devbox release; /root is node-local
    save = f"/personal/wan22_{c.world}gpu_lora/ckpt"

    ckpt_args = f"--hf-checkpoint {MODEL} --save {save} --save-interval 10 "

    rollout_args = (
        "--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {DS}/train.jsonl --input-key input "
        f"--rollout-batch-size {c.rollout_batch_size} --n-samples-per-prompt {c.n_samples_per_prompt} "
        f"--num-rollout {c.num_rollout} "
        f"--num-steps-per-rollout {c.num_steps_per_rollout} "
        f"--rollout-microgroup-size {c.microgroup_size} "
        "--diffusion-num-steps 10 --diffusion-output-num-frames 21 "
        "--diffusion-guidance-scale 4.0 --diffusion-guidance-scale-2 3.0 --diffusion-noise-level 0.9 "
        "--diffusion-height 480 --diffusion-width 480 --diffusion-flow-shift 3.0 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.epoch_global_random_choice "
        "--diffusion-num-sde-steps 1 --diffusion-sde-candidate-steps 1,2,3 "
    )

    eval_args = (
        f"--eval-prompt-data pickscore_test {DS}/test.jsonl "
        "--eval-interval 100 --diffusion-eval-num-steps 28 --skip-eval-before-train "
    )

    grpo_args = "--advantage-estimator grpo --diffusion-clip-range 1e-4 --diffusion-recompute-old-log-prob "

    optimizer_args = "--lr 1e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    lora_args = (
        # lora-ipc-weight-sync: only lora_A/B sync to the engine via CUDA IPC (no full 2x14B gather)
        "--use-lora --lora-ipc-weight-sync "
        f"--lora-rank {c.lora_rank} --lora-alpha {c.lora_alpha} "
        f"--lora-target-modules {LORA_TARGET_MODULES} --lora-init-weights gaussian "
    )

    # reward on its own GPU (reward_node): non-colocate, 1 worker x full GPU -> lands on the 1-GPU pod
    reward_args = (
        "--rm-type pickscore "
        "--pickscore-num-workers 1 --pickscore-num-gpus-per-worker 1 --pickscore-batch-size 8 "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
    )

    wandb_args = ""
    if wandb_key:
        wandb_args = (
            f"--use-wandb --wandb-project {WANDB_PROJECT} --wandb-group wan22_16gpu_lora "
            "--disable-wandb-random-suffix --wandb-log-num-images 8 --wandb-log-image-interval 10 "
            f"--wandb-key {wandb_key} "
        )
    wandb_args += f"--use-miles-dashboard --miles-dashboard-workspace {REPO}/miles_dashboard "

    # sglang engine: router (raise health-check threshold so a busy eval isn't false-marked DEAD) +
    # colocate memory offload (VAE + text-encoder + layerwise DiT keep the rollout phase <=40GB).
    sglang_args = (
        "--use-miles-router --miles-router-health-check-failure-threshold 30 "
        "--sglang-server-concurrency 8 --update-weight-buffer-size 536870912 "
        "--sglang-vae-cpu-offload --sglang-text-encoder-cpu-offload "
        "--sglang-dit-layerwise-offload --sglang-dit-offload-prefetch-size 2 "
    )

    train_backend_args = (
        "--train-backend fsdp --fsdp-master-dtype fp32 --fsdp-reduce-dtype fp32 --diffusion-forward-dtype bf16 "
        "--update-weight-target-module transformer,transformer_2 "
    )

    # perf/memory: gradient-checkpointing caps the train-phase activation spike for peak <=40GB
    perf_args = f"--micro-batch-size {c.micro_batch_size} --gradient-checkpointing "

    misc_args = (
        f"--actor-num-nodes {len(c.nodes)} --actor-num-gpus-per-node {c.gpus_per_node} "
        f"--num-gpus-per-node {c.gpus_per_node} "
        f"--rollout-num-gpus {c.world} --rollout-num-gpus-per-engine {c.rollout_tp} "
        f"--dp-replicate-size {c.dp_replicate} --sequence-parallel-size {c.sp} --ulysses-degree {c.ulysses} "
        "--colocate --diffusion-debug-mode "
    )

    return (
        f"{ckpt_args}{rollout_args}{eval_args}{grpo_args}{optimizer_args}"
        f"{lora_args}{reward_args}{wandb_args}{sglang_args}{train_backend_args}{perf_args}"
        f"{misc_args}"
    )


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
    print(run_on(c.nodes[0], launch, timeout=120))


def cmd_status(c: Cfg):
    for n in c.nodes:
        print(f"[{n}] " + run_on(n, "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | paste -sd' ' -").strip())


def cmd_down(c: Cfg):
    for n in c.nodes:
        run_on(n, 'pkill -9 -f "train_diffusio[n]" 2>/dev/null; pkill -9 -f "sgl_diffusio[n]" 2>/dev/null; ray stop --force >/dev/null 2>&1; echo down')
    print("cluster torn down")


def main():
    d = Cfg()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "train", "status", "down"):
        sp = sub.add_parser(name)
        sp.add_argument("--nodes", default=",".join(d.nodes))
        sp.add_argument("--ssh-user", default="root", help="ssh login user; empty -> use --nodes verbatim (ssh config)")
        if name in ("prepare", "train"):
            sp.add_argument("--hf-token", default="")
        if name == "train":
            sp.add_argument("--wandb-key", default="")
            sp.add_argument("--dp-replicate", type=int, default=d.dp_replicate)
            sp.add_argument("--sp", type=int, default=d.sp)
            sp.add_argument("--ulysses", type=int, default=d.ulysses)
            sp.add_argument("--rollout-tp", type=int, default=d.rollout_tp)
            sp.add_argument("--micro-batch-size", type=int, default=d.micro_batch_size)
            sp.add_argument("--microgroup-size", type=int, default=d.microgroup_size)
            sp.add_argument("--rollout-batch-size", type=int, default=d.rollout_batch_size)
    args = p.parse_args()
    global SSH_USER
    SSH_USER = args.ssh_user
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
