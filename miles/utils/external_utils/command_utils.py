"""
This file is not for miles framework itself, but as an optional utility to easily launch miles jobs and tests.
"""

import datetime
import json
import os
import random
import shlex
from dataclasses import dataclass
from pathlib import Path

from miles.utils.misc import exec_command
from miles.utils.typer_utils import dataclass_cli

_ = exec_command, dataclass_cli

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[3]


def hf_download_dataset(full_name: str, include: str | None = None, data_dir: str = "/root/datasets") -> str:
    """Download a dataset (or one subdirectory of it) and return the local path."""
    _, partial_name = full_name.split("/")
    local_dir = f"{data_dir}/{partial_name}"
    include_arg = f"--include {shlex.quote(include)} " if include else ""
    exec_command(f"hf download --repo-type dataset {full_name} {include_arg}--local-dir {shlex.quote(local_dir)}")
    return local_dir


def ssh_start_ray_workers(
    master_addr: str,
    num_gpus_per_node: int,
    hostfile: str = "/root/h3_hostfile",
    head_host: str | None = None,
    ssh_port: int = 2222,
) -> None:
    """Join every host in a hostfile to the ray cluster over ssh, in parallel.

    Ray itself cannot bring up the workers: the head is already running locally
    and the workers have no agent yet. Pass this as
    ``execute_train(before_ray_job_submit=...)`` so the cluster is complete
    before the job is submitted; the placement group's ``ready()`` then blocks
    until every worker has joined.

    A hostfile line is ``ip [num_gpus]``; the optional second column covers
    heterogeneous nodes (e.g. a 2-GPU reward node). Every worker is killed
    clean (sglang, sgl_diffusion, miles, its raylet) before rejoining — a stale
    raylet is how engine/trainer GPU pairing drifts apart. The devbox pods
    share the node's network namespace, so sshd listens on ``ssh_port``
    (port 22 belongs to the host machine).
    """
    head_host = head_host or master_addr
    exec_command(
        f"while read -r worker_ip worker_gpus _; do "
        f'[ -z "$worker_ip" ] && continue; '
        f'if [ "$worker_ip" = {shlex.quote(head_host)} ]; then continue; fi; '
        f'echo "Starting Ray worker on $worker_ip"; '
        f'ssh -p {ssh_port} -o StrictHostKeyChecking=no root@"$worker_ip" '
        f'"pkill -9 sglang ; pkill -9 sgl_diffusion ; ray stop --force ; pkill -9 miles ; pkill -9 ray ; '
        f"ray start --address={master_addr}:6379 --num-gpus ${{worker_gpus:-{num_gpus_per_node}}} "
        f'--node-ip-address $worker_ip --disable-usage-stats" & '
        f"done < {shlex.quote(hostfile)}; wait"
    )


# This class can be extended by concrete scripts
@dataclass
class ExecuteTrainConfig:
    cuda_core_dump: bool = False
    num_nodes: int = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    cuda_visible_devices: str = ""
    extra_env_vars: str = ""
    output_dir: str = str(repo_base_dir / "logs")


def _cvd_export(config: ExecuteTrainConfig, num_gpus_per_node: int) -> str:
    """Shell prefix pinning the raylet's GPUs; empty means inherit the caller's environment."""
    if not config.cuda_visible_devices:
        return ""
    visible = [x.strip() for x in config.cuda_visible_devices.split(",") if x.strip()]
    assert len(visible) == num_gpus_per_node, (
        f"--cuda-visible-devices lists {len(visible)} GPU(s) ({config.cuda_visible_devices}) but this "
        f"recipe runs on {num_gpus_per_node}; ray would hand out ids the visible list does not have."
    )
    return f"export CUDA_VISIBLE_DEVICES={','.join(visible)} && "


def execute_train(
    train_args: str,
    num_gpus_per_node: int,
    config: ExecuteTrainConfig | None = None,
    train_script: str = "train_diffusion.py",
    before_ray_job_submit=None,
    extra_env_vars: dict[str, str] | None = None,
) -> None:
    """Start a Ray cluster if we own one, then submit the trainer into it.

    Set MILES_SCRIPT_EXTERNAL_RAY=1 when a scheduler already built the cluster: the
    teardown and `ray start` are skipped and the job is submitted to the running one.
    Submitting rather than running `python` directly is what makes the driver live in
    the cluster, so it sees every node's GPUs and every worker gets the same runtime env.
    """
    if config is None:
        config = ExecuteTrainConfig()
    if not os.path.isabs(train_script):
        train_script = f"{repo_base_dir}/{train_script}"
    external_ray = get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY")
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

    exec_command(
        "pkill -9 sglang; "
        "pkill -9 sgl_diffusion; "
        "sleep 3; "
        f"{'' if external_ray else 'ray stop --force; '}"
        f"{'' if external_ray else 'pkill -9 ray; '}"
        "pkill -9 miles; "
        "sleep 3; "
        "pkill -9 redis; "
        "true; "
    )

    if not external_ray:
        exec_command(
            # Ray reads the device list once, at raylet startup; set per job or per actor
            # it never reaches the scheduler, which then places work on excluded GPUs.
            f"{_cvd_export(config, num_gpus_per_node)}"
            # keeps ray from buffering stdout/stderr
            "export PYTHONUNBUFFERED=1 && "
            f"ray start --head --node-ip-address {master_addr} "
            f"--num-gpus {num_gpus_per_node} --disable-usage-stats"
        )
    else:
        assert not config.cuda_visible_devices, (
            "--cuda-visible-devices cannot be applied to an externally started cluster: "
            "export CUDA_VISIBLE_DEVICES before `ray start` instead."
        )

    if (f := before_ray_job_submit) is not None:
        f()

    runtime_env_vars = {
        # exported for the submitting client too, but only the runtime env reaches the ray workers
        "PYTHONUNBUFFERED": "1",
        "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", str(int(check_has_nvlink()))),
        **{
            k: os.environ[k]
            for k in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_DEBUG", "NCCL_DEBUG_FILE")
            if k in os.environ
        },
        "no_proxy": f"127.0.0.1,{master_addr}",
        # torch distributed needs this on every node, not just the submitting shell.
        "MASTER_ADDR": master_addr,
        **(
            {
                "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
                "CUDA_COREDUMP_SHOW_PROGRESS": "1",
                "CUDA_COREDUMP_GENERATION_FLAGS": "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory",
                "CUDA_COREDUMP_FILE": f"{config.output_dir}/cuda_coredump_%h.%p.%t",
            }
            if config.cuda_core_dump
            else {}
        ),
        **(extra_env_vars or {}),
        **_parse_extra_env_vars(config.extra_env_vars),
    }
    runtime_env_vars["PYTHONPATH"] = _pythonpath_with_sources(runtime_env_vars.get("PYTHONPATH"))
    runtime_env_json = json.dumps({"env_vars": runtime_env_vars})

    if not get_bool_env_var("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1"):
        return

    exec_command(
        "export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && "
        f"""ray job submit {'' if 'RAY_ADDRESS' in os.environ else '--address="http://127.0.0.1:8265" '}"""
        f"--runtime-env-json={shlex.quote(runtime_env_json)} "
        f"-- python3 {shlex.quote(train_script)} {train_args}"
    )


def _pythonpath_with_sources(*additional_pythonpaths: str | None) -> str:
    entries = [str(repo_base_dir)]
    for pythonpath in (*additional_pythonpaths, os.environ.get("PYTHONPATH")):
        if pythonpath:
            entries.extend(pythonpath.split(os.pathsep))
    return os.pathsep.join(dict.fromkeys(entries))


def _parse_extra_env_vars(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return {kv[0]: kv[1] for item in text.split(" ") if item.strip() != "" if (kv := item.split("=")) or True}


def check_has_nvlink():
    output = exec_command("nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l", capture_output=True)
    return int(output) > 0


def get_default_wandb_args(
    test_file: str,
    run_name_prefix: str | None = None,
    run_id: str | None = None,
    project: str | None = None,
    **extra_flags: object,
) -> str:
    """miles' wandb args, plus a `project` override.

    miles derives one project per script from the caller's filename. The diffusion
    recipes deliberately share three projects (grpo, nft, sft) so runs across models stay
    comparable in one panel, which a filename cannot express.
    """
    if not os.environ.get("WANDB_API_KEY"):
        print("Skip wandb configuration since WANDB_API_KEY is not found")
        return ""

    test_file = Path(test_file)
    test_name = test_file.stem
    if len(test_name) < 6:
        test_name = f"{test_file.parent.name}_{test_name}"

    wandb_run_name = run_id or create_run_id()
    if (x := os.environ.get("GITHUB_COMMIT_NAME")) is not None:
        wandb_run_name += f"_{x}"
    if (x := run_name_prefix) is not None:
        wandb_run_name = f"{x}_{wandb_run_name}"

    # Use the actual key value from environment to avoid shell expansion issues
    wandb_key = os.environ.get("WANDB_API_KEY")
    args = (
        "--use-wandb "
        f"--wandb-project {project or f'miles-{test_name}'} "
        f"--wandb-group {wandb_run_name} "
        f"--wandb-key '{wandb_key}' "
        "--disable-wandb-random-suffix "
    )
    for name, value in extra_flags.items():
        args += f"--{name.replace('_', '-')} {value} "
    return args


def create_run_id() -> str:
    return datetime.datetime.utcnow().strftime("%y%m%d-%H%M%S") + f"-{random.Random().randint(0, 999):03d}"


_warned_bool_env_var_keys = set()


# copied from SGLang
def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            print(f"get_bool_env_var({name}) see non-understandable value={value} and treat as false")
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values
