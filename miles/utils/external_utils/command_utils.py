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


def hf_download_dataset(full_name: str, include: str | None = None, data_dir: str = "/root/datasets"):
    _, partial_name = full_name.split("/")
    include_arg = f"--include {shlex.quote(include)} " if include else ""
    exec_command(f"hf download --repo-type dataset {full_name} {include_arg}--local-dir {data_dir}/{partial_name}")


# This class can be extended by concrete scripts
@dataclass
class ExecuteTrainConfig:
    num_nodes: int = 1
    cuda_visible_devices: str = ""
    extra_env_vars: str = ""
    output_dir: str = str(repo_base_dir / "logs")


def execute_train(
    train_args: str,
    config: ExecuteTrainConfig,
    run_name: str,
    train_script: str = "train_diffusion.py",
    extra_env_vars: dict[str, str] | None = None,
):
    """Run the diffusion trainer in-process via `python -u`.

    Unlike miles, the rollout engines are sglang-diffusion servers the trainer
    starts itself, so there is no Ray job submission or Megatron model sourcing
    to set up here.
    """
    if not os.path.isabs(train_script):
        train_script = f"{repo_base_dir}/{train_script}"

    env_vars = {
        **(extra_env_vars or {}),
        **({"CUDA_VISIBLE_DEVICES": config.cuda_visible_devices} if config.cuda_visible_devices else {}),
        **_parse_extra_env_vars(config.extra_env_vars),
    }
    exports = "".join(f"export {k}={shlex.quote(v)} && " for k, v in env_vars.items())

    log_dir = Path(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"

    exec_command(
        f"{exports}"
        f"python3 -u {train_script} "
        f"{train_args}"
        f"2>&1 | tee -a {shlex.quote(str(log_file))}"
    )


def _parse_extra_env_vars(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return {kv[0]: kv[1] for item in text.split(" ") if item.strip() != "" if (kv := item.split("=")) or True}


def get_default_wandb_args(project: str, run_name: str, log_images: int = 8, log_image_interval: int = 10):
    if not os.environ.get("WANDB_API_KEY"):
        print("Skip wandb configuration since WANDB_API_KEY is not found")
        return ""

    # Use the actual key value from environment to avoid shell expansion issues
    wandb_key = os.environ["WANDB_API_KEY"]
    return (
        "--use-wandb "
        f"--wandb-project {project} "
        f"--wandb-group {run_name} "
        f"--wandb-key '{wandb_key}' "
        f"--diffusion-log-images {log_images} "
        f"--diffusion-log-image-interval {log_image_interval} "
        "--disable-wandb-random-suffix "
    )


def create_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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
