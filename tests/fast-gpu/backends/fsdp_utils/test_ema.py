"""Compare EMA weight switching with an independent FSDP2 reference model.

The launcher supplies the repository import path to each torchrun worker.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="stage-b-5-gpu-h200", labels=["fsdp"])

import os
import subprocess
import sys
from pathlib import Path

import pytest

_WORKER = Path(__file__).with_name("_ema_worker.py")


@pytest.mark.parametrize(
    "worker_args",
    [
        [],
        ["--checkpoint"],
        ["--checkpoint", "--bf16"],
        ["--cpu-offload"],
        ["--checkpoint", "--bf16", "--lora"],
    ],
    ids=["fp32", "fp32-checkpoint", "bf16-checkpoint", "fp32-cpu-offload", "bf16-lora-checkpoint"],
)
def test_ema_switch_matches_independent_reference(worker_args):
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(Path(__file__).resolve().parents[4]), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(_WORKER),
            *worker_args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
