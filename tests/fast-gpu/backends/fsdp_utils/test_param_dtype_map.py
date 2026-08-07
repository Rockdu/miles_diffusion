"""FSDP2 per-parameter mixed precision."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120,
    suite="stage-b-5-gpu-h200",
    labels=["fsdp"],
)

import os
import subprocess
import sys
from pathlib import Path


_WORKER = Path(__file__).with_name("_param_dtype_map_worker.py")


def test_param_dtype_map():
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            str(_WORKER),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
