"""Two-rank CPU regression for the FSDP2 weight-backup compatibility matrix.

    layout: replicate x shard = (2, 1) or (1, 2)
       x training: full parameters or PEFT with nonzero initial LoRA weights
       x backup: actor / initial reference / base / different base / EMA / EMA copy
                                  |
    actor forward --> reference forwards --> restored actor backward --> AdamW
         |                                  |                           |
    independent model ----------------------+--> averaged gradient oracle
                                                                        |
                                                two independent EMA schedules

The worker disables post-forward resharding to exercise cache refresh, and
compares values, gradients, optimizer state, bindings, and copied EMA snapshots.
Adapter-disabled forwards must restore the sharded trainable mask before actor backward.
Base-only forwards explicitly disable the adapter; this tests tensor operations,
not reference configuration/checkpoint loading. CPU Gloo does not cover CUDA
offload, sleep/wake, or rollout publication.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="stage-a-cpu", labels=["fsdp"])

import os
import subprocess
import sys
from pathlib import Path

_WORKER = Path(__file__).with_name("_weight_backup_matrix_worker.py")


def test_weight_backup_matrix():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(_WORKER.resolve().parents[4]) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--local-addr=127.0.0.1",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(_WORKER),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: 4 layouts/training combinations" in result.stdout
    print(result.stdout)
