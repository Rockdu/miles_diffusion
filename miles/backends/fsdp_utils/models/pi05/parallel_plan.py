"""pi0.5 FSDP wrapping declarations.

One wrap per ``DualExpertBlock``, so a single all-gather covers both experts'
slice of a layer plus the adaRMS Denses that modulate it. Those Denses are
~117M parameters against a ~311M action expert, so splitting them into their
own wraps would add 37 small collectives for no obvious gain.
"""

from __future__ import annotations

import torch

from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan
from miles.backends.fsdp_utils.sequence_parallel.plan import SequenceParallelPlan

FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    no_split_modules=("DualExpertBlock",),
)


def sequence_parallel_plan(model: torch.nn.Module) -> SequenceParallelPlan:
    raise NotImplementedError(
        "pi0.5 does not support sequence parallelism yet. Its prefix is under a "
        "thousand tokens, so USP has little to shard; revisit if multi-view or "
        "longer action chunks change that."
    )
