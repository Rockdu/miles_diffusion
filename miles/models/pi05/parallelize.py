"""Apply activation checkpointing, compile, and FSDP2 to pi0.5.

Only the ``dp_shard`` axis is filled. TP, PP, CP and EP are absent by intent,
not by omission: adding them means adding a ``sharding.py`` and setting
activation placements here, and none of the model code changes.
"""

import logging
from typing import Any

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    fully_shard,
    MixedPrecisionPolicy,
)

from torchtitan.config import (
    CompileConfig,
    FSDPSymmMemScope,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import resolve_fsdp_mesh

from .model import Pi05Model

logger = logging.getLogger(__name__)


def parallelize_pi05(
    model: Pi05Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    model.parallelize(parallel_dims)

    if compile_config.enable and "model" in compile_config.components:
        apply_compile(
            model, compile_config=compile_config, parallel_dims=parallel_dims
        )

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cpu_offload=training.enable_cpu_offload,
        symm_mem_scope=parallelism.fsdp_symm_mem_scope,
        dp_mesh_dims=dp_mesh_dims,
    )
    logger.info("Applied fully_shard to pi0.5")

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: Any,
    reduce_dtype: Any,
    cpu_offload: bool = False,
    symm_mem_scope: FSDPSymmMemScope = None,
    dp_mesh_dims: DataParallelMeshDims | None = None,
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # One wrap per DualExpertBlock, so each all-gather covers both experts' slice
    # of a layer and the adaRMS Dense that modulates it. Splitting the modulation
    # into its own wrap would add 37 small collectives for ~117M parameters; if
    # that turns out to be the wrong tradeoff it is a change here only.
    for block in model.layers:
        fully_shard(block, **fsdp_config)

    fully_shard(model.tok_embeddings, **fsdp_config)
    fully_shard(model, **fsdp_config)
