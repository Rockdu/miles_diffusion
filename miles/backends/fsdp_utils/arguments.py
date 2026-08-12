import argparse
import dataclasses
import importlib
from dataclasses import dataclass

import yaml


@dataclass
class FSDPArgs:
    # Optim
    optimizer: str = "adam"  # Optimizer type: "adam" (AdamW)
    lr: float = 2e-5
    lr_warmup_init: float = 0.0
    min_lr: float = 0.0
    lr_decay_style: str = "constant"
    lr_decay_iters: int | None = None
    lr_warmup_iters: int = 0
    lr_warmup_fraction: float | None = None
    lr_wsd_decay_iters: int | None = None
    lr_wsd_decay_style: str | None = None
    use_checkpoint_lr_scheduler: bool = True
    override_lr_scheduler: bool = False
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    # Aligned with flow_grpo (config/base.py:80) and PyTorch's Adam paper default.
    # Old miles default was 0.95 (LLM-pretraining convention) — switched here so
    # users who forget --adam-beta2 don't silently fall out of sync with flow_grpo
    # diffusion comparisons.
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # sglang-diffusion's backend words (see validate_attention_backend); None keeps
    # the model backend's default. Ring and deterministic_mode accept subsets.
    fsdp_attention_backend: str | None = None

    # Logging
    wandb_project: str = "miles-fsdp"

    # Precision
    gradient_checkpointing: bool = False

    # FSDP configuration
    fsdp_cpu_offload: bool = (
        False  # If True, offload parameters, gradients, and optimizer states to CPU (optimizer runs on CPU)
    )
    fsdp_cpu_backend: str | None = (
        "gloo"  # CPU backend for FSDP CPU offload (e.g., "gloo"). Set to None to disable hybrid backend.
    )
    # Hybrid sharding: parameter replica count; dp_shard uses the ranks left by this and SP.
    dp_replicate_size: int = 1

    # Train-actor deterministic mode, gated in deterministic.py. Name kept identical to Megatron's.
    deterministic_mode: bool = False

    # Sequence Parallelism (USP = Ulysses x Ring)
    sequence_parallel_size: int = 1
    # 0=auto: ulysses fills sp; ring = sp // ulysses. Ring degrees > 1 run on
    # torch's experimental (private) ring-attention implementation and require
    # torch >= 2.11 (the CI image's pin).
    ulysses_degree: int = 0

    # YAML bookkeeping
    config: str | None = None


def parse_fsdp_cli(extra_args_provider=None):
    parser = argparse.ArgumentParser("FSDP SFT Training (miles)")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    for f in dataclasses.fields(FSDPArgs):
        if f.name == "config":
            continue

        # Handle union types like int | None, str | None, etc.
        if hasattr(f.type, "__args__"):  # Check if it's a Union type
            # For T | None, use T as the type
            non_none_types = [t for t in f.type.__args__ if t is not type(None)]
            arg_type = non_none_types[0] if non_none_types else str
        else:
            arg_type = f.type

        if arg_type is bool:
            parser.add_argument(f"--{f.name.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=arg_type, default=f.default)

    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    args = parser.parse_args()
    return args


def resolve_sp_degrees(args) -> tuple[int, int]:
    """(ulysses, ring) degrees; ``ulysses_degree=0`` means ulysses fills sp."""
    ulysses_degree = args.ulysses_degree or args.sequence_parallel_size
    return ulysses_degree, args.sequence_parallel_size // ulysses_degree


def resolve_attention_module(args):
    """The ``models/<backend>/attention.py`` module driving this run."""
    from miles.utils.misc import load_function

    from .model_backend import MilesModelBackend

    if issubclass(load_function(args.model_backend_path), MilesModelBackend):
        package = load_function(args.train_pipeline_config_path).model_package
        return importlib.import_module(f"{package}.attention")

    from .models.diffusers import attention

    return attention


def validate_attention_backend(args) -> None:
    """Reject an --fsdp-attention-backend word this run's model backend has no kernel for.

    The words are sglang-diffusion's ``AttentionBackendEnum`` names: miles and SGL-D are
    one system, and a kernel SGL-D cannot serve is not one worth training with.
    ``torch_{math,flash,efficient}_sdpa`` narrow SGL-D's ``torch_sdpa`` to one kernel of
    that dispatcher (rollout has no backward and lets it choose; training pins the kernel
    whose backward it trusts), and SGL-D's bare ``fa`` is not accepted -- the train side
    has to name a generation. Each model backend spells the words for its own kernel
    library in ``models/<backend>/attention.py``.
    """
    if args.fsdp_attention_backend is None:
        return
    table = resolve_attention_module(args).MILES_TO_KERNEL
    name = args.fsdp_attention_backend.strip().lower()
    if name not in table:
        raise ValueError(
            f"--fsdp-attention-backend {args.fsdp_attention_backend!r} is not a kernel this "
            f"model backend serves; choose from {sorted(table)}."
        )
    # Downstream readers (spelling tables, RING_KERNELS, the deterministic sets) compare exactly.
    args.fsdp_attention_backend = name


def validate_sp_config(world_size, sequence_parallel_size, ulysses_degree=0):
    if sequence_parallel_size < 1:
        raise ValueError(f"sequence_parallel_size must be positive, got {sequence_parallel_size}")
    if ulysses_degree < 0:
        raise ValueError(f"ulysses_degree must be non-negative, got {ulysses_degree}")
    resolved_ulysses_degree = ulysses_degree or sequence_parallel_size
    if sequence_parallel_size % resolved_ulysses_degree:
        raise ValueError(
            f"sequence_parallel_size({sequence_parallel_size}) is not divisible by "
            f"ulysses_degree({resolved_ulysses_degree})"
        )
    if world_size % sequence_parallel_size:
        raise ValueError(
            f"world_size({world_size}) is not divisible by sequence_parallel_size({sequence_parallel_size})"
        )


def validate_hybrid_shard_args(args) -> None:
    """Fail fast on a dp_replicate_size the world size or SP degree cannot honor."""
    world_size = args.actor_num_gpus_per_node * args.actor_num_nodes
    validate_sp_config(world_size, args.sequence_parallel_size, args.ulysses_degree)
    if args.dp_replicate_size < 1:
        raise ValueError(f"dp_replicate_size must be at least 1, got {args.dp_replicate_size}")
    if world_size % (args.dp_replicate_size * args.sequence_parallel_size):
        raise ValueError(
            f"world_size({world_size}) is not divisible by dp_replicate_size({args.dp_replicate_size}) "
            f"* sequence_parallel_size({args.sequence_parallel_size})"
        )


def validate_sp_args(args) -> None:
    """Validate finalized SP topology and ring-kernel arguments on the driver.

    Model-instance constraints such as ``_cp_plan`` availability and attention
    dispatch compatibility are checked later, after the model is loaded.
    """
    from .sequence_parallel.attention import RING_KERNELS

    validate_sp_config(
        args.actor_num_gpus_per_node * args.actor_num_nodes,
        args.sequence_parallel_size,
        args.ulysses_degree,
    )
    if args.sequence_parallel_size == 1:
        return
    _, ring_degree = resolve_sp_degrees(args)
    if ring_degree > 1 and args.fsdp_attention_backend not in RING_KERNELS:
        raise ValueError(
            f"--fsdp-attention-backend {args.fsdp_attention_backend!r} cannot drive ring attention; "
            f"supported: {sorted(k for k in RING_KERNELS if k is not None)}"
        )


def load_fsdp_args(extra_args_provider=None):
    args = parse_fsdp_cli(extra_args_provider)
    if args.config:
        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(args, k):
                setattr(args, k, v)
    return args
