import argparse
import dataclasses
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
    warmup_ratio: float = 0.03

    attn_implementation: str = "flash_attention_2"

    # DiT attention backend, passed to diffusers set_attention_backend (e.g.
    # "flash", "sage", "native"). None keeps the diffusers default.
    fsdp_attention_backend: str | None = None

    # Force a deterministic backward on diffusers' flash backends (forward is
    # already deterministic). diffusers never passes deterministic=True, so this
    # monkey-patches the flash_attn entry points. Keeps the flash forward (and so
    # train/rollout consistency) while making the training backward reproducible.
    fsdp_attention_deterministic: bool = False

    # Logging
    wandb_project: str = "miles-fsdp"
    wandb_run_name: str | None = None

    # Precision
    gradient_checkpointing: bool = False
    fp16: bool = False

    # FSDP configuration
    fsdp_state_dict_cpu_offload: bool = True  # If True, offload full state dict to CPU during collection.
    fsdp_cpu_offload: bool = (
        False  # If True, offload parameters, gradients, and optimizer states to CPU (optimizer runs on CPU)
    )
    fsdp_cpu_backend: str | None = (
        "gloo"  # CPU backend for FSDP CPU offload (e.g., "gloo"). Set to None to disable hybrid backend.
    )

    deterministic_mode: bool = False  # This name must be the same as Megatron's

    # Context Parallelism
    context_parallel_size: int = 1  # Context Parallelism size

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


# diffusers' flash backends (FA2 `flash`, FA3 `_flash_3`) dispatch through these
# module globals; the FA3 custom op also reads them at call time.
_FLASH_ATTN_DISPATCH_FNS = (
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_3_func",
    "flash_attn_3_varlen_func",
)


def deterministic_capable_flash_fns():
    """diffusers flash entry points whose signature accepts a `deterministic` arg."""
    import inspect

    import diffusers.models.attention_dispatch as ad

    out = []
    for name in _FLASH_ATTN_DISPATCH_FNS:
        fn = getattr(ad, name, None)
        if fn is None:
            continue
        try:
            if "deterministic" in inspect.signature(fn).parameters:
                out.append(name)
        except (TypeError, ValueError):
            continue
    return out


def validate_attention_args(args):
    """Fail fast (before any actor is launched) on attention misconfiguration."""
    if not getattr(args, "fsdp_attention_deterministic", False):
        return
    # deterministic only changes the flash backward; on any other backend the
    # switch would silently do nothing, which is exactly the dangerous case.
    backend = args.fsdp_attention_backend
    if backend is None or "flash" not in backend.lower():
        raise ValueError(
            "--fsdp-attention-deterministic only affects flash backends; set "
            "--fsdp-attention-backend to a flash variant (e.g. flash, _flash_3)."
        )
    if not deterministic_capable_flash_fns():
        raise RuntimeError(
            "--fsdp-attention-deterministic is set but no diffusers flash entry "
            "point exposes a deterministic argument (is flash-attn installed and "
            "recent enough?)."
        )


def load_fsdp_args(extra_args_provider=None):
    args = parse_fsdp_cli(extra_args_provider)
    if args.config:
        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(args, k):
                setattr(args, k, v)
    validate_attention_args(args)
    return args
