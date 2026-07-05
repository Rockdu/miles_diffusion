"""Training-side pipeline config for diffusion models.

Mirrors the spirit of sglang-d's PipelineConfig but only contains the
model-specific logic needed for the GRPO training loop:
  - How to prepare conditioning kwargs from DenoisingEnv
  - How to unpack trajectories
  - How to apply CFG (with or without rescale)
  - How to expand conditioning for timestep batching

Each model (QwenImage, SD3, Flux, ...) subclasses TrainPipelineConfig
and overrides the relevant methods.

Backends (ModelBackend / SdeStepBackend) are selected via CLI paths
(``--model-backend-path`` / ``--sde-step-backend-path``); the family config
only provides their defaults.
"""

from __future__ import annotations

import abc
import os

import torch
from miles.utils.types import CondKwargs, DiTTrajectory

def _pack_cond_for_joint_cfg(pos: dict, neg: dict) -> dict:
    out: dict = {}
    for key, value in pos.items():
        if isinstance(value, torch.Tensor):
            out[key] = torch.cat([value, neg[key]], dim=0)
        elif isinstance(value, list):
            out[key] = value + neg[key]
        else:
            out[key] = value
    return out


_REGISTRY: dict[str, type[TrainPipelineConfig]] = {}


def register_train_pipeline_config(family: str):
    """Decorator: register a TrainPipelineConfig subclass under a family key (``ltx``, ``sd3``, ...)."""

    def wrapper(cls):
        _REGISTRY[family.lower()] = cls
        return cls

    return wrapper


def _populate_registry() -> None:
    # Import every config module here (lazily — they import this module back);
    # registration is an import side effect.
    import importlib
    import pkgutil

    package = importlib.import_module("miles.backends.fsdp_utils.configs")
    for module_info in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package.__name__}.{module_info.name}")


def resolve_diffusion_model_family(model_ref: str) -> str:
    """Map a model reference to a family key by the refs each family declares."""
    override = os.environ.get("MILES_DIFFUSION_MODEL_FAMILY")
    if override:
        return override.strip().lower()

    _populate_registry()
    ref = str(model_ref).lower()
    for family, config_cls in _REGISTRY.items():
        if any(pattern in ref for pattern in config_cls.hf_ckpt_name_patterns):
            return family
    raise ValueError(
        f"Cannot resolve diffusion model family for '{model_ref}' "
        f"(known families: {list(_REGISTRY)}). "
        "Set MILES_DIFFUSION_MODEL_FAMILY to override."
    )


def get_train_pipeline_config_cls(family: str) -> type[TrainPipelineConfig]:
    """The TrainPipelineConfig class registered for a resolved family key."""
    cls = _REGISTRY.get(family.lower())
    if cls is None:
        raise ValueError(
            f"No TrainPipelineConfig registered for family '{family}'. " f"Known families: {list(_REGISTRY.keys())}"
        )
    return cls


class TrainPipelineConfig(abc.ABC):
    """Base class. Subclass per model family."""

    lora_target_modules: list[str] = ["to_q", "to_k", "to_v", "to_out.0"]
    needs_timestep_scaling: bool = True
    optimizer_state_allowed_missing: list[str] = []
    sde_timestep_divisor: float = 1.0
    supports_cfg_training: bool = True
    # Case-insensitive substrings matched against the HF checkpoint name
    # (--diffusion-model / --hf-checkpoint); could grow into regexes later.
    hf_ckpt_name_patterns: tuple[str, ...] = ()
    # Env flag enabling this family's rollout parity patches in the engine (None = none).
    rollout_patch_env: str | None = None
    # Default backend paths (miles custom-function style); CLI args override.
    model_backend_path: str = "miles.backends.fsdp_utils.model_backend.DiffusersModelBackend"
    sde_step_backend_path: str = "miles.backends.fsdp_utils.sde_step_backend.DiffusersSdeStepBackend"

    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        pos_cond: dict,
        neg_cond: dict | None,
        use_cfg: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
        fsdp_cfg_batching: bool,
    ) -> torch.Tensor:
        """Default diffusers forward with CFG; families override (e.g. LTX: unguided velocity)."""

        def _forward(cond: dict) -> torch.Tensor:
            return model(
                hidden_states=latents_input,
                timestep=timesteps_input,
                return_dict=False,
                **cond,
            )[0]

        if not use_cfg:
            return _forward(pos_cond)
        if fsdp_cfg_batching:
            joint_cond = _pack_cond_for_joint_cfg(pos_cond, neg_cond)
            joint_out = model(
                hidden_states=torch.cat([latents_input, latents_input], dim=0),
                timestep=torch.cat([timesteps_input, timesteps_input], dim=0),
                return_dict=False,
                **joint_cond,
            )[0]
            noise_pred_pos, noise_pred_neg = joint_out.chunk(2, dim=0)
        else:
            noise_pred_pos = _forward(pos_cond)
            noise_pred_neg = _forward(neg_cond)
        return self.cfg_combine(
            noise_pred_pos,
            noise_pred_neg,
            guidance_scale,
            true_cfg_scale=true_cfg_scale,
        )

    def should_use_cfg(self, args) -> bool:
        if not self.supports_cfg_training:
            return False
        guidance_scale = args.diffusion_guidance_scale
        true_cfg_scale = args.diffusion_true_cfg_scale
        cfg_scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        return cfg_scale > 0

    def prepare_trajectory(
        self,
        traj: DiTTrajectory,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack trajectory into (latents, next_latents, timesteps).

        Default handles the common (T+1, ...) layout. Override for models
        with different trajectory formats.
        """
        all_latents = traj.latents.to(device, dtype=torch.float32)
        latents = all_latents[:-1]
        next_latents = all_latents[1:]
        timesteps = traj.timesteps.to(device, dtype=torch.float32)
        return latents, next_latents, timesteps

    @abc.abstractmethod
    def prepare_cond_kwargs(
        self,
        cond: CondKwargs | None,
        device: torch.device,
    ) -> dict:
        """Convert CondKwargs to model-specific forward() kwargs."""

    def build_train_cond_kwargs(
        self,
        cond: CondKwargs | None,
        *,
        latents: torch.Tensor,
        args,
        device: torch.device,
    ) -> dict:
        """Build per-sample cond for training; default reuses rollout embeds only."""
        return self.prepare_cond_kwargs(cond, device)

    def expand_cond_for_timestep_batch(
        self,
        cond_kwargs: dict,
        batch_size: int,
    ) -> dict:
        """Expand per-sample conditioning to a timestep batch."""
        out = {}
        for k, v in cond_kwargs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.expand(batch_size, *v.shape[1:]) if v.shape[0] == 1 else v
            elif isinstance(v, list):
                out[k] = v * batch_size if len(v) == 1 else v
            else:
                out[k] = v
        return out

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
    ) -> dict:
        """Stack a list of per-sample cond_kwargs (output of prepare_cond_kwargs)
        into a single batched dict suitable for one DiT forward over M samples.

        Model-specific because variable-length text embeds need padding + mask.
        Default: naive concat along batch dim, only valid when shapes match.
        """
        raise NotImplementedError(
            "Must implement collate_cond_for_sample_batch to enable --micro-batch-size-sample in fsdp training"
        )

    @abc.abstractmethod
    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """Apply classifier-free guidance. Model-specific (e.g. rescale or not)."""

    @abc.abstractmethod
    def preprocess_model_before_fsdp(self, model: torch.nn.Module) -> None:
        """Preprocess the model before FSDP."""
