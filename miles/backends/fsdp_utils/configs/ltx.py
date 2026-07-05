"""LTX-2 family config: train pipeline config + rollout engine hooks."""

from __future__ import annotations

import logging
import os
from argparse import Namespace

import torch

from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config

logger = logging.getLogger(__name__)

def _normalize_ltx_dynamics_type(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    allowed = ("flow_sde", "cps", "ode", "dance_sde")
    if key not in allowed:
        raise ValueError(f"Unknown ltx dynamics_type {name!r}; expected one of {allowed}")
    return key


def patch_rollout_engine_env_vars(env_vars: dict[str, str], args) -> None:
    """Add LTX-specific env vars for Ray rollout engine workers."""
    if getattr(args, "diffusion_model_family", None) != "ltx":
        return

    from miles.backends.sglang_diffusion_utils.monkey_patches import LTX_ROLLOUT_PATCHES_ENV

    if os.environ.get(LTX_ROLLOUT_PATCHES_ENV):
        env_vars[LTX_ROLLOUT_PATCHES_ENV] = os.environ[LTX_ROLLOUT_PATCHES_ENV]




def validate_args(args: Namespace) -> None:
    # LTX family defaults for the generic diffusion args (parity with the old --ltx-* defaults).
    if getattr(args, "diffusion_output_num_frames", None) is None:
        args.diffusion_output_num_frames = 25
    if getattr(args, "diffusion_fps", None) is None:
        args.diffusion_fps = 24.0
    if not getattr(args, "diffusion_sde_window_size", 0):
        args.diffusion_sde_window_size = 3
    # --diffusion-sde-type was never honored for LTX before (always CPS); map the generic default.
    if getattr(args, "diffusion_sde_type", "sde") in (None, "sde"):
        args.diffusion_sde_type = "cps"
    args.diffusion_sde_type = _normalize_ltx_dynamics_type(args.diffusion_sde_type)
    if args.diffusion_sde_type == "dance_sde":
        raise NotImplementedError("dance_sde rollout is not implemented in sglang-d flow_sde_sampling yet.")
    ltx_gs = float(getattr(args, "diffusion_guidance_scale", 1.0))
    if ltx_gs != 1.0:
        logger.warning(
            "LTX rollout/train alignment expects --diffusion-guidance-scale 1.0 "
            "(no CFG); using %s may break log_prob parity.",
            ltx_gs,
        )
    if getattr(args, "fsdp_master_dtype", "fp32") == "fp32":
        logger.warning(
            "LTX with fsdp_master_dtype=fp32 is unlikely to fit "
            "on small GPU counts; consider --fsdp-master-dtype bf16."
        )


# --- FSDP train pipeline config ---


@register_train_pipeline_config("ltx")
class LTXTrainPipelineConfig(TrainPipelineConfig):
    """Training-side adapter for LTX-2.3 video DiT."""

    model_backend_path = "miles.backends.fsdp_utils.model_backend.LTXModelBackend"
    sde_step_backend_path = "miles.backends.fsdp_utils.sde_step_backend.LTXSdeStepBackend"
    needs_timestep_scaling = False
    supports_cfg_training = False
    # Rollout stores σ×1000 in dit_trajectory.timesteps; CPS uses scheduler σ∈[0,1].
    sde_timestep_divisor = 1000.0
    rollout_patch_env = "MILES_APPLY_LTX_ROLLOUT_PATCHES"
    hf_ckpt_name_patterns = ("ltx",)

    lora_target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "net.0.proj",
        "net.2",
    ]

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}
        kwargs: dict = {}
        if cond.encoder_hidden_states:
            ctx = torch.cat(cond.encoder_hidden_states).to(device)
            if ctx.ndim == 2:
                ctx = ctx.unsqueeze(0)
            kwargs["context"] = ctx
        if cond.audio_encoder_hidden_states:
            audio_ctx = torch.cat(cond.audio_encoder_hidden_states).to(device)
            if audio_ctx.ndim == 2:
                audio_ctx = audio_ctx.unsqueeze(0)
            kwargs["audio_context"] = audio_ctx
        if cond.encoder_attention_mask is not None:
            mask = cond.encoder_attention_mask.to(device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            kwargs["context_mask"] = mask
        if cond.audio_encoder_attention_mask is not None:
            audio_mask = cond.audio_encoder_attention_mask.to(device)
            if audio_mask.ndim == 1:
                audio_mask = audio_mask.unsqueeze(0)
            kwargs["audio_context_mask"] = audio_mask
        return kwargs

    def build_train_cond_kwargs(
        self,
        cond: CondKwargs | None,
        *,
        latents: torch.Tensor,
        args,
        device: torch.device,
    ) -> dict:
        """Merge rollout text embeds with locally rebuilt T2V geometry."""
        from miles.backends.fsdp_utils.ltx_geometry import build_ltx_t2v_geometry

        kwargs = self.prepare_cond_kwargs(cond, device)
        if "context" not in kwargs:
            raise ValueError("LTX train requires denoising_env.pos_cond_kwargs.encoder_hidden_states")

        ref = latents[0] if latents.ndim >= 2 else latents
        if ref.ndim == 2:
            batch_size, num_tokens, latent_dim = 1, ref.shape[0], ref.shape[1]
        else:
            batch_size, num_tokens, latent_dim = ref.shape[0], ref.shape[1], ref.shape[2]

        geom = build_ltx_t2v_geometry(
            batch_size=batch_size,
            num_tokens=num_tokens,
            latent_dim=latent_dim,
            height=int(getattr(args, "diffusion_height", 512)),
            width=int(getattr(args, "diffusion_width", 512)),
            num_frames=int(getattr(args, "diffusion_output_num_frames", 25)),
            fps=float(getattr(args, "diffusion_fps", 24.0)),
            device=device,
            dtype=ref.dtype,
        )
        kwargs.update(geom)
        return kwargs

    def build_sde_extra(
        self,
        scheduler,
        grids: dict,
        sample_indices: torch.Tensor,
        tstep_indices: torch.Tensor,
        args,
    ) -> dict | None:
        window = grids.get("sde_step_indices_window")
        if window is None:
            return None
        idx = window[sample_indices][:, tstep_indices].reshape(-1).long()
        return {
            "sde_step_indices": idx,
            "sigmas": scheduler.sigmas,
            "dynamics_type": getattr(args, "diffusion_sde_type", "cps"),
            "sigma_min_override": getattr(args, "diffusion_sigma_min", None),
        }

    def expand_cond_for_timestep_batch(self, cond_kwargs: dict, batch_size: int) -> dict:
        out: dict = {}
        for k, v in cond_kwargs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.expand(batch_size, *v.shape[1:]) if v.shape[0] == 1 else v
            else:
                out[k] = v
        return out

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
    ) -> dict:
        out: dict = {}
        for key in per_sample_cond_kwargs[0]:
            values = [kw[key] for kw in per_sample_cond_kwargs if key in kw]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                out[key] = torch.cat(values, dim=0).to(device)
            else:
                out[key] = values
        return out

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        if scale == 1.0:
            return noise_pred_pos
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)

    def preprocess_model_before_fsdp(self, model: torch.nn.Module) -> None:
        return None

    @staticmethod
    def _modality_timesteps_for_adaln(per_token_t: torch.Tensor) -> torch.Tensor:
        """Collapse per-token sigma to batch-global AdaLN input when uniform.

        sglang rollout builds temb with shape ``[B, 1, D]`` (scheduler timestep
        is batch-scalar expanded only for masking). ltx_core defaults to
        ``[B, T, D]`` when ``Modality.timesteps`` has length T, which diverges
        in AdaLN even when every active token shares the same sigma.
        """
        if per_token_t.ndim != 2 or per_token_t.shape[1] == 1:
            return per_token_t
        ref = per_token_t[:, :1]
        if torch.allclose(per_token_t, ref.expand_as(per_token_t), rtol=0.0, atol=0.0):
            return ref
        return per_token_t

    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        pos_cond: dict,
        neg_cond: dict | None = None,
        use_cfg: bool = False,
        guidance_scale: float = 1.0,
        true_cfg_scale: float | None = None,
        fsdp_cfg_batching: bool = False,
    ) -> torch.Tensor:
        # LTX trains unguided (supports_cfg_training=False): single velocity pass.
        return self.forward_velocity(model, latents_input, timesteps_input, pos_cond)

    def forward_velocity(
        self,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        cond: dict,
    ) -> torch.Tensor:
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.utils import to_denoised

        device = latents_input.device
        dtype = latents_input.dtype
        B = latents_input.shape[0]

        # dit_trajectory.timesteps are σ×1000; ltx_core AdaLN expects σ∈[0,1] and
        # multiplies by timestep_scale_multiplier (1000) internally.
        sigma_scaled = timesteps_input.to(latents_input.dtype)
        sigma_unit = sigma_scaled / float(self.sde_timestep_divisor)
        denoise_mask = cond["denoise_mask"].to(device)
        denoise_mask_2d = denoise_mask.squeeze(-1) if denoise_mask.ndim == 3 else denoise_mask
        denoise_mask_float = denoise_mask_2d.float()

        per_token_t = (sigma_unit.view(B, 1) * denoise_mask_2d).to(dtype)
        adaln_timesteps = self._modality_timesteps_for_adaln(per_token_t)

        video_modality = Modality(
            enabled=True,
            latent=latents_input,
            sigma=sigma_unit.reshape(B),
            timesteps=adaln_timesteps,
            positions=cond["positions"].to(dtype),
            context=cond["context"].to(dtype),
            context_mask=None,
        )
        with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype):
            velocity, _ = model(video=video_modality, audio=None, perturbations=None)

        per_token_t_3d = per_token_t.unsqueeze(-1) if per_token_t.ndim == 2 else per_token_t
        x0_pred = to_denoised(latents_input, velocity, per_token_t_3d).float()

        clean_latent = cond["clean_latent"].to(device).float()
        denoise_mask_3d = denoise_mask_float.unsqueeze(-1) if denoise_mask_float.ndim == 2 else denoise_mask_float
        x0_pred = x0_pred * denoise_mask_3d + clean_latent * (1.0 - denoise_mask_3d)

        sigma_safe = torch.clamp(sigma_unit, min=1e-8).view(B, 1, 1)
        velocity_for_sde = (latents_input.float() - x0_pred) / sigma_safe
        return velocity_for_sde.to(dtype)

    def sde_step(
        self,
        scheduler,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        prev_sample: torch.Tensor,
        *,
        noise_level: float,
        extra: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from miles.utils.sde_log_prob import sde_step_with_logprob

        if extra is None or "sigmas" not in extra or "sde_step_indices" not in extra:
            raise ValueError("LTXTrainPipelineConfig.sde_step requires extra={'sigmas','sde_step_indices',...}")
        sigmas = extra["sigmas"].to(sample.device).float()
        step_indices = extra["sde_step_indices"].to(sample.device).long()
        sigma_view = timesteps.float()
        sigma_next = sigmas[torch.clamp(step_indices + 1, max=len(sigmas) - 1)]

        dynamics_type = _normalize_ltx_dynamics_type(extra.get("dynamics_type", "cps"))
        if dynamics_type != "cps":
            raise NotImplementedError(
                f"LTXTrainPipelineConfig.sde_step supports dynamics_type='cps' only " f"(got {dynamics_type!r})."
            )

        prev, log_prob, prev_mean, std_dev_t = sde_step_with_logprob(
            None,
            noise_pred.float(),
            sigma_view,
            sample.float(),
            prev_sample.float(),
            noise_level=noise_level,
            sde_type="cps",
            sigma=sigma_view,
            sigma_prev=sigma_next,
        )
        if std_dev_t.ndim > 1:
            std_dev_t = std_dev_t.mean(dim=tuple(range(1, std_dev_t.ndim)))
        return prev, log_prob, prev_mean, std_dev_t
