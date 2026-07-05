"""LTX-2 family config: train pipeline config + family validation."""

from __future__ import annotations

import logging
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


@register_train_pipeline_config("ltx")
class LTXTrainPipelineConfig(TrainPipelineConfig):
    """LTX-2.3 video GRPO: unguided velocity forward over ltx_core."""

    needs_timestep_scaling = False
    supports_cfg_training = False
    # Rollout stores σ×1000 in trajectory timesteps; ltx_core AdaLN uses σ∈[0,1].
    sde_timestep_divisor = 1000.0
    rollout_patch_group = "ltx"
    hf_ckpt_name_patterns = ("ltx",)
    model_backend_path = "miles.backends.fsdp_utils.model_backend.LTXModelBackend"

    @classmethod
    def validate_args(cls, args: Namespace) -> None:
        # LTX family defaults for the generic diffusion args.
        if getattr(args, "diffusion_output_num_frames", None) is None:
            args.diffusion_output_num_frames = 25
        if getattr(args, "diffusion_fps", None) is None:
            args.diffusion_fps = 24.0
        if not getattr(args, "diffusion_sde_window_size", 0):
            args.diffusion_sde_window_size = 3
        # --diffusion-sde-type "sde" (the generic default) maps to LTX's CPS dynamics.
        if getattr(args, "diffusion_sde_type", "sde") in (None, "sde"):
            args.diffusion_sde_type = "cps"
        args.diffusion_sde_type = _normalize_ltx_dynamics_type(args.diffusion_sde_type)
        # The SDE step dynamics is set by the rollout (sgl-d SchedulerRLMixin.flow_sde_sampling);
        # LTX GRPO runs CPS, and the train scorer (CpsSdeStepBackend) must replicate it, so the
        # rollout sde_type is pinned to cps.
        if args.diffusion_sde_type != "cps":
            raise NotImplementedError(
                f"LTX GRPO runs CPS rollout dynamics; --diffusion-sde-type must be cps "
                f"(got {args.diffusion_sde_type!r})"
            )
        if getattr(args, "sde_step_backend_path", None) is None:
            args.sde_step_backend_path = "miles.backends.fsdp_utils.sde_step_backend.CpsSdeStepBackend"
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

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,
    ) -> dict:
        # Fixed-length embeds: naive concat (pad_to_len accepted and ignored, see base).
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

    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        timesteps_input: torch.Tensor,
        pos_cond: dict | None,
        neg_cond: dict | None,
        joint_cond: dict | None,
        use_cfg: bool,
        cfg_batching: bool,
        guidance_scale: float,
        true_cfg_scale: float | None,
    ) -> torch.Tensor:
        # LTX trains unguided (supports_cfg_training=False): single velocity pass.
        cond = dict(pos_cond or {})
        if "context" not in cond:
            raise ValueError("LTX train requires denoising_env.pos_cond_kwargs.encoder_hidden_states")
        if "positions" not in cond:
            cond.update(self._build_geometry(latents_input))
        return self.forward_velocity(model, latents_input, timesteps_input, cond)

    def _build_geometry(self, latents_input: torch.Tensor) -> dict:
        """T2V geometry is a pure function of latent shape + request constants (args)."""
        from miles.backends.fsdp_utils.ltx_geometry import build_ltx_t2v_geometry

        args = self.args
        batch_size, num_tokens, latent_dim = latents_input.shape
        return build_ltx_t2v_geometry(
            batch_size=batch_size,
            num_tokens=num_tokens,
            latent_dim=latent_dim,
            height=int(getattr(args, "diffusion_height", 512)),
            width=int(getattr(args, "diffusion_width", 512)),
            num_frames=int(getattr(args, "diffusion_output_num_frames", 25)),
            fps=float(getattr(args, "diffusion_fps", 24.0)),
            device=latents_input.device,
            dtype=latents_input.dtype,
        )

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

        # Trajectory timesteps are σ×1000; ltx_core AdaLN expects σ∈[0,1] and
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
