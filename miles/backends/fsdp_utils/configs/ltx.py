"""LTX-2 family config: train pipeline config + family validation."""

from __future__ import annotations

from argparse import Namespace

import torch

from miles.utils.types import CondKwargs

from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


@register_train_pipeline_config("ltx")
class LTXTrainPipelineConfig(TrainPipelineConfig):
    """LTX-2.3 video GRPO: unguided velocity forward over ltx_core.

    Every tensor of this forward is bit-exact against the sglang-d rollout, verified by
    paired dumps. The dtypes below are what buys that, so treat them as load-bearing.
    """

    supports_cfg_training = False
    # Rollout stores σ×1000 in trajectory timesteps; the CPS SDE path resolves σ linearly.
    sde_timestep_divisor = 1000.0
    hf_ckpt_name_patterns = ("ltx",)
    model_backend_path = "miles.backends.fsdp_utils.model_backend.MilesModelBackend"
    model_package = "miles.backends.fsdp_utils.models.ltx"
    # Audio branch has no optimizer state: we only train the video stream.
    optimizer_state_allowed_missing = ["audio"]
    # forward_velocity derives every element-wise dtype from latents; rollout runs it bf16.
    input_dtype_policy = {"latents": "default", "cond": "default", "timestep": None}

    def process_timestep_as_input(self, timesteps, *, sigmas, num_train_timesteps):
        # ltx_core consumes sigma; the rollout snapshot is exact where t/divisor drifts ULPs.
        return sigmas

    def configure(self, args: Namespace) -> None:
        self._height = args.diffusion_height
        self._width = args.diffusion_width
        self._num_frames = args.diffusion_output_num_frames
        self._fps = args.diffusion_fps

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
        if cond.encoder_attention_mask:
            mask = torch.cat(cond.encoder_attention_mask).to(device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            kwargs["context_mask"] = mask
        if cond.audio_encoder_attention_mask:
            audio_mask = torch.cat(cond.audio_encoder_attention_mask).to(device)
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
            from miles.backends.fsdp_utils.models.ltx.positions import prepare_video_positions

            batch_size, num_tokens, _ = latents_input.shape
            cond["positions"] = prepare_video_positions(
                batch_size=batch_size,
                num_tokens=num_tokens,
                height=self._height,
                width=self._width,
                num_frames=self._num_frames,
                fps=self._fps,
                device=latents_input.device,
                dtype=latents_input.dtype,
            )
        return self.forward_velocity(model, latents_input, timesteps_input, cond)

    def forward_velocity(
        self,
        model: torch.nn.Module,
        latents_input: torch.Tensor,
        sigmas_input: torch.Tensor,
        cond: dict,
    ) -> torch.Tensor:
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.utils import to_denoised

        dtype = latents_input.dtype
        B = latents_input.shape[0]

        # Rounding σ to bf16 before the sinusoid costs ~2e-3 rel, re-injected by every AdaLN block.
        sigma_unit = sigmas_input.to(dtype)
        per_token_t = sigma_unit.view(B, 1)

        video_modality = Modality(
            enabled=True,
            latent=latents_input,
            sigma=sigmas_input.float().reshape(B),
            timesteps=sigmas_input.float().view(B, 1),
            positions=cond["positions"].to(dtype),
            context=cond["context"].to(dtype),
            context_mask=None,
        )
        # Compute dtype comes from the trainer's ambient autocast around compute_noise_pred.
        velocity, _ = model(video=video_modality, audio=None, perturbations=None)

        # Keep the original fp32 denoised reconstruction path: although this is
        # algebraically an identity for T2V, strict e2e metrics depend on its rounding.
        x0_pred = to_denoised(latents_input, velocity, per_token_t.unsqueeze(-1)).float()
        sigma_safe = torch.clamp(sigma_unit, min=1e-8).view(B, 1, 1)
        return ((latents_input.float() - x0_pred) / sigma_safe).to(dtype)

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
