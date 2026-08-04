"""Wan2.2 training pipeline config."""

from __future__ import annotations

import torch
from miles.utils.types import CondKwargs

from ..precision import ModuleSel, PrecisionSpec, Rule
from .train_pipeline_config import TrainPipelineConfig, register_train_pipeline_config


@register_train_pipeline_config("wan2_2")
class Wan2_2TrainPipelineConfig(TrainPipelineConfig):
    hf_ckpt_name_patterns = ("wan2.2", "wan-2.2")
    # Rollout (sglang-d) keeps every FP32LayerNorm's affine params resident and
    # consumed in fp32 (verified on the Wan2.2 full40 dump: rollout norm2
    # weight/bias are float32 in the forward while the FSDP default policy
    # gathered them as bf16). Pin the gather dtype so the training matmul
    # consumes the same weight dtype; the resident master is already fp32 via
    # the run-level --fsdp-master-dtype. Affine-less FP32LayerNorms
    # (norm1/norm3/norm_out) carry no tensors and compile to nothing.
    precision_spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="FP32LayerNorm"), gather="fp32"),))
    # Boundary inputs, verified against paired sglang-d dumps: rollout casts
    # the latent to the forward dtype at the boundary but passes the T5 text
    # embeds through in fp32 -- the first context linear consumes an fp32
    # input with bf16 weights -- and keeps the raw timestep fp32.
    # Known residue this policy cannot remove: diffusers ties temb to the text
    # embeds' dtype (WanTimeTextImageEmbedding: `.type_as(encoder_hidden_states)`),
    # so with fp32 cond the act_fn/time_proj input records read fp32 where
    # rollout reads bf16. autocast re-quantizes at the time_proj matmul, so the
    # effective compute dtype still matches everywhere; only the SiLU on the
    # 1x1536 time vector runs at higher precision than rollout. The alternative
    # (cond cast to forward dtype) would misalign the 512x4096 text embeds
    # feeding every block's cross attention instead.
    input_dtype_policy = {"latents": "default", "cond": None, "timestep": "fp32"}
    # High-noise expert ("transformer") handles t >= boundary, low-noise expert
    # ("transformer_2") the rest.
    boundary_ratio = 0.875
    # Wan DiT expects raw scheduler timesteps (0..num_train_timesteps), no /1000 scaling.
    needs_timestep_scaling = False

    def component_for_timestep(self, timestep: float, num_train_timesteps: int) -> str:
        if timestep >= self.boundary_ratio * num_train_timesteps:
            return "transformer"
        return "transformer_2"

    def select_guidance_scale(
        self,
        timestep: float,
        num_train_timesteps: int,
        guidance_scale: float,
        guidance_scale_2: float | None,
    ) -> float:
        if timestep >= self.boundary_ratio * num_train_timesteps:
            return guidance_scale
        # Rollout backend (sglang-diffusion) uses batch.guidance_scale_2 for low-noise steps with NO fallback;
        # While high-noise and low-noise can be different;
        # A misalignment of guidance_scale_2 between training and rollout would hurt training significantly, so we require it to be set explicitly.
        assert guidance_scale_2 is not None, (
            "Wan2.2 low-noise steps require --diffusion-guidance-scale-2 "
            "(rollout already denoises them with guidance_scale_2)."
        )
        return guidance_scale_2

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None or not cond.encoder_hidden_states:
            return {}
        enc = torch.cat(cond.encoder_hidden_states).to(device)
        if enc.ndim == 2:
            enc = enc.unsqueeze(0)
        return {"encoder_hidden_states": enc}

    def collate_cond_for_sample_batch(
        self,
        per_sample_cond_kwargs: list[dict],
        device: torch.device,
        pad_to_len: int | None = None,  # accepted for interface parity (PR #10); Wan2.2 concats fixed-length T5 embeds
    ) -> dict:
        encs = [kw["encoder_hidden_states"] for kw in per_sample_cond_kwargs]
        return {"encoder_hidden_states": torch.cat(encs, dim=0).to(device)}

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        scale = true_cfg_scale if true_cfg_scale is not None else guidance_scale
        return noise_pred_neg + scale * (noise_pred_pos - noise_pred_neg)
