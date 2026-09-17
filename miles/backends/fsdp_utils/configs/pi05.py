"""pi0.5 family config: VLA SFT over the native pi0.5 package."""

from __future__ import annotations

import torch

from miles.utils.types import CondKwargs

from .train_pipeline_config import register_train_pipeline_config, TrainPipelineConfig


@register_train_pipeline_config("pi05")
class Pi05TrainPipelineConfig(TrainPipelineConfig):
    """pi0.5 behavior cloning: flow-matching velocity over a joint-attention VLA.

    This family is SFT only. There is no rollout engine, no guidance, and no
    denoising trajectory to replay, so the CFG and sampling hooks the diffusion
    families rely on are unreachable here rather than merely unused.
    """

    supports_cfg_training = False
    hf_ckpt_name_patterns = ("pi05", "pi0.5")
    model_backend_path = "miles.backends.fsdp_utils.model_backend.MilesModelBackend"
    model_package = "miles.backends.fsdp_utils.models.pi05"
    # Both experts' residual math is anchored on the incoming embeds, and the
    # adaRMS modulation is fp32 inside GemmaRMSNorm already.
    input_dtype_policy = {"latents": "default", "cond": "default", "timestep": None}
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        """Token ids and image embeds, following Cosmos3's token-level conditioning.

        pi0.5 has no separate text encoder: the prompt is PaliGemma tokens, and
        ``encoder_hidden_states`` carries the vision tower's image tokens.
        """
        if cond is None:
            return {}
        kwargs: dict = {}
        if cond.text_ids is not None:
            kwargs["tokens_BT"] = cond.text_ids.to(device)
        if cond.text_mask is not None:
            kwargs["token_mask_BT"] = cond.text_mask.to(device)
        if cond.encoder_hidden_states:
            image_embeds = torch.cat(cond.encoder_hidden_states).to(device)
            if image_embeds.ndim == 2:
                image_embeds = image_embeds.unsqueeze(0)
            kwargs["image_embeds_BIP"] = image_embeds
            kwargs["image_mask_BI"] = torch.ones(
                image_embeds.shape[:2], dtype=torch.bool, device=device
            )
        return kwargs

    def cfg_combine(
        self,
        noise_pred_pos: torch.Tensor,
        noise_pred_neg: torch.Tensor,
        guidance_scale: float,
        true_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "pi0.5 trains without classifier-free guidance; there is no negative "
            "branch to combine. Run it with --no-cfg-training."
        )
