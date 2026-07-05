"""Model backends for diffusion GRPO training.

Owns the trainable model lifecycle: loading (with its scheduler counterpart),
gradient checkpointing and FSDP wrap classes. Selected via
``--model-backend-path``. The denoising forward + CFG live on
``TrainPipelineConfig``; SDE math lives in ``SdeStepBackend``.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch
from diffusers import DiffusionPipeline

if TYPE_CHECKING:
    from miles.backends.fsdp_utils.configs.train_pipeline_config import TrainPipelineConfig


class ModelBackend(abc.ABC):
    """Load / forward / grad-ckpt / FSDP-wrap for one model family."""

    fsdp_wrap_classes: list[str] | None = None

    def __init__(self, config: TrainPipelineConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def load_model_and_scheduler(
        self,
        args,
        init_context_factory,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[torch.nn.Module, object]: ...

    def apply_gradient_checkpointing(self, model: torch.nn.Module, args) -> None:
        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()

    def get_fsdp_wrap_classes(self) -> list[str] | None:
        return self.fsdp_wrap_classes


class DiffusersModelBackend(ModelBackend):
    """Default path: diffusers DiT loaded via DiffusionPipeline."""

    def load_model_and_scheduler(
        self,
        args,
        init_context_factory,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[torch.nn.Module, object]:
        with init_context_factory():
            pipeline = DiffusionPipeline.from_pretrained(
                args.hf_checkpoint,
                torch_dtype=master_dtype,
                trust_remote_code=True,
                text_encoder=None,
                vae=None,
                tokenizer=None,
            )
            model = pipeline.transformer
            scheduler = pipeline.scheduler
            del pipeline
        return model, scheduler


class LTXModelBackend(ModelBackend):
    """LTX-2.3: native ltx_core modeling behind the diffusers interface protocol.

    ``LTX2TransformerModeling`` mirrors the diffusers ModelMixin surface
    (``from_pretrained`` / ``_no_split_modules`` / ``enable_gradient_checkpointing``),
    so loading, grad-ckpt and FSDP wrapping follow the same path as diffusers models.
    """

    def load_model_and_scheduler(
        self,
        args,
        init_context_factory,
        *,
        master_dtype: torch.dtype | None = None,
    ) -> tuple[torch.nn.Module, object]:
        from miles.backends.fsdp_utils.models.ltx2 import (
            LTX2TransformerModeling,
            build_ltx_train_scheduler,
        )

        master_dtype_name = getattr(args, "fsdp_master_dtype", "bf16")
        resolved_dtype = (
            master_dtype
            or {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }[master_dtype_name]
        )

        model = LTX2TransformerModeling.from_pretrained(
            args.diffusion_model,
            torch_dtype=resolved_dtype,
            weights_path=getattr(args, "sglang_transformer_weights_path", None),
        )
        self._loaded_model = model
        return model, build_ltx_train_scheduler(args)

    def get_fsdp_wrap_classes(self) -> list[str] | None:
        # Diffusers protocol: the loaded model declares its own no-split modules.
        model = getattr(self, "_loaded_model", None)
        return getattr(model, "_no_split_modules", None)
