"""Model backend: owns model-side behavior for the FSDP trainer.

Selected via ``--model-backend-path`` (miles custom-function style); the
family config declares the default. Three concerns, all properties of the
concrete modeling rather than of the training loop:

  - ``load_models_and_scheduler``: checkpoint -> ``({component: model}, scheduler)``
  - ``enable_gradient_checkpointing``: how this model turns on grad ckpt
  - ``fsdp_no_split_modules``: which block classes FSDP wraps

Defaults implement the diffusers protocol (see ``models/__init__.py``); a
native model overrides methods here instead of retrofitting its instances.
"""

from __future__ import annotations

import abc
from typing import Any

import torch
from diffusers import DiffusionPipeline


class ModelBackend(abc.ABC):
    def __init__(self, train_pipeline_config):
        self.config = train_pipeline_config

    @abc.abstractmethod
    def load_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        """Return ``({component: model}, scheduler)`` on CPU."""

    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        """Turn on grad checkpointing; default = the diffusers protocol method."""
        model.enable_gradient_checkpointing()

    def fsdp_no_split_modules(self, model: torch.nn.Module) -> list[str]:
        """Block class names FSDP wraps; default = the model's own declaration."""
        return model._no_split_modules

    def set_attention_backend(self, model: torch.nn.Module, backend: str) -> None:
        """Select the DiT attention backend; default = the diffusers protocol method."""
        model.set_attention_backend(backend)


class DiffusersModelBackend(ModelBackend):
    """Load trainable components from a diffusers pipeline checkpoint."""

    def load_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        pipeline = DiffusionPipeline.from_pretrained(
            args.hf_checkpoint,
            torch_dtype=master_dtype,
            trust_remote_code=True,
            text_encoder=None,
            vae=None,
            tokenizer=None,
        )
        raw_models: dict[str, torch.nn.Module] = {}
        for component in args.update_weight_target_modules:
            sub_model = getattr(pipeline, component, None)
            if sub_model is None:
                raise ValueError(
                    f"--update-weight-target-module: pipeline {args.hf_checkpoint} " f"has no component '{component}'"
                )
            raw_models[component] = sub_model
        scheduler = pipeline.scheduler
        del pipeline
        return raw_models, scheduler


class LTXModelBackend(ModelBackend):
    """Native LTX-2 loading via ltx_core; model instances stay unmodified."""

    def load_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        from miles.backends.fsdp_utils.models.ltx2 import (
            build_ltx_train_scheduler,
            load_ltx_transformer_for_train,
            resolve_transformer_checkpoint,
        )

        modules = list(args.update_weight_target_modules)
        if modules != ["transformer"]:
            raise ValueError(f"LTX trains the single DiT ('transformer'); got {modules}")
        # TODO: meta-init on non-rank-0 before multi-node runs (每 rank 全量加载).
        checkpoint = resolve_transformer_checkpoint(
            str(args.diffusion_model),
            explicit_path=getattr(args, "sglang_transformer_weights_path", None),
        )
        model = load_ltx_transformer_for_train(checkpoint, device="cpu", dtype=master_dtype)
        return {"transformer": model}, build_ltx_train_scheduler(args)

    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        model.set_gradient_checkpointing(True)

    def fsdp_no_split_modules(self, model: torch.nn.Module) -> list[str]:
        return ["BasicAVTransformerBlock"]

    def set_attention_backend(self, model: torch.nn.Module, backend: str) -> None:
        # ltx_core picks attention via its AttentionFunction enum per Attention submodule,
        # not diffusers' set_attention_backend(str); FA3/FA4 switch only the unmasked path.
        from ltx_core.model.transformer.attention import Attention, AttentionFunction, MaskedAttentionFunction

        aliases = {"fa3": "FLASH_ATTENTION_3", "fa4": "FLASH_ATTENTION_4", "sdpa": "PYTORCH", "native": "PYTORCH"}
        name = aliases.get(backend.strip().lower(), backend.strip().upper())
        if name not in AttentionFunction.__members__:
            valid = ", ".join(m.name.lower() for m in AttentionFunction)
            raise ValueError(
                f"LTX --fsdp-attention-backend='{backend}' is not an ltx_core backend; "
                f"choose one of {{{valid}}} (aliases: fa3, fa4, sdpa)."
            )
        attn_fn = AttentionFunction[name].to_callable()
        masked_fn = (
            MaskedAttentionFunction[name].to_callable() if name in MaskedAttentionFunction.__members__ else None
        )
        for module in model.modules():
            if isinstance(module, Attention):
                module.attention_function = attn_fn
                if masked_fn is not None:
                    module.masked_attention_function = masked_fn
