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
import logging
from typing import Any

import torch
from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)


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

    def build_meta_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        """Architecture-only twin of ``load_models_and_scheduler`` for non-rank-0
        FSDP ranks: same components with every parameter on the meta device and no
        checkpoint weight IO. Rank 0 loads real weights and broadcasts into the
        sharded model (``actor._fsdp2_load_full_state_dict``), so state-dict keys
        and param dtypes must match rank 0 exactly.

        Default falls back to a full load (the pre-meta-init behavior) so custom
        backends keep working; override for the memory/IO win.
        """
        logger.warning(
            "%s does not implement build_meta_models_and_scheduler; "
            "falling back to loading full weights on every rank",
            type(self).__name__,
        )
        return self.load_models_and_scheduler(args, master_dtype=master_dtype)

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

    def build_meta_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        # diffusers' from_pretrained materializes real weights even inside an outer
        # accelerate.init_empty_weights (loading happens after __init__ via direct
        # tensor assignment), so build skeletons from component configs instead.
        import diffusers
        from accelerate import init_empty_weights

        pipeline_config = DiffusionPipeline.load_config(args.hf_checkpoint)

        raw_models: dict[str, torch.nn.Module] = {}
        for component in args.update_weight_target_modules:
            entry = pipeline_config.get(component)
            if entry is None:
                raise ValueError(
                    f"--update-weight-target-module: pipeline {args.hf_checkpoint} " f"has no component '{component}'"
                )
            _, class_name = entry
            cls = getattr(diffusers, class_name, None)
            if cls is None:
                raise ValueError(
                    f"component '{component}' class '{class_name}' is not a diffusers-native class; "
                    "override build_meta_models_and_scheduler in a custom ModelBackend"
                )
            sub_config = cls.load_config(args.hf_checkpoint, subfolder=component)
            with init_empty_weights():
                model = cls.from_config(sub_config)
            model.to(master_dtype)
            raw_models[component] = model

        _, scheduler_class = pipeline_config["scheduler"]
        scheduler = getattr(diffusers, scheduler_class).from_pretrained(args.hf_checkpoint, subfolder="scheduler")
        return raw_models, scheduler
