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
        """Meta-device twin of ``load_models_and_scheduler`` (same state-dict keys and
        dtypes, no weight IO) for ranks that receive weights via broadcast."""
        logger.warning(
            "%s lacks build_meta_models_and_scheduler; every rank loads full weights",
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
    """Load exactly the ``--update-weight-target-module`` components (plus the
    scheduler) straight from the checkpoint; no pipeline-level loading."""

    @staticmethod
    def _component_class(pipeline_config, component: str, hf_checkpoint: str):
        import diffusers

        entry = pipeline_config.get(component)
        if entry is None:
            raise ValueError(
                f"--update-weight-target-module: pipeline {hf_checkpoint} " f"has no component '{component}'"
            )
        _, class_name = entry
        cls = getattr(diffusers, class_name, None)
        if cls is None:
            raise ValueError(
                f"component '{component}' class '{class_name}' is not diffusers-native; "
                "override the loader in a custom ModelBackend"
            )
        return cls

    @staticmethod
    def _load_scheduler(args, pipeline_config):
        import diffusers

        _, scheduler_class = pipeline_config["scheduler"]
        return getattr(diffusers, scheduler_class).from_pretrained(args.hf_checkpoint, subfolder="scheduler")

    def load_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        pipeline_config = DiffusionPipeline.load_config(args.hf_checkpoint)
        raw_models: dict[str, torch.nn.Module] = {}
        for component in args.update_weight_target_modules:
            cls = self._component_class(pipeline_config, component, args.hf_checkpoint)
            raw_models[component] = cls.from_pretrained(
                args.hf_checkpoint, subfolder=component, torch_dtype=master_dtype
            )
        return raw_models, self._load_scheduler(args, pipeline_config)

    def build_meta_models_and_scheduler(
        self,
        args,
        *,
        master_dtype: torch.dtype,
    ) -> tuple[dict[str, torch.nn.Module], Any]:
        # from_pretrained ignores an outer init_empty_weights, so build from configs.
        from accelerate import init_empty_weights

        pipeline_config = DiffusionPipeline.load_config(args.hf_checkpoint)
        raw_models: dict[str, torch.nn.Module] = {}
        for component in args.update_weight_target_modules:
            cls = self._component_class(pipeline_config, component, args.hf_checkpoint)
            with init_empty_weights():
                model = cls.from_config(cls.load_config(args.hf_checkpoint, subfolder=component))
            model.to(master_dtype)
            raw_models[component] = model
        return raw_models, self._load_scheduler(args, pipeline_config)
