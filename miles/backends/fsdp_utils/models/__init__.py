"""Self-maintained (native) modeling for the FSDP diffusion trainer.

miles-diffusion supports two modeling mechanisms, driven through ONE
per-family ``TrainPipelineConfig`` on top:

1. **diffusers checkpoints** — nothing to do here. ``DiffusersModelBackend``
   loads via ``DiffusionPipeline.from_pretrained``; the checkpoint's
   ``model_index.json`` resolves the classes.

2. **native / self-built modeling** (a model that does not ship as a
   diffusers pipeline, e.g. LTX-2 via ``ltx_core``) — add a module here that
   exposes the model behind the *diffusers interface protocol*:

   - ``from_pretrained(ref, *, torch_dtype=..., ...) -> nn.Module``
   - the returned module declares ``_no_split_modules`` (FSDP wrap classes)
   - the returned module has ``enable_gradient_checkpointing()``
   - a ``build_<family>_train_scheduler(args)`` returning an object with the
     scheduler surface the trainer touches (``sigmas`` / ``timesteps`` /
     ``num_inference_steps`` / ``to(device)``)

To onboard a new model family end-to-end:

- ``models/<family>.py``     — this protocol shim (only if not diffusers)
- ``configs/<family>.py``    — ``TrainPipelineConfig`` subclass: cond schema,
  CFG policy, trajectory unpack, plus ``detect()``/``pipeline_class_prefixes``
  so the family is recognized from the checkpoint itself
  (``@register_train_pipeline_config("<family>")``)
- a thin ``ModelBackend`` subclass in ``model_backend.py`` pointing at the
  shim (see ``LTXModelBackend`` — ~30 lines)

See ``models/ltx2.py`` for the reference implementation.
"""
