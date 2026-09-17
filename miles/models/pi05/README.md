# pi0.5, torchtitan-shaped

Draft. Written to torchtitan's model contract so it lands unchanged once we
migrate; see [the design note](../../../docs/developer/pi05-torchtitan-design.md)
for why each decision went the way it did.

`miles/models/` rather than `miles/backends/fsdp_utils/models/` because this
targets torchtitan's `ModelSpec`, not our FSDP backend's loader, and mirrors
torchtitan's own `torchtitan/models/<name>/` layout.

## What is here

| file | contents |
|---|---|
| `layers.py` | `DualExpertBlock`, `ExpertStream`, `AdaRMSModulation`, `GemmaRMSNorm`, `GemmaMLP`, `TimestepMLP`, RoPE, the block mask |
| `model.py` | `Pi05Model(BaseModel)` -- prefix embedding, the layer stack, flow-matching velocity head |
| `__init__.py` | `libero` and `base` flavors, `model_registry(flavor) -> ModelSpec` |
| `parallelize.py` | `parallelize_pi05()`: AC, compile, then FSDP2 per block |

The shape worth reviewing is `DualExpertBlock`: two experts with separate
weights at different residual widths (2048 and 1024), q/k/v concatenated for one
joint attention call, split back so each expert applies its own output
projection and MLP. Only the action expert's norms are modulated, which is
openpi's `use_adarms=[False, True]`.

## Not here yet, by intent

- **Vision tower.** `Pi05Model.forward` takes already-encoded image tokens.
  `qwen3_5` and `kimi_k3` both build SigLIP on `models/common/vision_encoder.py`
  and one of them should be followed rather than a third variant written.
- **`trainer.py`.** Needed, not optional: `BaseModel.preprocess_inputs` raises
  by design for models with a bespoke pipeline. Subclass `Trainer` and override
  `forward_backward_step` and `batch_generator`, as flux does.
- **`state_dict_adapter.py`.** `pi05_base` ships as an orbax param tree rather
  than HF safetensors, so `to_hf`/`from_hf` need a conversion step first.
  `model_registry` passes `state_dict_adapter=None` until then.
- **`sharding.py`.** Nothing to place while only `dp_shard` is filled.
- **Tokenizer and data pipeline.** pi0.5's prompt is
  `"Task: {task}, State: {s0} {s1} ...;\nAction: "` with state discretized into
  256 bins -- except on LIBERO, where `discrete_state_input=False` means state
  reaches the model nowhere at all. Reproduce that deliberately.
- **`FlexInnerAttention`.** `MaskedInnerAttention` passes an explicit mask to
  SDPA because torchtitan's SDPA wrapper only offers `is_causal`. The flex path
  wants `[T, H, K]` with the batch folded away, which is a larger change to
  sequence handling than this draft settles.

## Status

Syntax-checked only. Nothing here has been executed -- it needs `torchtitan`,
`spmd_types==0.2.5` and a GPU box, none of which were available where it was
written. Treat every numeric detail as unverified against openpi until there is
a forward-parity test.
