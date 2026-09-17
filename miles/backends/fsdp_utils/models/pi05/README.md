# pi0.5

> **Read the docs:** [Pi0.5 support design](../../../../../docs/developer/pi05-torchtitan-design.md)

Draft. Native model package for `MilesModelBackend`, plus
`configs/pi05.py` registering the `pi05` family.

| file | contents |
|---|---|
| `layers.py` | `DualExpertBlock`, `ExpertStream`, `AdaRMSModulation`, `GemmaRMSNorm`, `GemmaMLP`, `TimestepMLP`, RoPE, the block mask |
| `model.py` | `Pi05Model`, `Pi05Config` |
| `loading.py` | `load_component`, checkpoint resolution |
| `modeling.py` | `load_scheduler` (flow-matching timesteps), `enable_gradient_checkpointing` |
| `parallel_plan.py` | `FSDP_PARALLEL_PLAN`, `sequence_parallel_plan` |
| `attention.py` | `set_attention_backend` |

The piece to review is `DualExpertBlock`: two experts with separate weights at
different residual widths (2048 and 1024), q/k/v concatenated for one joint
attention call, split back so each applies its own output projection and MLP.
Only the action expert's norms are modulated, which is openpi's
`use_adarms=[False, True]`.

Every module takes one frozen config dataclass and contains no parallelism
logic, so the eventual torchtitan port swaps `nn.Module` for its `Module` and
expands the dataclasses into nested `Config`s, rather than restructuring.

## Three details that are silently wrong if reversed

- adaRMS chunks **`scale, shift, gate`**. Most DiT code, Flux included, chunks
  `shift, scale`.
- openpi's flow matching has **`t = 1` as noise and `t = 0` as data**, so
  `x_t = t * noise + (1 - t) * actions` and the target is `noise - actions`.
- An adaRMS norm has **no `weight`**; the modulation Dense replaces it. The
  unconditional norms use Gemma's `(1 + weight)` with a zero-centered weight.

## Not here yet, by intent

- **Vision tower.** `Pi05Model.forward` takes already-encoded image tokens via
  `image_embeds_BIP`. PaliGemma's is SigLIP-So400m/14.
- **Checkpoint conversion.** `pi05_base` ships as an orbax parameter tree under
  `gs://openpi-assets`, which torch cannot read. `loading.resolve_checkpoint`
  requires an already-converted `.safetensors` and says so; the converter and
  its key map are separate work, and `load_state_dict` stays `strict=False`
  until that map is settled.
- **Tokenizer and data pipeline.** pi0.5's prompt is
  `"Task: {task}, State: {s0} {s1} ...;\nAction: "` with state discretized into
  256 bins -- except on LIBERO, where `discrete_state_input=False` means state
  reaches the model nowhere at all. Reproduce that deliberately.
- **Sequence parallelism.** `sequence_parallel_plan` raises. The prefix is under
  a thousand tokens, so USP has little to shard.
- **`pi05_action_horizon` as a real flag.** `build_config` reads it via `getattr`
  because `arguments.py` does not define it yet.

## Status

Syntax-checked only. Nothing here has been executed -- it needs torch and a GPU
box, neither available where it was written. Treat every numeric detail as
unverified against openpi until there is a forward-parity test.
