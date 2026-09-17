---
title: Pi0.5 Support, torchtitan-Aligned
description: Design proposal for the pi0.5 action expert and dual-expert block, written to torchtitan's model contract ahead of the planned migration.
---

Proposal, not yet implemented. It exists to settle *how* pi0.5 gets written before any of it
gets written, because we plan to migrate onto [torchtitan](https://github.com/pytorch/torchtitan)
and a model authored against our current conventions would have to be rewritten to land there.

Scope is **SFT only**. RL on pi0.5 was investigated and dropped: LIBERO's reward is a sparse 0/1
BDDL predicate check evaluated on MuJoCo state, so it never enters a behavior-cloning training
loop, and openpi's released pi05 loss is plain flow-matching MSE with no discrete-token branch.

## 1. Premise: torchtitan is FSDP2, so the migration is re-organization

`ParallelDims` carries six axes — `dp_replicate`, `dp_shard`, `cp`, `tp`, `pp`, `ep` — and
`dp_shard` *is* the FSDP shard degree. `fully_shard` appears 72 times in the tree;
`FullyShardedDataParallel` appears zero times. With `tp=pp=cp=ep=1` torchtitan is exactly the
FSDP2 setup we already run, so our `fully_shard` call sites move into a `parallelize_fn` and our
`MixedPrecisionPolicy` arguments move into torchtitan's `TrainingConfig`. Nothing about parameter
sharding or checkpoint layout changes.

What that buys, and the reason to write pi0.5 this way now: torchtitan's hard rule is that **model
code contains no parallelism logic**. If we honor it, the same `model.py` serves FSDP-only today
and TP/PP later without edits.

## 2. What pi0.5 actually is, relative to pi0

Verified against openpi at `Physical-Intelligence/openpi@main`. `Pi0Config` documents two
differences, and they are the whole model delta:

- the state input joins the discrete language tokens instead of being a continuous suffix input
- the action expert injects the flow-matching timestep through adaRMSNorm

Concretely:

| | pi0 | pi0.5 |
|---|---|---|
| timestep injection | `action_time_mlp_in/out`, time embedding concatenated onto action tokens | `time_mlp_in/out` -> swish -> `adarms_cond` |
| state | `state_proj` emits one continuous suffix token | no `state_proj`; state is discretized into the prompt |
| prompt | `"{task}\n"` | `"Task: {task}, State: {s0} {s1} ...;\nAction: "` |
| `max_token_len` | 48 | 200 |
| `use_adarms` | `[False, False]` | `[False, True]` -- action expert only |

The two experts are Gemma variants with **identical attention geometry**:

| | PaliGemma prefix expert | action expert |
|---|---|---|
| `width` | 2048 | 1024 |
| `mlp_dim` | 16384 | 4096 |
| `depth` / `num_heads` / `num_kv_heads` / `head_dim` | 18 / 8 / 1 / 256 | 18 / 8 / 1 / 256 |

Matching `num_heads` and `head_dim` is what makes the design below work: each expert's q/k/v
projection maps its own width to the same `8 * 256`, so the two token groups concatenate into one
attention call despite different residual widths.

## 3. Key finding: `DoubleStreamBlock` is already this shape

torchtitan's Flux model has `DoubleStreamBlock`, and structurally it is the pi0 dual-expert block:

- two streams (`img`, `txt`), each with its own `qkv`, `proj`, `mlp`, `norm1`, `norm2`, `mod`
- q, k, v concatenated across streams -> **one** attention call
- output split back per stream, each applying its own projection and MLP
- a conditioning vector drives `Modulation` -> `shift`, `scale`, `gate` per stream

Mapping onto pi0.5: `txt` stream becomes the PaliGemma prefix (images + language + discretized
state), `img` stream becomes the action expert suffix (noisy action chunk). Only the suffix stream
is modulated.

This is worth more than a stylistic resemblance. It means the dual-expert pattern is not a new
abstraction we have to argue for in review -- it is an in-tree, converged pattern, and writing
pi0.5 against it is the cheapest path to something torchtitan would accept.

Two places pi0.5 diverges from Flux's version:

1. Flux's streams share `hidden_size`; ours do not (2048 vs 1024). Every per-stream module is
   already separately configured, so this costs nothing structurally -- but any code that assumes
   one `hidden_size` for the block has to take two.
2. Flux modulates both streams with adaLN over `LayerNorm`; we modulate one stream with adaRMS
   over `RMSNorm`, and the gate multiplies the sublayer output rather than appearing in the norm.

## 4. Decision: adaRMS is a separate module, not a modified `RMSNorm`

openpi implements adaRMS by changing `GemmaRMSNorm.forward` to take a `cond` argument and return
`(hidden_states, gate)` instead of one tensor, then ships a `transformers_replace/` directory that
**overwrites** HF transformers' `modeling_gemma.py`, `modeling_paligemma.py`, and
`modeling_siglip.py` on install.

We should not copy that. Overwriting an upstream package's source is not portable, and torchtitan
would reject both the signature change (`RMSNorm` is shared by every model in
`models/common/`) and the conditional branch inside a shared module.

Flux's answer is the one to take: keep the norm unconditional and put the conditioning in its own
`Modulation` module whose output the block applies. The math is identical; only the ownership moves.

The exact parametrization to reproduce, read off both the JAX (`models/gemma.py:129`) and PyTorch
(`transformers_replace/.../modeling_gemma.py`) paths so it can be matched bit-for-bit:

```
modulation      = Dense(cond_dim -> 3 * dim, bias=True)(cond)     # weight zero-init
scale, shift, gate = chunk(modulation[:, None, :], 3, dim=-1)     # NOTE: this order
normed          = x * rsqrt(mean(x^2) + eps)                      # variance computed in fp32
out             = normed * (1 + scale) + shift
residual        = x + sublayer(out) * gate
```

Three details that are easy to get wrong:

- The chunk order is **scale, shift, gate**. Flux's `LastLayer` chunks `shift, scale`. Getting this
  backwards produces a model that trains and is simply wrong.
- An adaRMS norm has **no** `weight` parameter at all -- the `Dense` replaces it. The unconditional
  path uses `normed * (1 + weight)` (Gemma's zero-init convention). The state-dict adapter has to
  know which norms have which.
- `nn.init.zeros_` is applied to the `Dense` weight but **not** its bias, so this is not true
  adaLN-Zero at init. Irrelevant when loading `pi05_base`, load-bearing if anyone trains from
  scratch.

Sizing note, because it affects FSDP wrapping decisions: each adaRMS `Dense` is
`1024 x 3072 + 3072 ~= 3.15M` parameters, and there are 37 of them (2 per layer x 18, plus the
final norm) -- about **117M parameters of modulation against a ~311M action expert**. Not a
rounding error; worth deciding deliberately whether modulation shares an FSDP wrap with its block.

## 5. Attention mask

pi0 builds a block-causal mask by cumulative sum over a per-token `mask_ar`:

```python
cumsum    = cumsum(mask_ar, axis=1)
attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
attn_mask = attn_mask & (input_mask[:, None, :] * input_mask[:, :, None])
```

For pi0.5 the `ar_mask` is `[False] * n_prefix` for images and language, then `[True] + [False] *
(action_horizon - 1)` for the action chunk. That is two blocks: the prefix attends within itself
bidirectionally, and the action chunk attends to the whole prefix plus bidirectionally within
itself. Prefix-LM with one trailing block, nothing more exotic.

This maps onto a FlexAttention `mask_mod` — two blocks compared by block index — rather than a
materialized `[B, N, N]` boolean. `models/common/attention.py` already exposes `FlexAttention` and
mask-mod helpers, so this should reuse rather than hand-roll.

## 6. What to reuse from `models/common/`

`Linear`, `RMSNorm`, `Embedding`, `LayerNorm`, `SiLU`, `GELU`, `Identity`, `FeedForward`,
`RoPE` / `ComplexRoPE`, `FlexAttention`, `VisionTransformerBlock`, and `scatter_vision_embeds` in
`common/multimodal.py`.

The SigLIP tower and image-token scatter are the same problem the in-tree VLMs already solved --
`qwen3_5`, `kimi_k3`, `kimi_k2_7`, `muse_glimmer` all build on these -- so the PaliGemma half
should follow one of those rather than being designed here.

Note that using these is not optional decoration. `Trainer` calls
`model.verify_module_protocol()`, which walks `named_modules()` and raises on any submodule that
is not a torchtitan `Module`. A bare `nn.Linear` fails the check at startup. The wrappers are thin
(`class Linear(nn.Linear, Module)`), so this is mechanical, but it has to be done everywhere from
the start.

## 7. Shape of the contribution

Following llama3 (minimal) and flux (bespoke pipeline):

```text
pi05/
  __init__.py            # flavor dict + model_registry(flavor) -> ModelSpec
  model.py               # Pi05Model(BaseModel) + nested Config
  layers.py              # DualExpertBlock, AdaRMSModulation, action in/out projections
  parallelize.py         # parallelize_pi05(): AC -> compile -> apply_fsdp
  sharding.py
  state_dict_adapter.py  # to_hf() / from_hf() against pi05_base
  trainer.py             # flow-matching step; Pi0.5 has no next-token path
  config_registry.py
```

`parallelize_pi05` only needs to fill the `dp_shard` axis for the first cut — one
`apply_fsdp` call, per-block wrapping as flux does it. TP/PP get added later without touching
`model.py`, which is the whole point of section 1.

A bespoke `trainer.py` is required rather than chosen: `BaseModel.preprocess_inputs` raises
`NotImplementedError` by design, and its docstring names Flux as the precedent for models whose
pipeline never calls it. Subclass `Trainer` and override `forward_backward_step` and
`batch_generator`.

## 8. Replicate deliberately: `pi05_libero` drops robot state

`TrainConfig(name="pi05_libero")` sets `Pi0Config(pi05=True, action_horizon=10,
discrete_state_input=False)`, overriding the pi05 default of `discrete_state_input=True`. The
consequence is that `obs.state` reaches the model nowhere:

- `pi05=True` means `state_proj` is never constructed and `embed_suffix` skips the state token
- `discrete_state_input=False` means `TokenizePrompt` does not splice state into the prompt
- `embed_prefix` consumes only images and the tokenized prompt

`LiberoInputs` still emits `"state"`, so the tensor is built and then ignored. This is easy to
mistake for a bug and "fix", which would silently stop matching the released checkpoint. On LIBERO,
pi0.5 is vision-and-language conditioned only.

Other `pi05_libero` values worth carrying over rather than re-deriving: `action_horizon=10` (pi0
defaults to 50), `batch_size=256`, 30k steps, cosine schedule with 10k warmup at peak `5e-5`,
AdamW with gradient-norm clipping at 1.0, EMA decay 0.999.

## 9. Open questions for review

1. **Where does this live?** There is no torchtitan fork under `Rockdu` today, so this proposal
   assumes miles_diffusion. If the migration lands first, `torchtitan/experiments/pi05/` is the
   more natural home and the layout in section 7 transfers unchanged.
2. **One `DualExpertBlock` or two configured streams?** Flux hardcodes `img_*` / `txt_*` attribute
   pairs. A `list[ExpertStream.Config]` generalizes to pi0's three-group case and to any future
   third expert, at the cost of diverging from the in-tree precedent.
3. **FSDP wrap granularity for modulation.** Given the 117M figure in section 4, does each
   adaRMS `Dense` share its block's wrap, or get its own?
4. **Do we want `pi0` too, or only `pi0.5`?** The delta is two switches. Supporting both costs
   little and makes the adaRMS path testable against a non-adaRMS baseline, but doubles the
   checkpoint-conversion surface.
5. **Vision tower**: follow `qwen3_5` or `kimi_k3`? Both use `common/vision_encoder.py`; I have not
   compared how closely either matches SigLIP-So400m/14 as PaliGemma configures it.

## 10. Deliberately not in scope

- **RL / Pi RL reproduction.** Dropped, per the scope note at the top.
- **FAST discrete-token training and knowledge insulation.** Both are pi0.5 *pretraining*
  machinery. openpi's open-sourced loss is `mean((v_t - u_t)^2)` and nothing else, so neither is
  needed for SFT.
- **LIBERO evaluation infrastructure.** Reward needs no GPU -- a few numpy geometric checks -- but
  offscreen rendering of two camera views for hundreds of steps per trial does, and that stack is
  a separate piece of work from training support.
- **The `transformers_replace` mechanism.** Rejected in section 4; we implement the layers instead.
