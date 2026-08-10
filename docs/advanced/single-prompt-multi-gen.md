---
title: Single-Prompt Multi-Generation
description: One rollout request, N outputs — engine-side conditioning expansion, microgroup mechanics, and seed layout.
---
GRPO needs `n_samples_per_prompt` outputs per prompt to normalize advantages
within a group. The naive way is N separate requests, which encodes the same
prompt N times and denoises N batches of size 1. Miles-diffusion instead sends
**one request per microgroup**: sglang-diffusion encodes the conditioning
once, expands it engine-side, and denoises all N outputs as a batch.
Engine-side expansion is opt-in per model family in sglang-diffusion.

## 1. Microgroups

`--diffusion-microgroup-size M` splits each prompt group of
`n_samples_per_prompt` samples into requests of at most M
(`generate_and_rm_group` in `miles/rollout/sglang_diffusion_rollout.py`).
Each request carries `num_outputs_per_prompt = M`:

```
group (1 prompt × n_samples_per_prompt samples)
  ├─ microgroup 0  → POST /rollout/generate  num_outputs_per_prompt=M
  ├─ microgroup 1  → POST /rollout/generate  num_outputs_per_prompt=M
  └─ ...
```

Microgroups of one group run as concurrent asyncio tasks, load-balanced across
engines; in-flight requests per engine are bounded by
`--sglang-server-concurrency`.

Canonical values on `main`:

| Recipe | `n_samples_per_prompt` | microgroup size |
|---|---|---|
| SD3.5 GRPO + OCR | 16 | 8 |
| Qwen-Image GRPO + PickScore | 16 | 8 |
| Wan2.2 GRPO + PickScore | 16 | 8 |
| LTX-2.3 GRPO + PickScore | 8 | 1 |
| Cosmos3 GRPO | 16 | 1 — packed forward is single-sample |

## 2. Seed layout

Rollout stays deterministic and collision-free per sample:

- sgl-d expands one request's `seed` into `seed, seed+1, …, seed+M−1` — one
  RNG stream per output.
- The trainer spaces request seeds so streams never overlap:

```python
seed_base = (rollout_seed + group_index * n_samples_per_prompt) % 2**31
microgroup_seed = seed_base + idx   # idx = offset of the microgroup in its group
```

`group_index` is monotonic across the run, so every `(rollout, prompt-group,
sample)` triple gets a distinct seed. This is what makes rollout replay and
[deterministic mode](/advanced/deterministic) possible at microgroup
granularity.

<Note>
sgl-d currently accepts only the first seed per request and derives the rest
by increment; per-sample seed lists are a tracked TODO.
</Note>

## 3. Interaction with SDE step strategies

The SDE step subset (`--diffusion-step-strategy-path`) is computed **once per
request** from the microgroup's seed, and every output in that request shares
it (`rollout_sde_step_indices` in the payload). Samples within a microgroup
therefore train on the same denoising steps — the diversity GRPO needs comes
from the per-output noise streams, not from different step windows.

The strategy fn receives the request seed, so two microgroups of the same
group can still get different windows under `sde_window` while
`epoch_global_random_choice` deliberately shares one subset across the whole
rollout (the Wan2.2 recipe).

## 4. When to keep microgroup size at 1

- **Packed-forward models.** Cosmos3's transformer takes one packed
  text+vision sequence per forward; a request cannot batch outputs.
- **Memory-bound video rollout.** LTX-2.3 keeps microgroup 1 and instead
  scales the [deserialization pool](/advanced/streaming-reward) — one 8-output
  video response would be a single multi-GB body on one parser actor.
- **Debugging.** One sample per request makes engine logs and
  `--diffusion-debug-mode` tensor dumps trivially attributable.

Otherwise, prefer larger microgroups: conditioning encodes once, the DiT
denoises N latents per step in one batch, and HTTP/parse overhead drops by
~M×. Microgroup size is also the first knob to lower on rollout OOM (before
touching batch size) — see the note in the
[Wan2.2 guide](/models/wan/wan2-2) §5.5.

## 5. Pairs well with

- [Streaming Reward and Deserialization](/advanced/streaming-reward) — what
  happens to a microgroup response after generation.
- [Deterministic Training](/advanced/deterministic) — seed layout is half of
  run reproducibility.
- [Core Concepts](/user-guide/concepts) — where microgroup size sits among the
  batch knobs.
