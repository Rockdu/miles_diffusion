---
title: Rewards
description: Built-in reward models (PickScore, OCR), rm_hub dispatch, and custom reward hooks.
---
Miles-diffusion scores generated images (or video frames) after each rollout
microgroup. Reward computation lives in `miles/rollout/rm_hub/` and is invoked
from `sglang_diffusion_rollout.generate_and_rm_microgroup`.

## 1. At a glance

| Stage | Flag | Role |
|---|---|---|
| Reward type | `--rm-type` | Selects built-in scorer (`pickscore`, `ocr`) |
| Custom reward | `--custom-rm-path` | Replaces the entire batched dispatch |
| Advantage norm | `--custom-reward-post-process-path` | Replaces GRPO mean/std normalization |
| Per-sample override | `metadata.rm_type` in JSONL | Overrides global `--rm-type` |

<Warning>
`--diffusion-reward` is a legacy CLI argument (default `"pickscore"`) that is
**not read by reward dispatch code**. Scripts may still pass
`--diffusion-reward pickscore:1.0`, but the effective knob is **`--rm-type`**.
</Warning>

## 2. Built-in reward models

### PickScore (`--rm-type pickscore`)

Implementation: `miles/rollout/rm_hub/pickscore.py`.

PickScore scores text–image alignment using a CLIP model pair:

- Processor: `--pickscore-processor-path` (e.g.
  `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`)
- Model: `--pickscore-model-path` (e.g. `yuvalkirstain/PickScore_v1`)

Scoring formula:

```
score = exp(logit_scale) * dot(text_emb, image_emb) / 26.0
```

The `/ 26.0` scaling maps raw PickScore logits (~0–26) to roughly 0–1.

PickScore runs as a **Ray actor pool** (`PickScoreRewardActor`) with round-robin
batching. For video outputs, frames are uniformly sampled
(`--pickscore-num-frames`) and scores are averaged.

| Flag | Default | Description |
|---|---|---|
| `--pickscore-num-workers` | 1 | Ray actor count |
| `--pickscore-num-gpus-per-worker` | 1.0 | GPU per worker (non-colocate) |
| `--pickscore-batch-size` | 8 | Batch size per actor |
| `--pickscore-processor-path` | — | Required for pickscore |
| `--pickscore-model-path` | — | Required for pickscore |
| `--pickscore-num-frames` | None | Video frame sampling count |
| `--colocate-reward` | False | Share rollout GPUs (0.05 GPU/worker) |

Example from `scripts/run-diffusion-nft-sd3-pickscore.sh`:

```bash
--rm-type pickscore \
--pickscore-num-workers 1 \
--pickscore-num-gpus-per-worker 1.0 \
--pickscore-batch-size 8 \
--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
--pickscore-model-path yuvalkirstain/PickScore_v1
```

### OCR (`--rm-type ocr`)

Implementation: `miles/rollout/rm_hub/ocr.py`.

OCR reward compares PaddleOCR output against target text embedded in the
prompt. The target is extracted as the string between the first pair of double
quotes in the prompt:

```python
target = prompt.split('"')[1]
reward = 1 - levenshtein_distance(recognized, target) / len(target)
```

OCR runs on **CPU** Ray actors (`--ocr-num-workers`, default 4). Used by the
SD3 Flow-GRPO recipe (`scripts/run-diffusion-grpo-sd3-ocr-sglang.sh`).

### Remote RM (`--rm-type remote_rm`)

The CLI exposes `--rm-url` for a remote reward service, but **`rm_hub` does not
implement `remote_rm` today** — selecting it raises `NotImplementedError`.
Use `--custom-rm-path` to call an external service instead (see below).

## 3. Call chain

```
generate_and_rm_microgroup()
  → batched_async_rm(args, microgroup)          # rm_hub/__init__.py
    → custom_rm_path?  user batched function
    → all pickscore?   pickscore_rm (batched)
    → else             per-sample async_rm → ocr / pickscore / NotImplementedError
  → sample.reward = score
  → RolloutManager._post_process_rewards()      # GRPO advantage normalization
```

After rollout, `RolloutManager._post_process_rewards` subtracts the mean and
optionally divides by std to produce normalized advantages for training.

## 4. Custom reward functions

### `--custom-rm-path`

Replace the built-in dispatch entirely. Signature:

```python
async def custom_rm(args, samples: list[Sample], **kwargs) -> list[float]:
    ...
```

Wired only through `batched_async_rm` — implement per-sample routing inside
your batched function if needed.

Example registration:

```bash
--custom-rm-path my_project.rewards.aesthetic_rm
```

Debug tool: replay rewards on saved rollout data. Note that this helper calls
your function **per sample** `(args, sample)`, not batched — wrap a batched
implementation if needed:

```python
async def my_rm_for_replay(args, sample, **kwargs) -> float:
    return (await my_batched_rm(args, [sample]))[0]
```

```bash
python -m miles.utils.debug_utils.replay_reward_fn \
  --rollout-data-path /path/to/rollout.pt \
  --custom-rm-path my_project.rewards.my_rm_for_replay
```

### API reward service

To score via an HTTP API, implement a batched custom RM. Encode images from
`sample.generated_output` (see `_sample_to_rgb_hwc_uint8_frames` in
`miles/rollout/rm_hub/pickscore.py` for the tensor → uint8 path):

```python
import aiohttp
from miles.utils.types import Sample

async def api_rm(args, samples: list[Sample], **kwargs) -> list[float]:
    async with aiohttp.ClientSession() as session:
        rewards = []
        for sample in samples:
            payload = {"prompt": sample.prompt, "image_b64": "<your encoding>"}
            async with session.post(args.rm_url, json=payload) as resp:
                rewards.append((await resp.json())["score"])
        return rewards
```

Then launch with:

```bash
--custom-rm-path my_project.rewards.api_rm \
--rm-url http://localhost:8000/score
```

(`--rm-url` is passed through `args` for your function to read; it is not used
by built-in dispatch.)

### `--custom-reward-post-process-path`

Replace GRPO advantage normalization. Signature:

```python
def post_process(args, samples) -> tuple[list[float], list[float]]:
    # Returns (raw_rewards, normalized_rewards)
    ...
```

Default normalization (`RolloutManager._post_process_rewards`):

1. Reshape rewards to `(-1, n_samples_per_prompt)`.
2. Subtract mean (`--globalize-reward-mean` for batch-level, else per-group).
3. Divide by std when `--grpo-std-normalization` is enabled (default True);
   `--globalize-reward-std` uses batch-level std.

## 5. Prompt data

### JSONL format

Training prompts are loaded from `.jsonl` files via `miles/utils/diffusion_data.py`:

```json
{"input": "A photo of a cat wearing sunglasses"}
{"input": "A logo saying \"Miles\"", "metadata": {"rm_type": "ocr"}}
```

| Field | CLI mapping | Notes |
|---|---|---|
| Prompt text | `--input-key input` | Required non-empty string |
| Per-sample metadata | `--metadata-key metadata` | Optional dict; supports `rm_type` override |

### Dataset subsets

Dataset repo: [`rockdu/miles-diffusion-datasets`](https://huggingface.co/datasets/rockdu/miles-diffusion-datasets)

| Subset | Used by |
|---|---|
| `flowgrpo_pickscore/` | PickScore recipes (SD3 NFT, Qwen-Image, Wan2.2, LTX) |
| `flowgrpo_ocr/` | SD3 OCR Flow-GRPO, NFT smoke test |

### Per-sample rm_type override

JSONL `metadata.rm_type` overrides the global `--rm-type` for that sample:

```python
# rm_hub/__init__.py
metadata.get("rm_type") or args.rm_type
```

Mixed rm_types within one microgroup fall back to per-sample dispatch (no
batched PickScore fast path).
