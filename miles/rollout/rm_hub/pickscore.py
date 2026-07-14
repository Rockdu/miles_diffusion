from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import numpy as np
import ray
import torch
from PIL import Image

from miles.utils.misc import SingletonMeta
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def sample_frame_indices(num_total_frames: int, num_frames: int | None) -> list[int]:
    if num_total_frames <= 0:
        raise ValueError(f"video has no frames: {num_total_frames}")
    if num_frames is None or num_total_frames <= num_frames:
        return list(range(num_total_frames))
    if num_frames == 1:
        return [num_total_frames // 2]
    step = (num_total_frames - 1) / (num_frames - 1)
    return [int(round(i * step)) for i in range(num_frames)]


def generated_output_to_fchw(t: torch.Tensor) -> torch.Tensor:
    """Return ``[F, C, H, W]`` float tensor in ``[0, 1]``."""
    t = t.detach().cpu().float()
    if t.ndim == 3:
        if t.shape[0] not in (1, 3):
            raise ValueError(f"expected [C, H, W] with C in {{1, 3}}, got {tuple(t.shape)}")
        t = t.unsqueeze(0)
    elif t.ndim == 4:
        if t.shape[-1] in (1, 3):
            t = t.permute(0, 3, 1, 2)
        elif t.shape[0] in (1, 3):
            t = t.permute(1, 0, 2, 3)
        elif t.shape[1] not in (1, 3):
            raise ValueError(f"unrecognized 4D video layout: {tuple(t.shape)}")
    elif t.ndim == 5:
        if t.shape[0] == 1 and t.shape[-1] in (1, 3):
            t = t[0].permute(0, 3, 1, 2)
        else:
            raise ValueError(f"unrecognized 5D video layout: {tuple(t.shape)}")
    else:
        raise ValueError(f"generated_output must be 3D–5D, got {tuple(t.shape)}")

    if float(t.max()) > 1.0 + 1e-3:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def fchw_frame_to_hwc_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    hwc = frame_chw.numpy().transpose(1, 2, 0)
    if float(hwc.max()) <= 1.0 + 1e-3:
        hwc = hwc * 255.0
    return np.ascontiguousarray(hwc.clip(0, 255).astype(np.uint8))


def _feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output"):
        pooled = features.pooler_output
        if isinstance(pooled, torch.Tensor):
            return pooled
    for attr in ("image_embeds", "text_embeds"):
        value = getattr(features, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(features, tuple):
        for item in reversed(features):
            if isinstance(item, torch.Tensor) and item.ndim == 2:
                return item
        raise TypeError(f"No 2-D tensor in model output tuple (len={len(features)})")
    raise TypeError(f"Cannot extract embedding tensor from {type(features)!r}")


def _sample_to_rgb_hwc_uint8_frames(sample: Sample, num_frames: int | None) -> list[np.ndarray]:
    fchw = generated_output_to_fchw(sample.generated_output)
    return [fchw_frame_to_hwc_uint8(fchw[i]) for i in sample_frame_indices(fchw.shape[0], num_frames)]


class PickScoreScorer(torch.nn.Module):
    """CLIP PickScore for (prompt, image) pairs; raw logits scaled to ~0-1."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        processor_path: str,
        model_path: str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(device=self.device, dtype=dtype)

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        image_inputs = self.processor(images=list(images), return_tensors="pt", padding=True)
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        if "pixel_values" in image_inputs:
            image_inputs["pixel_values"] = image_inputs["pixel_values"].to(self.dtype)

        text_inputs = self.processor(
            text=list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        image_embs = _feature_tensor(self.model.get_image_features(**image_inputs))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        text_embs = _feature_tensor(self.model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        scores = self.model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)
        # Flow-Factory convention: scale raw PickScore logits (~0-26) to ~0-1.
        scores = scores.float() / 26.0
        return [float(score) for score in scores.detach().cpu()]


@ray.remote
class PickScoreRewardActor:
    def __init__(
        self,
        *,
        processor_path: str,
        model_path: str,
    ) -> None:
        gpu_ids = ray.get_gpu_ids()
        use_cuda = bool(gpu_ids) and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        device = "cuda" if use_cuda else "cpu"
        self.device = device
        self._processor_path = processor_path
        self._model_path = model_path
        self._scorers: dict[torch.dtype, PickScoreScorer] = {}

    def _scorer(self, dtype: torch.dtype) -> PickScoreScorer:
        if self.device == "cpu":
            dtype = torch.float32
        if dtype not in self._scorers:
            self._scorers[dtype] = PickScoreScorer(
                device=self.device,
                processor_path=self._processor_path,
                model_path=self._model_path,
                dtype=dtype,
            )
        return self._scorers[dtype]

    def score_batch(self, images: list, prompts: list[str], *, fp16: bool = False) -> list[float]:
        pil_images = [Image.fromarray(image) if isinstance(image, np.ndarray) else image for image in images]
        return self._scorer(torch.float16 if fp16 else torch.float32)(prompts, pil_images)


class AsyncPickScorePool(metaclass=SingletonMeta):
    """Ray actor pool for GPU PickScore reward inference."""

    def __init__(self, args) -> None:
        num_workers = args.pickscore_num_workers
        num_gpus_per_worker = args.pickscore_num_gpus_per_worker
        self._batch_size = args.pickscore_batch_size
        self._actors = [
            PickScoreRewardActor.options(
                num_cpus=1,
                num_gpus=num_gpus_per_worker,
                scheduling_strategy="DEFAULT",
            ).remote(
                processor_path=args.pickscore_processor_path,
                model_path=args.pickscore_model_path,
            )
            for _ in range(num_workers)
        ]
        self._round_robin_index = 0
        logger.info(
            "Initialized PickScore actor pool with %d workers, %.3f GPUs/worker, batch_size=%d.",
            num_workers,
            num_gpus_per_worker,
            self._batch_size,
        )

    def _next_actor(self):
        i = self._round_robin_index % len(self._actors)
        self._round_robin_index += 1
        return self._actors[i]

    async def score(self, images: list, prompts: list[str], *, fp16: bool = False) -> list[float]:
        refs = []
        for start in range(0, len(images), self._batch_size):
            end = start + self._batch_size
            refs.append(self._next_actor().score_batch.remote(images[start:end], prompts[start:end], fp16=fp16))

        loop = asyncio.get_running_loop()
        chunked_scores = await loop.run_in_executor(None, ray.get, refs)
        return [float(score) for chunk in chunked_scores for score in chunk]


async def pickscore_rm(args, samples: Sequence[Sample]) -> list[float]:
    pool = AsyncPickScorePool(args)
    images: list[np.ndarray] = []
    prompts: list[str] = []
    frame_counts: list[int] = []
    for sample in samples:
        frames = _sample_to_rgb_hwc_uint8_frames(sample, args.pickscore_num_frames)
        images.extend(frames)
        prompts.extend([sample.prompt] * len(frames))
        frame_counts.append(len(frames))

    flat_scores = await pool.score(images, prompts, fp16=args.pickscore_fp16)
    scores: list[float] = []
    offset = 0
    for count in frame_counts:
        scores.append(float(sum(flat_scores[offset : offset + count]) / count))
        offset += count
    return scores
