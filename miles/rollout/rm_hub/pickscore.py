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


def _feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features
    return features.pooler_output


def _generated_output_to_fchw(t: torch.Tensor) -> torch.Tensor:
    """Normalize any video/image tensor layout to ``[F, C, H, W]`` float in ``[0, 1]``.

    Detects the channel axis by C in {1, 3} rather than assuming a fixed order,
    so LTX (frames-first) and diffusers/Wan ([C, F, H, W]) both resolve correctly.
    """
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
        raise ValueError(f"generated_output must be 3D-5D, got {tuple(t.shape)}")
    if float(t.max()) > 1.0 + 1e-3:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def _sample_to_rgb_hwc_uint8_frames(sample: Sample, num_frames: int | None = None) -> list[np.ndarray]:
    fchw = _generated_output_to_fchw(sample.generated_output)  # [F, C, H, W] in [0, 1]
    total = fchw.shape[0]
    if num_frames is not None and 0 < num_frames < total:
        # Evenly spaced subset — long videos don't need every frame scored.
        frame_indices = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()
    else:
        frame_indices = range(total)
    frames = []
    for frame_index in frame_indices:
        hwc = fchw[frame_index].permute(1, 2, 0).numpy() * 255.0
        frames.append(np.ascontiguousarray(hwc.clip(0, 255).astype(np.uint8)))
    return frames


class PickScoreScorer(torch.nn.Module):
    """Small local copy of Flow-GRPO's PickScore scorer.

    The scorer consumes final PIL images and prompt strings, then returns one
    scalar reward per prompt/image pair.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        processor_path: str,
        model_path: str,
    ) -> None:
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        image_inputs = self.processor(
            images=list(images),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = self.processor(
            text=list(prompts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {key: value.to(device=self.device) for key, value in image_inputs.items()}
        text_inputs = {key: value.to(device=self.device) for key, value in text_inputs.items()}

        image_embs = _feature_tensor(self.model.get_image_features(**image_inputs))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        text_embs = _feature_tensor(self.model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)
        scores = scores.diag() / 26.0
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
        self.scorer = PickScoreScorer(
            device=device,
            processor_path=processor_path,
            model_path=model_path,
        )

    def score_batch(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        pil_images = [Image.fromarray(image) for image in images]
        return self.scorer(prompts, pil_images)


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

    async def score(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        refs = []
        for start in range(0, len(images), self._batch_size):
            end = start + self._batch_size
            refs.append(self._next_actor().score_batch.remote(images[start:end], prompts[start:end]))

        loop = asyncio.get_running_loop()
        chunked_scores = await loop.run_in_executor(None, ray.get, refs)
        return [float(score) for chunk in chunked_scores for score in chunk]


async def pickscore_rm(args, samples: Sequence[Sample]) -> list[float]:
    # Score every frame and mean-pool per sample
    pool = AsyncPickScorePool(args)
    images: list[np.ndarray] = []
    prompts: list[str] = []
    frame_counts: list[int] = []
    for sample in samples:
        frames = _sample_to_rgb_hwc_uint8_frames(sample, getattr(args, "pickscore_num_frames", None))
        images.extend(frames)
        prompts.extend([sample.prompt] * len(frames))
        frame_counts.append(len(frames))
    flat_scores = await pool.score(images, prompts)
    scores: list[float] = []
    offset = 0
    for count in frame_counts:
        scores.append(float(sum(flat_scores[offset : offset + count]) / count))
        offset += count
    return scores
