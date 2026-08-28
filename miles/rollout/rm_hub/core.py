"""Shared building blocks for asynchronous reward actor pools."""

from __future__ import annotations

import asyncio
import logging

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils import pool_stats

_manager_placement_group = None
logger = logging.getLogger(__name__)


def set_manager_placement_group(pg) -> None:
    """Publish the manager's (pg, bundle_indices, gpu_ids) for colocated actor pools."""
    global _manager_placement_group
    _manager_placement_group = pg


def get_manager_placement_group():
    return _manager_placement_group


set_reward_placement_group = set_manager_placement_group
get_reward_placement_group = get_manager_placement_group


class AsyncRewardActorPool:
    """Round-robin pool for Ray reward actors exposing ``score_batch``."""

    def __init__(
        self,
        *,
        actor_cls,
        actor_kwargs: dict,
        num_workers: int,
        batch_size: int,
        num_gpus_per_worker: float,
        colocate: bool,
        name: str,
    ) -> None:
        if colocate:
            pg, bundle_indices, _ = get_reward_placement_group()
            # bundle_indices is sorted by (node, gpu); stride so workers spread across nodes
            # instead of stacking onto the first node's GPUs.
            stride = max(1, len(bundle_indices) // num_workers)
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_indices[w * stride],
                )
                for w in range(num_workers)
            ]
            num_gpus_per_worker = 0.05
            num_cpus_per_worker = 0.05
        else:
            strategies = ["DEFAULT"] * num_workers
            num_cpus_per_worker = 1

        self._actors = [
            actor_cls.options(
                num_cpus=num_cpus_per_worker,
                num_gpus=num_gpus_per_worker,
                scheduling_strategy=strategies[i],
            ).remote(**actor_kwargs)
            for i in range(num_workers)
        ]
        self._batch_size = batch_size
        self._round_robin_index = 0
        self._inflight = [0] * num_workers
        logger.info(
            "Initialized %s actor pool with %d workers, %.3f GPUs/worker, batch_size=%d.",
            name,
            num_workers,
            num_gpus_per_worker,
            batch_size,
        )

    def _next_actor(self):
        i = self._round_robin_index % len(self._actors)
        self._round_robin_index += 1
        pool_stats.observe("reward", self._inflight[i])
        self._inflight[i] += 1
        return i, self._actors[i]

    async def score(self, images: list, prompts: list[str]) -> list[float]:
        chunks = [
            (*self._next_actor(), images[start:start + self._batch_size], prompts[start:start + self._batch_size])
            for start in range(0, len(images), self._batch_size)
        ]

        def submit_and_collect():
            # .remote() copies each chunk into plasma in the calling thread, so
            # submitting on the loop stalls every concurrent request for that copy.
            return ray.get([actor.score_batch.remote(img, prm) for _, actor, img, prm in chunks])

        loop = asyncio.get_running_loop()
        try:
            chunked_scores = await loop.run_in_executor(None, submit_and_collect)
        finally:
            for i, *_ in chunks:
                self._inflight[i] -= 1
        return [float(score) for chunk in chunked_scores for score in chunk]
