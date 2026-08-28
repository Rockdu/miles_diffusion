"""Max queue depth seen at dispatch per actor pool, drained once per rollout."""


_max: dict[str, int] = {}


def observe(pool: str, queued: int) -> None:
    if queued > _max.get(pool, -1):
        _max[pool] = queued


def drain() -> dict[str, float]:
    stats = {f"rollout/{name}_queue_max": float(depth) for name, depth in _max.items()}
    _max.clear()
    return stats
