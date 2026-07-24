"""Ray lifecycle glue for the optional diffusion dashboard collector."""

from __future__ import annotations

import logging
import time

from miles.dashboard.collector import COLLECTOR_ACTOR_NAME, CollectorConfig

logger = logging.getLogger(__name__)

GET_ACTOR_TIMEOUT_SECONDS = 60.0
GET_ACTOR_INTERVAL_SECONDS = 2.0

_handle = None
_is_primary = False
_resolution_failed = False


def init_dashboard(args) -> bool:
    """Create the collector on the driver. Failure disables dashboard telemetry."""
    global _handle, _is_primary
    if not args.use_miles_dashboard:
        return False

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from miles.dashboard.collector import DashboardCollector

    config = CollectorConfig(
        workspace=args.miles_dashboard_workspace,
        run_name="miles-diffusion",
        start_ts=time.time(),
        args_snapshot=dict(vars(args)),
    )
    handle = None
    try:
        handle = (
            ray.remote(DashboardCollector)
            .options(
                name=COLLECTOR_ACTOR_NAME,
                num_cpus=0,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            )
            .remote(config)
        )
        ray.get(handle.ping.remote())
        handle.start.remote()
    except Exception:
        logger.warning("dashboard collector startup failed; telemetry is disabled", exc_info=True)
        if handle is not None:
            try:
                ray.kill(handle)
            except Exception:
                pass
        _handle = None
        _is_primary = False
        return False

    _handle = handle
    _is_primary = True
    logger.info(
        "miles dashboard telemetry -> %s | view: python -m miles.dashboard.viewer --serve --workspace %s",
        config.workspace,
        config.workspace,
    )
    return True


def resolve_collector():
    """Resolve the named collector once per worker process."""
    global _handle, _resolution_failed
    if _handle is not None or _resolution_failed:
        return _handle

    import ray

    deadline = time.monotonic() + GET_ACTOR_TIMEOUT_SECONDS
    while True:
        try:
            _handle = ray.get_actor(COLLECTOR_ACTOR_NAME)
            return _handle
        except ValueError:
            if time.monotonic() >= deadline:
                logger.warning(
                    "dashboard collector not found after %.0fs; telemetry from this process is disabled",
                    GET_ACTOR_TIMEOUT_SECONDS,
                )
                _resolution_failed = True
                return None
            time.sleep(GET_ACTOR_INTERVAL_SECONDS)


def finish_dashboard() -> None:
    """Flush the driver's sinks, then synchronously stop the collector."""
    global _handle, _is_primary, _resolution_failed
    from miles.dashboard import hooks

    hooks.detach_and_flush()
    if _handle is not None and _is_primary:
        import ray

        try:
            ray.get(_handle.shutdown.remote(), timeout=30)
        except Exception:
            logger.warning("dashboard collector shutdown incomplete", exc_info=True)
        try:
            ray.kill(_handle)
        except Exception:
            logger.warning("dashboard collector actor could not be killed", exc_info=True)
    _handle = None
    _is_primary = False
    _resolution_failed = False
