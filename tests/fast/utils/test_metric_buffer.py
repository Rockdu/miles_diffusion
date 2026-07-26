"""CPU unit tests for MetricBuffer's single-rank accumulation."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.utils.metric_buffer import MetricBuffer, MetricReduce


def _buffer(**schema) -> MetricBuffer:
    return MetricBuffer(group=None, device=torch.device("cpu"), schema=schema)


def test_mean_weights_items_not_calls():
    """A 3-item micro-batch must outweigh a 1-item one: (6+1)/(3+1), not (2+1)/2."""
    metrics = _buffer(loss=MetricReduce.MEAN)
    metrics.add("loss", torch.tensor(6.0), 3)
    metrics.add("loss", torch.tensor(1.0), 1)
    assert metrics.reduce() == {"loss": 7.0 / 4.0}


def test_max_keeps_the_largest():
    metrics = _buffer(abs_diff=MetricReduce.MAX)
    metrics.add("abs_diff", torch.tensor(0.25))
    metrics.add("abs_diff", torch.tensor(2.5))
    metrics.add("abs_diff", torch.tensor(1.0))
    assert metrics.reduce() == {"abs_diff": 2.5}


def test_replicated_passes_the_value_through():
    metrics = _buffer(grad_norm=MetricReduce.REPLICATED)
    metrics.add("grad_norm", torch.tensor(0.125))
    assert metrics.reduce() == {"grad_norm": 0.125}


def test_declared_but_never_recorded_metrics_are_dropped():
    """A phase no micro-batch hit, or a debug tensor the rollout did not emit."""
    metrics = _buffer(loss=MetricReduce.MEAN, unused_mean=MetricReduce.MEAN, unused_max=MetricReduce.MAX)
    metrics.add("loss", torch.tensor(4.0), 2)
    assert metrics.reduce() == {"loss": 2.0}


def test_zero_count_contribution_does_not_create_a_datapoint():
    metrics = _buffer(loss=MetricReduce.MEAN)
    metrics.add("loss", torch.tensor(0.0), 0)
    assert metrics.reduce() == {}


def test_rejects_undeclared_metric():
    """The schema is what keeps the reduction layout aligned across ranks."""
    metrics = _buffer(loss=MetricReduce.MEAN)
    with pytest.raises(KeyError):
        metrics.add("typo", torch.tensor(1.0), 1)


def test_rejects_mean_without_an_item_count():
    """Defaulting the count would silently weight ranks instead of items."""
    metrics = _buffer(loss=MetricReduce.MEAN)
    with pytest.raises(ValueError, match="item count"):
        metrics.add("loss", torch.tensor(1.0))


def test_rejects_non_scalar_values():
    """Silently reducing a vector would hide a wrong metric expression."""
    metrics = _buffer(loss=MetricReduce.MEAN)
    with pytest.raises(RuntimeError):
        metrics.add("loss", torch.zeros(4), 4)


def test_accumulates_in_float64_so_long_runs_do_not_drift():
    """Summing 4096 float32 values in float32 drifts; in float64 it stays exact."""
    value = torch.tensor(0.1, dtype=torch.float32)
    metrics = _buffer(loss=MetricReduce.MEAN)
    for _ in range(4096):
        metrics.add("loss", value, 1)
    assert metrics.reduce()["loss"] == value.item()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
