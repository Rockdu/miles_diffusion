from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FSDPParallelPlan:
    no_split_modules: tuple[str, ...] | None = None
    param_dtype_patterns: Mapping[str, str] = field(default_factory=dict)


__all__ = ["FSDPParallelPlan"]
