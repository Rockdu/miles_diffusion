"""Named tensor snapshots: share fixed groups and own mutable storage."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import count

import torch


class _GroupSnapshot:
    def __init__(self, version, tensors, *, fixed=False):
        self.version = version
        self.fixed = fixed
        self.replace_storage(tensors)

    def replace_storage(self, tensors):
        self.tensors = tensors
        self.devices = frozenset(tensor.device for tensor in tensors.values())
        self.is_pinned = all(tensor.is_pinned() for tensor in tensors.values())


@dataclass
class _EmaState:
    tensor_names: tuple[str, ...]
    initial_decay: float
    decay_ramp: float
    max_decay: float
    flat_steps: int
    update_count: int = 0

    def decay_at(self, update_count):
        if update_count <= self.flat_steps:
            return self.initial_decay
        return min((update_count - self.flat_steps) * self.decay_ramp, self.max_decay)


class TensorBackuper:
    """Keep fixed tensor schemas and CUDA tensors on the worker's current device.

    Callers mark external writes and abort the owning worker after a copy failure.
    Host transfers finish before returning; same-device EMA stays stream-ordered.
    """

    @staticmethod
    def create(source_getter, *, groups=None):
        return TensorBackuper(source_getter, groups=groups)

    def __init__(
        self,
        get_named_tensors: Callable[[Iterable[str] | None], Mapping[str, torch.Tensor]],
        *,
        groups: Mapping[str, Iterable[str]] | None = None,
    ):
        self._get_named_tensors = get_named_tensors
        self._tensor_names = tuple(get_named_tensors(None))
        self._groups = (
            {"base": self._tensor_names}
            if groups is None
            else {group: tuple(names) for group, names in groups.items()}
        )
        self._snapshots: dict[str, dict[str, _GroupSnapshot]] = {}
        self._active_weight_versions = dict.fromkeys(self._groups)
        self._next_version = count()
        self.ema_states: dict[str, _EmaState] = {}

    @property
    def backup_tags(self):
        return tuple(self._snapshots)

    @staticmethod
    def _matches_storage(snapshot, device, pin_memory):
        return (device is None or snapshot.devices.issubset({device})) and (not pin_memory or snapshot.is_pinned)

    @torch.no_grad()
    def backup(self, tag, *, device=None, pin_memory=False, fixed_groups=(), reuse=None):
        if tag in self.ema_states:
            raise ValueError(f"Cannot overwrite EMA tag {tag!r}")
        device = torch.device(device) if device is not None else None
        fixed_groups = set(fixed_groups)
        snapshots = {group: self._snapshots[source][group] for group, source in (reuse or {}).items()}
        for group, snapshot in snapshots.items():
            if not snapshot.fixed:
                raise ValueError(f"Only fixed groups can share storage: {group}")
            if self._active_weight_versions[group] != snapshot.version:
                raise ValueError(f"Live weights changed: {group}")
        previous = self._snapshots.get(tag, {})
        copied = False
        try:
            for group, names in self._groups.items():
                if group in snapshots:
                    continue
                saved = previous.get(group)
                version = self._active_weight_versions[group]
                fixed = group in fixed_groups
                same_storage = saved is not None and self._matches_storage(saved, device, pin_memory)
                if same_storage and saved.version == version and saved.fixed == fixed:
                    snapshots[group] = saved
                    continue
                live = self._get_named_tensors(names)
                if not same_storage or saved.fixed or fixed:
                    saved = _GroupSnapshot(
                        version,
                        {
                            name: torch.empty_like(tensor, device=device or tensor.device, pin_memory=pin_memory)
                            for name, tensor in live.items()
                        },
                        fixed=fixed,
                    )
                copied |= bool(names)
                for name, tensor in live.items():
                    saved.tensors[name].copy_(tensor, non_blocking=True)
                saved.version = next(self._next_version) if version is None else version
                snapshots[group] = saved
        finally:
            if copied and torch.cuda.is_available():
                torch.cuda.synchronize()
        self._snapshots[tag] = snapshots
        self._active_weight_versions.update({group: saved.version for group, saved in snapshots.items()})

    def get(self, tag):
        """Borrow read-only tensors; copy a tag to retain its mutable weights."""
        return {name: tensor for saved in self._snapshots[tag].values() for name, tensor in saved.tensors.items()}

    @torch.no_grad()
    def copy(self, *, src_tag, dst_tag):
        if dst_tag in self.ema_states:
            raise ValueError(f"Cannot overwrite EMA tag {dst_tag!r}")
        snapshots = {}
        for group, saved in self._snapshots[src_tag].items():
            if saved.fixed:
                snapshots[group] = saved
            else:
                tensors = {
                    name: torch.empty_like(tensor, pin_memory=tensor.is_pinned())
                    for name, tensor in saved.tensors.items()
                }
                for name, tensor in saved.tensors.items():
                    tensors[name].copy_(tensor, non_blocking=True)
                snapshots[group] = _GroupSnapshot(saved.version, tensors)
        self._snapshots[dst_tag] = snapshots

    @torch.no_grad()
    def restore(self, tag, *, groups=None):
        snapshots = self._snapshots[tag]
        selected = self._groups if groups is None else groups
        changed = tuple(group for group in selected if self._active_weight_versions[group] != snapshots[group].version)
        if not changed:
            return ()
        live = self._get_named_tensors(name for group in changed for name in self._groups[group])
        self.mark_weights_updated(changed)
        try:
            for group in changed:
                for name, tensor in snapshots[group].tensors.items():
                    live[name].copy_(tensor, non_blocking=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        self._active_weight_versions.update({group: snapshots[group].version for group in changed})
        return changed

    def mark_weights_updated(self, groups=None):
        for group in self._groups if groups is None else groups:
            self._active_weight_versions[group] = None

    def share_live_storage(self, tag):
        """Adopt the live tensor storage after the caller rebinds it."""
        live = self._get_named_tensors(None)
        for group, snapshot in self._snapshots[tag].items():
            snapshot.replace_storage({name: live[name].detach() for name in self._groups[group]})

    def release(self, tag):
        del self._snapshots[tag]
        self.ema_states.pop(tag, None)

    def configure_ema(
        self, tag, *, tensor_names=None, initial_decay=0.001, decay_ramp=0.001, max_decay=0.5, flat_steps=0
    ):
        if tag in self.ema_states:
            raise ValueError(f"EMA is already configured for {tag!r}")
        names = self._tensor_names if tensor_names is None else tuple(tensor_names)
        selected_names = set(names)
        for group, members in self._groups.items():
            if not selected_names.isdisjoint(members) and self._snapshots[tag][group].fixed:
                raise ValueError(f"EMA cannot update a fixed group: {group}")
        self.ema_states[tag] = _EmaState(names, initial_decay, decay_ramp, max_decay, flat_steps)

    @torch.no_grad()
    def update_ema(self, tag):
        state = self.ema_states[tag]
        decay = state.decay_at(state.update_count + 1)
        live = self._get_named_tensors(state.tensor_names)
        shadows = self.get(tag)
        for group, members in self._groups.items():
            if not live.keys().isdisjoint(members):
                self._snapshots[tag][group].version = next(self._next_version)
        has_cuda_transfer = False
        try:
            for name, source in live.items():
                shadow = shadows[name]
                if shadow.device.type == "cpu" and source.is_cuda:
                    has_cuda_transfer = True
                    gpu_shadow = shadow.to(source.device, non_blocking=True)
                    gpu_shadow.mul_(decay).add_(source, alpha=1.0 - decay)
                    shadow.copy_(gpu_shadow, non_blocking=True)
                    del gpu_shadow
                else:
                    if source.device != shadow.device:
                        has_cuda_transfer |= source.is_cuda or shadow.is_cuda
                        source = source.to(shadow.device, non_blocking=True)
                    shadow.mul_(decay).add_(source, alpha=1.0 - decay)
        finally:
            if has_cuda_transfer:
                torch.cuda.synchronize()
        state.update_count += 1
        return decay
