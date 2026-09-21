"""Named tensor snapshots: share fixed groups and own mutable storage."""

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor


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
    copy_tensor_names: tuple[str, ...]
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

    def __init__(
        self,
        active_models: Mapping[str, torch.nn.Module],
        *,
        groups: Mapping[str, Iterable[str]] | None = None,
    ):
        self._active_models = active_models
        self.active_model_parameters = {}
        self._active_model_buffers = {}
        model_tensor_groups = {}
        for component_name, component_model in active_models.items():
            component_prefix = f"{component_name}." if component_name else ""
            model_tensor_groups[f"{component_prefix}base"] = []
            model_tensor_groups[f"{component_prefix}lora"] = []
            # FSDP may expose gathered Parameters after forward; retain the registered shards.
            for parameter_name, parameter in component_model.named_parameters():
                qualified_name = f"{component_prefix}{parameter_name}"
                self.active_model_parameters[qualified_name] = parameter
                parameter_path = f".{parameter_name}"
                is_lora_parameter = ".lora_A." in parameter_path or ".lora_B." in parameter_path
                group_name = f"{component_prefix}{'lora' if is_lora_parameter else 'base'}"
                model_tensor_groups[group_name].append(qualified_name)
            for buffer_name, _ in component_model.named_buffers(remove_duplicate=False):
                qualified_name = f"{component_prefix}{buffer_name}"
                self._active_model_buffers[qualified_name] = (component_model, buffer_name)
                model_tensor_groups.setdefault(f"{component_prefix}buffers", []).append(qualified_name)
        self._tensor_groups = {
            group_name: tuple(tensor_names)
            for group_name, tensor_names in (model_tensor_groups if groups is None else groups).items()
        }
        self.buffer_names = tuple(self._active_model_buffers)
        self.buffer_groups = tuple(
            group_name
            for group_name, tensor_names in self._tensor_groups.items()
            if not self._active_model_buffers.keys().isdisjoint(tensor_names)
        )
        self.trainable_parameter_names = tuple(
            name for name, parameter in self.active_model_parameters.items() if parameter.requires_grad
        )
        trainable_parameter_names = set(self.trainable_parameter_names)
        self.trainable_tensor_groups = {
            group_name: tensor_names
            for group_name, tensor_names in self._tensor_groups.items()
            if not trainable_parameter_names.isdisjoint(tensor_names)
        }
        self.frozen_tensor_groups = tuple(
            group_name
            for group_name in self._tensor_groups
            if group_name not in self.trainable_tensor_groups and group_name not in self.buffer_groups
        )
        self._tensor_names = tuple(
            tensor_name for tensor_names in self._tensor_groups.values() for tensor_name in tensor_names
        )
        self._snapshots: dict[str, dict[str, _GroupSnapshot]] = {}
        self._active_model_group_versions = dict.fromkeys(self._tensor_groups)
        self._next_group_version = count()
        self.ema_states: dict[str, _EmaState] = {}

    def _get_active_model_local_tensors(self, tensor_names):
        # Fresh .data views preserve version counters for pending actor backward.
        selected_tensor_names = self._tensor_names if tensor_names is None else tensor_names
        local_tensors = {}
        for tensor_name in selected_tensor_names:
            if tensor_name in self.active_model_parameters:
                model_tensor = self.active_model_parameters[tensor_name].data
            else:
                buffer_owner, buffer_name = self._active_model_buffers[tensor_name]
                model_tensor = buffer_owner.get_buffer(buffer_name).data
            local_tensors[tensor_name] = model_tensor.to_local() if isinstance(model_tensor, DTensor) else model_tensor
        return local_tensors

    @property
    def backup_tags(self):
        return tuple(self._snapshots)

    @staticmethod
    def _storage_matches_request(group_snapshot, device, pin_memory):
        return (device is None or group_snapshot.devices.issubset({device})) and (
            not pin_memory or group_snapshot.is_pinned
        )

    @torch.no_grad()
    def backup(self, tag, *, groups=None, device=None, pin_memory=False, fixed_groups=(), reuse=None):
        """Copy selected live groups and reuse requested fixed snapshots."""
        if tag in self.ema_states:
            raise ValueError(f"Cannot overwrite EMA tag {tag!r}")
        device = torch.device(device) if device is not None else None
        fixed_groups = set(fixed_groups)
        snapshot_groups = {
            group_name: self._snapshots[source_tag][group_name] for group_name, source_tag in (reuse or {}).items()
        }
        for group_name, group_snapshot in snapshot_groups.items():
            if not group_snapshot.fixed:
                raise ValueError(f"Only fixed groups can share storage: {group_name}")
            if self._active_model_group_versions[group_name] != group_snapshot.version:
                raise ValueError(f"Live weights changed: {group_name}")
        previous_snapshot_groups = self._snapshots.get(tag, {})
        selected_groups = self._tensor_groups if groups is None else groups
        needs_copy_synchronization = False
        try:
            for group_name in selected_groups:
                tensor_names = self._tensor_groups[group_name]
                if group_name in snapshot_groups:
                    continue
                group_snapshot = previous_snapshot_groups.get(group_name)
                active_group_version = self._active_model_group_versions[group_name]
                group_is_fixed = group_name in fixed_groups
                storage_matches_request = group_snapshot is not None and self._storage_matches_request(
                    group_snapshot, device, pin_memory
                )
                snapshot_is_current = (
                    storage_matches_request
                    and group_snapshot.version == active_group_version
                    and group_snapshot.fixed == group_is_fixed
                )
                if snapshot_is_current:
                    snapshot_groups[group_name] = group_snapshot
                    continue
                active_model_tensors = self._get_active_model_local_tensors(tensor_names)
                can_reuse_snapshot_storage = (
                    storage_matches_request and not group_snapshot.fixed and not group_is_fixed
                )
                if not can_reuse_snapshot_storage:
                    group_snapshot = _GroupSnapshot(
                        active_group_version,
                        {
                            tensor_name: torch.empty_like(
                                active_tensor, device=device or active_tensor.device, pin_memory=pin_memory
                            )
                            for tensor_name, active_tensor in active_model_tensors.items()
                        },
                        fixed=group_is_fixed,
                    )
                needs_copy_synchronization |= bool(tensor_names)
                for tensor_name, active_tensor in active_model_tensors.items():
                    group_snapshot.tensors[tensor_name].copy_(active_tensor, non_blocking=True)
                group_snapshot.version = (
                    next(self._next_group_version) if active_group_version is None else active_group_version
                )
                snapshot_groups[group_name] = group_snapshot
        finally:
            if needs_copy_synchronization and torch.cuda.is_available():
                torch.cuda.synchronize()
        self._snapshots[tag] = snapshot_groups
        self._active_model_group_versions.update(
            {group_name: group_snapshot.version for group_name, group_snapshot in snapshot_groups.items()}
        )

    def backup_active_model(self, tag, *, device="cpu"):
        self.mark_buffers_updated()
        self.backup(
            tag,
            device=device,
            pin_memory=torch.device(device).type == "cpu" and torch.cuda.is_available(),
            fixed_groups=self.frozen_tensor_groups,
        )

    def initialize_ema(
        self,
        tag,
        *,
        share_frozen_from,
        device,
        initial_decay,
        decay_ramp,
        max_decay,
        flat_steps,
    ):
        self.backup(
            tag,
            device=device,
            pin_memory=torch.device(device).type == "cpu" and torch.cuda.is_available(),
            reuse={group_name: share_frozen_from for group_name in self.frozen_tensor_groups},
            fixed_groups=self.frozen_tensor_groups,
        )
        self.configure_ema(
            tag,
            tensor_names=self.trainable_parameter_names,
            copy_tensor_names=self.buffer_names,
            initial_decay=initial_decay,
            decay_ramp=decay_ramp,
            max_decay=max_decay,
            flat_steps=flat_steps,
        )

    def get(self, tag):
        """Borrow read-only tensors; copy a tag to retain its mutable weights."""
        return {
            tensor_name: snapshot_tensor
            for group_snapshot in self._snapshots[tag].values()
            for tensor_name, snapshot_tensor in group_snapshot.tensors.items()
        }

    @torch.no_grad()
    def copy(self, *, src_tag, dst_tag, groups=None):
        if dst_tag in self.ema_states:
            raise ValueError(f"Cannot overwrite EMA tag {dst_tag!r}")
        destination_snapshot_groups = {}
        source_snapshot_groups = self._snapshots[src_tag]
        selected_groups = source_snapshot_groups if groups is None else groups
        for group_name in selected_groups:
            source_snapshot = source_snapshot_groups[group_name]
            if source_snapshot.fixed:
                destination_snapshot_groups[group_name] = source_snapshot
            else:
                copied_tensors = {
                    tensor_name: torch.empty_like(source_tensor, pin_memory=source_tensor.is_pinned())
                    for tensor_name, source_tensor in source_snapshot.tensors.items()
                }
                for tensor_name, source_tensor in source_snapshot.tensors.items():
                    copied_tensors[tensor_name].copy_(source_tensor, non_blocking=True)
                destination_snapshot_groups[group_name] = _GroupSnapshot(source_snapshot.version, copied_tensors)
        self._snapshots[dst_tag] = destination_snapshot_groups

    @torch.no_grad()
    def restore(self, tag, *, groups=None):
        target_snapshot_groups = self._snapshots[tag]
        target_groups = target_snapshot_groups if groups is None else groups
        groups_to_restore = tuple(
            group_name
            for group_name in target_groups
            if self._active_model_group_versions[group_name] != target_snapshot_groups[group_name].version
        )
        if not groups_to_restore:
            return ()
        active_model_tensors = self._get_active_model_local_tensors(
            tensor_name for group_name in groups_to_restore for tensor_name in self._tensor_groups[group_name]
        )
        self.mark_weights_updated(groups_to_restore)
        try:
            for group_name in groups_to_restore:
                for tensor_name, snapshot_tensor in target_snapshot_groups[group_name].tensors.items():
                    active_model_tensors[tensor_name].copy_(snapshot_tensor, non_blocking=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        self._active_model_group_versions.update(
            {group_name: target_snapshot_groups[group_name].version for group_name in groups_to_restore}
        )
        return groups_to_restore

    def mark_weights_updated(self, groups=None):
        selected_groups = self._tensor_groups if groups is None else groups
        for group_name in selected_groups:
            self._active_model_group_versions[group_name] = None

    def mark_parameters_with_grad_updated(self):
        parameter_names_with_grad = {
            name for name, parameter in self.active_model_parameters.items() if parameter.grad is not None
        }
        self.mark_weights_updated(
            group_name
            for group_name, tensor_names in self._tensor_groups.items()
            if not parameter_names_with_grad.isdisjoint(tensor_names)
        )

    def mark_buffers_updated(self):
        self.mark_weights_updated(self.buffer_groups)

    def restore_trainable_parameter_flags(self):
        # PEFT may re-enable gathered parameters while leaving their shards frozen.
        trainable_parameter_names = set(self.trainable_parameter_names)
        for name, parameter in self.active_model_parameters.items():
            parameter.requires_grad_(name in trainable_parameter_names)

    @contextmanager
    def use_temporary_buffers(self):
        buffer_bindings = [
            (module, buffer_name, buffer)
            for model in self._active_models.values()
            for module in model.modules()
            for buffer_name, buffer in module.named_buffers(recurse=False, remove_duplicate=False)
        ]
        # Reference writes must not change buffers saved by the pending backward.
        unique_buffers = {id(buffer): buffer for _, _, buffer in buffer_bindings}
        buffer_copies = {buffer_id: buffer.detach().clone() for buffer_id, buffer in unique_buffers.items()}
        try:
            for module, buffer_name, buffer in buffer_bindings:
                setattr(module, buffer_name, buffer_copies[id(buffer)])
            yield
        finally:
            for module, buffer_name, buffer in buffer_bindings:
                setattr(module, buffer_name, buffer)
            self.mark_buffers_updated()

    def bind_active_model_to_cpu_snapshot(self, tag):
        cpu_snapshot_tensors = self.get(tag)
        for component_name, component_model in self._active_models.items():
            bind_fsdp_model_to_cpu_snapshot(
                component_model,
                cpu_snapshot_tensors,
                prefix=component_name,
                pin_memory=torch.cuda.is_available(),
            )
        self.active_model_cpu_storage_as_snapshot(
            tag,
            groups=tuple(group_name for group_name in self._snapshots[tag] if group_name not in self.buffer_groups),
        )

    def active_model_cpu_storage_as_snapshot(self, tag, *, groups=None):
        """Call after the active model has been bound to CPU storage."""
        active_model_tensors = self._get_active_model_local_tensors(None)
        snapshot_groups = self._snapshots[tag]
        selected_groups = snapshot_groups if groups is None else groups
        for group_name in selected_groups:
            group_snapshot = snapshot_groups[group_name]
            group_snapshot.replace_storage(
                {
                    tensor_name: active_model_tensors[tensor_name].detach()
                    for tensor_name in self._tensor_groups[group_name]
                }
            )

    def release(self, tag):
        del self._snapshots[tag]
        self.ema_states.pop(tag, None)

    def configure_ema(
        self,
        tag,
        *,
        tensor_names=None,
        copy_tensor_names=(),
        initial_decay=0.001,
        decay_ramp=0.001,
        max_decay=0.5,
        flat_steps=0,
    ):
        if tag in self.ema_states:
            raise ValueError(f"EMA is already configured for {tag!r}")
        ema_tensor_names = self._tensor_names if tensor_names is None else tuple(tensor_names)
        copied_tensor_names = tuple(copy_tensor_names)
        updated_tensor_names = set(ema_tensor_names) | set(copied_tensor_names)
        for group_name, group_tensor_names in self._tensor_groups.items():
            if not updated_tensor_names.isdisjoint(group_tensor_names) and self._snapshots[tag][group_name].fixed:
                raise ValueError(f"EMA cannot update a fixed group: {group_name}")
        self.ema_states[tag] = _EmaState(
            tensor_names=ema_tensor_names,
            copy_tensor_names=copied_tensor_names,
            initial_decay=initial_decay,
            decay_ramp=decay_ramp,
            max_decay=max_decay,
            flat_steps=flat_steps,
        )

    @torch.no_grad()
    def update_ema(self, tag):
        ema_state = self.ema_states[tag]
        decay = ema_state.decay_at(ema_state.update_count + 1)
        active_model_tensors = self._get_active_model_local_tensors(
            (*ema_state.tensor_names, *ema_state.copy_tensor_names)
        )
        copy_tensor_names = set(ema_state.copy_tensor_names)
        ema_tensors = self.get(tag)
        for group_name, group_tensor_names in self._tensor_groups.items():
            if not active_model_tensors.keys().isdisjoint(group_tensor_names):
                self._snapshots[tag][group_name].version = next(self._next_group_version)
        needs_transfer_synchronization = False
        try:
            for tensor_name, active_tensor in active_model_tensors.items():
                ema_tensor = ema_tensors[tensor_name]
                if tensor_name in self._active_model_buffers or tensor_name in copy_tensor_names:
                    needs_transfer_synchronization |= active_tensor.device != ema_tensor.device and (
                        active_tensor.is_cuda or ema_tensor.is_cuda
                    )
                    ema_tensor.copy_(active_tensor, non_blocking=True)
                elif ema_tensor.device.type == "cpu" and active_tensor.is_cuda:
                    needs_transfer_synchronization = True
                    gpu_ema_tensor = ema_tensor.to(active_tensor.device, non_blocking=True)
                    gpu_ema_tensor.mul_(decay).add_(active_tensor, alpha=1.0 - decay)
                    ema_tensor.copy_(gpu_ema_tensor, non_blocking=True)
                    del gpu_ema_tensor
                else:
                    if active_tensor.device != ema_tensor.device:
                        needs_transfer_synchronization |= active_tensor.is_cuda or ema_tensor.is_cuda
                        active_tensor = active_tensor.to(ema_tensor.device, non_blocking=True)
                    ema_tensor.mul_(decay).add_(active_tensor, alpha=1.0 - decay)
        finally:
            if needs_transfer_synchronization:
                torch.cuda.synchronize()
        ema_state.update_count += 1
        return decay


def bind_fsdp_model_to_cpu_snapshot(
    model: torch.nn.Module,
    cpu_snapshot_tensors: Mapping[str, torch.Tensor],
    *,
    prefix: str = "",
    pin_memory: bool,
) -> None:
    """Preserve Parameter bindings while moving their storage to CPU weights.

    Call only between completed training steps. FSDP can replace uneven shard
    storage while padding it; the caller must adopt the resulting local tensors.
    """
    named_tensors = dict(model.named_parameters(prefix=prefix))
    named_tensors.update(model.named_buffers(prefix=prefix, remove_duplicate=False))
    cpu_snapshot_by_tensor_id = {}
    for name, tensor in named_tensors.items():
        if name in cpu_snapshot_tensors:
            cpu_snapshot_by_tensor_id[id(tensor)] = cpu_snapshot_tensors[name]

    fsdp_parameters = []
    for module in model.modules():
        if isinstance(module, FSDPModule):
            fsdp_parameter_group = module._get_fsdp_state()._fsdp_param_group
            if fsdp_parameter_group is not None:
                fsdp_parameters.extend(fsdp_parameter_group.fsdp_params)

    def use_cpu_snapshot_storage(tensor: torch.Tensor) -> torch.Tensor:
        snapshot_tensor = cpu_snapshot_by_tensor_id.get(id(tensor))
        if snapshot_tensor is None:
            return tensor.cpu()
        if isinstance(tensor, DTensor):
            # from_local would move CPU weights back to the CUDA mesh device.
            return DTensor(snapshot_tensor, tensor._spec, requires_grad=tensor.requires_grad)
        return snapshot_tensor

    original_pin_memory_settings = [parameter.pin_memory for parameter in fsdp_parameters]
    try:
        # Let FSDP pin any storage it allocates while padding uneven shards.
        for parameter in fsdp_parameters:
            parameter.pin_memory = pin_memory
        model._apply(use_cpu_snapshot_storage)
    finally:
        for parameter, original_pin_memory in zip(fsdp_parameters, original_pin_memory_settings, strict=True):
            parameter.pin_memory = original_pin_memory
