from __future__ import annotations

import hashlib
import inspect
import math
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy as TorchMixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group
from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatter
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _div_if_needed,
    _get_device_handle,
    _get_dim0_padded_size,
    _get_gradient_divide_factors,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    compiled_autograd_enabled,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.tensor import DTensor


_EXPECTED_TORCH_VERSION = "2.11.0+cu129"
_PATCH_SENTINEL = "_miles_param_dtype_map_patch_applied"
_SOURCE_HASHES = {
    "FSDPParam.init_dtype_attrs": "2cc968770804055cdde959db7cfa47b92a1137badcdaf2e448555741c2fd282c",
    "FSDPParamGroup.__init__": "323868f31033bb5696eaf30f278838901c0d6b7e0da031c517177707b716409b",
    "_get_param_all_gather_inputs": "a70fbae57b8aa3dc669d01a409d30c74428546bc4d69635a90cffe9380a5785f",
    "foreach_reduce": "46bcdaa0df40823e13922359db131cf329f8adf12eefa72ff16ceb41f9650d90",
    "foreach_reduce_scatter_copy_in": "559a065467abbaa578f348bd6a8c6478cfc8ee516e602227b795bc3f6727eb22",
}

_ORIGINAL_PARAM_GROUP_INIT = FSDPParamGroup.__init__
_ORIGINAL_INIT_DTYPE_ATTRS = FSDPParam.init_dtype_attrs
_ORIGINAL_GET_PARAM_ALL_GATHER_INPUTS = _fsdp_collectives._get_param_all_gather_inputs
_ORIGINAL_FOREACH_REDUCE = _fsdp_collectives.foreach_reduce
_ORIGINAL_FOREACH_REDUCE_SCATTER_COPY_IN = _fsdp_collectives.foreach_reduce_scatter_copy_in


@dataclass(frozen=True)
class ParamDtypeMixedPrecisionPolicy(TorchMixedPrecisionPolicy):
    param_dtype_map: Mapping[str, torch.dtype] | None = None


def _source_hash(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def _verify_source(name: str, fn) -> None:
    actual = _source_hash(fn)
    expected = _SOURCE_HASHES[name]
    if actual != expected:
        raise RuntimeError(
            f"Cannot apply the Miles FSDP patch: source hash for {name} changed "
            f"(expected {expected}, got {actual})"
        )


def _bind_to_collectives(fn, name: str, *, no_grad: bool):
    bound = types.FunctionType(
        fn.__code__,
        _fsdp_collectives.__dict__,
        name,
        fn.__defaults__,
        fn.__closure__,
    )
    bound.__kwdefaults__ = fn.__kwdefaults__
    bound.__annotations__ = fn.__annotations__
    bound.__module__ = _fsdp_collectives.__name__
    return torch.no_grad()(bound) if no_grad else bound


def _patched_param_group_init(
    self,
    params: list[nn.Parameter],
    modules: tuple[nn.Module, ...],
    mesh_info,
    post_forward_mesh_info,
    device: torch.device,
    shard_placement_fn,
    mp_policy: TorchMixedPrecisionPolicy,
    offload_policy,
) -> None:
    param_dtype_map = (
        mp_policy.param_dtype_map
        if isinstance(mp_policy, ParamDtypeMixedPrecisionPolicy)
        else None
    )
    if param_dtype_map is None:
        _ORIGINAL_PARAM_GROUP_INIT(
            self,
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            shard_placement_fn,
            mp_policy,
            offload_policy,
        )
        return

    # MILES_PATCH_BEGIN: fsdp-param-dtype-map
    managed_params = set(params)
    fqn_to_param: dict[str, nn.Parameter] = {}
    for module in modules:
        for fqn, param in module.named_parameters():
            if param not in managed_params:
                continue
            previous = fqn_to_param.get(fqn)
            if previous is not None and previous is not param:
                raise ValueError(
                    f"param_dtype_map FQN {fqn!r} is ambiguous across the fully_shard modules"
                )
            fqn_to_param[fqn] = param
    unknown_fqns = sorted(set(param_dtype_map).difference(fqn_to_param))
    if unknown_fqns:
        raise ValueError(
            "param_dtype_map contains FQNs that do not name a parameter managed "
            f"by this fully_shard call: {unknown_fqns}"
        )
    param_overrides = {
        fqn_to_param[fqn]: dtype for fqn, dtype in param_dtype_map.items()
    }
    effective_dtypes = {
        param_overrides.get(param, mp_policy.param_dtype) or param.dtype
        for param in params
        if param.requires_grad
    }
    if len(effective_dtypes) > 1 and mp_policy.reduce_dtype is None:
        raise ValueError("Mixed parameter dtypes require an explicit reduce_dtype")
    # MILES_PATCH_END: fsdp-param-dtype-map

    _ORIGINAL_PARAM_GROUP_INIT(
        self,
        params,
        modules,
        mesh_info,
        post_forward_mesh_info,
        device,
        shard_placement_fn,
        mp_policy,
        offload_policy,
    )

    # MILES_PATCH_BEGIN: fsdp-param-dtype-map
    for fsdp_param, param in zip(self.fsdp_params, params, strict=True):
        override = param_overrides.get(param)
        fsdp_param._param_dtype_override = override
        effective_param_dtype = (
            override if override is not None else mp_policy.param_dtype
        )
        fsdp_param.mp_policy = replace(
            mp_policy,
            param_dtype=effective_param_dtype,
            param_dtype_map=None,
        )
    # MILES_PATCH_END: fsdp-param-dtype-map


def _patched_init_dtype_attrs(
    self: FSDPParam,
    mp_policy: TorchMixedPrecisionPolicy,
) -> None:
    if not isinstance(mp_policy, ParamDtypeMixedPrecisionPolicy):
        _ORIGINAL_INIT_DTYPE_ATTRS(self, mp_policy)
        return

    # MILES_PATCH_BEGIN: fsdp-param-dtype-map
    param_dtype = (
        self._param_dtype_override
        if self._param_dtype_override is not None
        else mp_policy.param_dtype
    )
    reduce_dtype = mp_policy.reduce_dtype
    self.orig_dtype = self.sharded_param.dtype
    if mp_policy.param_dtype_map is None and reduce_dtype == param_dtype:
        reduce_dtype = None
    if param_dtype == self.orig_dtype:
        param_dtype = None
    self.param_dtype = param_dtype
    self.reduce_dtype = reduce_dtype
    # MILES_PATCH_END: fsdp-param-dtype-map


# Copied from PyTorch v2.11.0 at 70d99e998b4955e0049d13a98d77ae1b14db1f45.
def _patched_get_param_all_gather_inputs(
    fsdp_params: list[FSDPParam],
) -> list[list[torch.Tensor]]:
    if compiled_autograd_enabled():
        return [fsdp_param.all_gather_inputs for fsdp_param in fsdp_params]

    # Intentionally try to run a fast-path that bypasses abstractions for the
    # common FSDP case of bf16/fp32 mixed precision in order to use foreach
    # copy for lower CPU overhead and more efficient copying in eager
    def use_foreach_copy(fsdp_param: FSDPParam) -> bool:
        return (
            fsdp_param.param_dtype is not None
            and not fsdp_param.offload_to_cpu
            and not hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
        )

    param_all_gather_inputs: list[list[torch.Tensor]] = [[] for _ in fsdp_params]
    # MILES_PATCH_BEGIN: fsdp-param-dtype-map
    foreach_copy_infos: dict[
        torch.dtype, tuple[list[int], list[torch.Tensor], list[int]]
    ] = {}

    # 1st pass: for foreach-copy parameters, get inputs and metadata for the
    # foreach copy, and for the others, actually get their all-gather inputs
    for i, fsdp_param in enumerate(fsdp_params):
        if use_foreach_copy(fsdp_param):
            param_dtype = cast(torch.dtype, fsdp_param.param_dtype)
            indices, inputs, input_numels = foreach_copy_infos.setdefault(
                param_dtype, ([], [], [])
            )
            indices.append(i)
            all_gather_input = (
                fsdp_param._sharded_param_data
                if fsdp_param.sharded_state == ShardedState.SHARDED
                else cast(torch.Tensor, fsdp_param._sharded_post_forward_param_data)
            )
            inputs.append(all_gather_input)
            input_numels.append(all_gather_input.numel())
        else:
            param_all_gather_inputs[i] = fsdp_param.all_gather_inputs

    # 2nd pass: use foreach copy to compute the remaining all-gather inputs
    for param_dtype, (indices, inputs, input_numels) in foreach_copy_infos.items():
        device = fsdp_params[indices[0]].device
        flat_foreach_copy_input = torch.empty(
            (sum(input_numels),), device=device, dtype=param_dtype
        )
        splits = torch.split(flat_foreach_copy_input, input_numels)
        torch._foreach_copy_(splits, inputs)
        for i, split in zip(indices, splits):
            param_all_gather_inputs[i] = [split]
    # MILES_PATCH_END: fsdp-param-dtype-map

    return param_all_gather_inputs


# Copied from PyTorch v2.11.0 at 70d99e998b4955e0049d13a98d77ae1b14db1f45.
def _patched_foreach_reduce(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    reduce_scatter_comm: ReduceScatter,
    orig_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    device: torch.device,
    gradient_divide_factor: float | None,
    all_reduce_group: dist.ProcessGroup | None,
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: torch.Tensor | None,
    all_reduce_hook: Callable[[torch.Tensor], None] | None,
    force_sum_reduction_for_comms: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Event,
    torch.Event,
    torch.Tensor | None,
    torch.Event | None,
    torch.Tensor | None,
]:
    """
    ``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients.
    """

    # MILES_PATCH_BEGIN: fsdp-param-dtype-map
    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if reduce_dtype is None and len(grad_dtypes) != 1:
        _raise_assert_with_print(
            "FSDP reduce-scatter requires an explicit reduce dtype for mixed "
            f"gradient dtypes but got {grad_dtypes}"
        )
    reduce_dtype = reduce_dtype or unsharded_grads[0].dtype
    # MILES_PATCH_END: fsdp-param-dtype-map
    (predivide_factor, postdivide_factor, reduce_scatter_op, all_reduce_op) = (
        _get_gradient_divide_factors(
            reduce_scatter_group,
            all_reduce_group,
            reduce_dtype,
            device.type,
            gradient_divide_factor,
            force_sum_reduction_for_comms,
        )
    )

    if reduce_scatter_group is None:
        world_size = 1
    else:
        world_size = reduce_scatter_group.size()
    device_handle = _get_device_handle(device.type)
    current_stream = device_handle.current_stream()

    if world_size > 1:
        for i, (fsdp_param, unsharded_grad) in enumerate(
            zip(fsdp_params, unsharded_grads)
        ):
            if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
                continue
            if unsharded_grad.size(shard_dim) % world_size != 0:
                raise AssertionError(
                    f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
                )
            chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
            unsharded_grads[i] = torch.cat(chunks, dim=0)

    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    reduce_scatter_input = reduce_scatter_comm.allocate(
        (reduce_scatter_input_numel,),
        dtype=reduce_dtype,
        device=device,
    )

    foreach_reduce_scatter_copy_in(
        unsharded_grads, reduce_scatter_input, world_size
    )

    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)
    all_reduce_input = None
    all_reduce_event = None

    with device_handle.stream(reduce_scatter_stream):
        reduce_output = reduce_scatter_comm.allocate(
            (reduce_scatter_output_numel,),
            dtype=reduce_dtype,
            device=device,
        )
        _div_if_needed(reduce_scatter_input, predivide_factor)
        if world_size > 1:
            reduce_scatter_comm(
                output_tensor=reduce_output,
                input_tensor=reduce_scatter_input,
                group=reduce_scatter_group,
                op=reduce_scatter_op,
            )
        else:
            # For single GPU, just copy the input to output (no actual reduce-scatter needed), and
            # account for a possible gradient_divide_factor.
            if gradient_divide_factor is not None:
                reduce_output.copy_(
                    reduce_scatter_input / gradient_divide_factor
                )
            else:
                reduce_output.copy_(reduce_scatter_input)
        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        if all_reduce_group is not None:  # HSDP or DDP/replicate
            # Accumulations must run in the reduce-scatter stream
            if not all_reduce_grads:
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                return (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream.record_event(),
                    all_reduce_input,
                    all_reduce_event,
                    partial_reduce_output,
                )
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            if world_size >= 1:
                all_reduce_stream.wait_stream(reduce_scatter_stream)
            else:
                all_reduce_stream.wait_stream(current_stream)
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=all_reduce_op,
                )
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
    # -- END: ops in reduce_scatter stream

    if all_reduce_hook is not None:
        # Execute user-specified all reduce hook.
        # If native HSDP is used, this is executed after the HSDP all reduce.
        # If 1-d FSDP is used, this is executed post reduce-scatter.
        post_reduce_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with device_handle.stream(all_reduce_stream):
            all_reduce_hook(reduce_output)
    # -- END: ops post reduce_scatter

    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) if not
                # accumulating gradients since the CPU add kernel depends on
                # the copy result and we cannot run the add as a callback
                non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
                if non_blocking:
                    # Record an event on which to block the CPU thread to
                    # ensure that the D2H copy finishes before the optimizer
                    fsdp_param.grad_offload_event = (
                        post_reduce_stream.record_event()
                    )
            if to_accumulate_grad:
                if not isinstance(fsdp_param.sharded_param.grad, DTensor):
                    raise AssertionError(
                        f"Expected fsdp_param.sharded_param.grad to be DTensor, got {type(fsdp_param.sharded_param.grad)}"
                    )
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(
                    new_sharded_grad
                )
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            if not compiled_autograd_enabled():
                for hook in (
                    getattr(
                        fsdp_param.sharded_param,
                        "_post_accumulate_grad_hooks",
                        {},
                    )
                    or {}
                ).values():
                    hook(fsdp_param.sharded_param)
            padded_sharded_numel = (
                padded_unsharded_size.numel() // world_size
            )
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
    )


# Copied from PyTorch v2.11.0 at 70d99e998b4955e0049d13a98d77ae1b14db1f45.
def _patched_foreach_reduce_scatter_copy_in(
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
    # MILES_PATCH_BEGIN: fsdp-param-dtype-map
    if len({grad.dtype for grad in unsharded_grads}) == 1:
        torch.ops.fsdp.chunk_cat(
            unsharded_grads,
            dim=0,
            num_chunks=world_size,
            out=reduce_scatter_input,
        )
        return

    copy_infos: dict[
        torch.dtype, tuple[list[torch.Tensor], list[torch.Tensor]]
    ] = {}
    padding_views: list[torch.Tensor] = []
    output_offset = 0
    for grad in unsharded_grads:
        chunk_size = math.ceil(grad.size(0) / world_size)
        trailing_numel = grad.numel() // grad.size(0)
        padded_chunk_numel = chunk_size * trailing_numel
        destinations, sources = copy_infos.setdefault(grad.dtype, ([], []))
        for rank in range(world_size):
            destination = reduce_scatter_input[rank].narrow(
                0, output_offset, padded_chunk_numel
            )
            start = rank * chunk_size
            length = min(chunk_size, max(grad.size(0) - start, 0))
            if length > 0:
                source = grad.narrow(0, start, length).reshape(-1)
                destinations.append(
                    destination.narrow(0, 0, source.numel())
                )
                sources.append(source)
            if length < chunk_size:
                padding_start = length * trailing_numel
                padding_views.append(
                    destination.narrow(
                        0,
                        padding_start,
                        padded_chunk_numel - padding_start,
                    )
                )
        output_offset += padded_chunk_numel

    for destinations, sources in copy_infos.values():
        torch._foreach_copy_(destinations, sources)
    if padding_views:
        torch._foreach_zero_(padding_views)
    # MILES_PATCH_END: fsdp-param-dtype-map


def apply_param_dtype_map_patch() -> None:
    if getattr(_fsdp_collectives, _PATCH_SENTINEL, False):
        return
    if torch.__version__ != _EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            "The Miles FSDP param-dtype patch requires "
            f"torch=={_EXPECTED_TORCH_VERSION}, got {torch.__version__}"
        )

    _verify_source("FSDPParamGroup.__init__", _ORIGINAL_PARAM_GROUP_INIT)
    _verify_source("FSDPParam.init_dtype_attrs", _ORIGINAL_INIT_DTYPE_ATTRS)
    _verify_source(
        "_get_param_all_gather_inputs",
        _ORIGINAL_GET_PARAM_ALL_GATHER_INPUTS,
    )
    _verify_source("foreach_reduce", _ORIGINAL_FOREACH_REDUCE)
    _verify_source(
        "foreach_reduce_scatter_copy_in",
        _ORIGINAL_FOREACH_REDUCE_SCATTER_COPY_IN,
    )

    get_param_all_gather_inputs = _bind_to_collectives(
        _patched_get_param_all_gather_inputs,
        "_get_param_all_gather_inputs",
        no_grad=True,
    )
    foreach_reduce_scatter_copy_in = _bind_to_collectives(
        _patched_foreach_reduce_scatter_copy_in,
        "foreach_reduce_scatter_copy_in",
        no_grad=False,
    )
    foreach_reduce = _bind_to_collectives(
        _patched_foreach_reduce,
        "foreach_reduce",
        no_grad=True,
    )

    FSDPParamGroup.__init__ = _patched_param_group_init
    FSDPParam.init_dtype_attrs = _patched_init_dtype_attrs
    _fsdp_collectives._get_param_all_gather_inputs = get_param_all_gather_inputs
    _fsdp_collectives.foreach_reduce_scatter_copy_in = (
        foreach_reduce_scatter_copy_in
    )
    _fsdp_collectives.foreach_reduce = foreach_reduce
    _fsdp_param_group.foreach_reduce = foreach_reduce
    setattr(_fsdp_collectives, _PATCH_SENTINEL, True)


__all__ = [
    "ParamDtypeMixedPrecisionPolicy",
    "apply_param_dtype_map_patch",
]
