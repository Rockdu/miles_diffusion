# Ported from PyTorch v2.11.0 at 70d99e998b4955e0049d13a98d77ae1b14db1f45.
# ruff: noqa
# isort: skip_file
# fmt: off

import itertools
from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _div_if_needed,
    _get_gradient_divide_factors,
    DefaultAllGather,
    DefaultReduceScatter,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
)
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTestMultiThread,
    get_devtype,
)
from torch.testing._internal.common_utils import run_tests


device_type = torch.device(get_devtype())
device_module = torch.get_device_module(device_type)


class TestFullyShardCollectiveOps(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 128

    @property
    def device(self) -> torch.device:
        return torch.device(device_type.type, 0)

    def _get_param_sizes(self) -> list[torch.Size]:
        # For world size 128, the fp32 all-gather and reduce-scatter testing
        # requires ~0.22 GB
        return [
            torch.Size([17, 257]),
            torch.Size([17]),
            torch.Size([64, 312]),
            torch.Size([64]),
            torch.Size([64, 64]),
            torch.Size([512, 64]),
            torch.Size([256]),
            torch.Size([64, 297]),
        ]

    def _init_params(self, param_sizes: list[torch.Size]) -> list[nn.Parameter]:
        torch.manual_seed(42)
        orig_params = [
            nn.Parameter(torch.randn(size, device=self.device)) for size in param_sizes
        ]
        # Since seed is per process, not per thread, we broadcast to ensure the
        # same original parameters across ranks
        for orig_param in orig_params:
            dist.broadcast(orig_param, src=0)
        return orig_params

    def _init_fsdp_param_group(
        self, params: list[nn.Parameter], reshard_after_forward: Union[bool, int]
    ):
        module = nn.ParameterList([param.detach().clone() for param in params])
        mesh_info = FSDPMeshInfo(_init_default_fully_shard_mesh(), shard_mesh_dim=0)
        post_forward_mesh_info = _get_post_forward_mesh_info(
            reshard_after_forward, mesh_info
        )
        fsdp_param_group = FSDPParamGroup(
            list(module.parameters()),
            (module,),
            mesh_info,
            post_forward_mesh_info,
            self.device,
            None,  # shard_placement_fn
            MixedPrecisionPolicy(),
            OffloadPolicy(),
        )
        fsdp_param_group.lazy_init()
        return fsdp_param_group

    @skip_if_lt_x_gpu(1)
    def test_all_gather_fp32(self):
        param_sizes = self._get_param_sizes()
        default_stream = device_module.current_stream()
        stream1, stream2 = (
            device_module.Stream(),
            device_module.Stream(),
        )
        for async_op, streams, reshard_after_forward in itertools.product(
            (False, True),
            ((default_stream, default_stream), (stream1, stream2)),
            (True, 8),
        ):
            all_gather_copy_in_stream, all_gather_stream = streams
            # Save test time by only testing reshard after forward as an int
            # for non-async and non-default streams (like in pre-backward)
            if type(reshard_after_forward) is int and (
                async_op or all_gather_stream is default_stream
            ):
                continue
            self._test_all_gather(
                param_sizes,
                reshard_after_forward=reshard_after_forward,
                async_op=async_op,
                all_gather_copy_in_stream=all_gather_copy_in_stream,
                all_gather_stream=all_gather_stream,
            )

    def _test_all_gather(
        self,
        param_sizes: list[torch.Size],
        reshard_after_forward: Union[bool, int],
        async_op: bool,
        all_gather_copy_in_stream,
        all_gather_stream,
    ):
        def all_gather(fsdp_param_group: FSDPParamGroup, group: dist.ProcessGroup):
            all_gather_comm = DefaultAllGather()
            all_gather_result = foreach_all_gather(
                fsdp_param_group.fsdp_params,
                group,
                async_op=async_op,
                all_gather_copy_in_stream=all_gather_copy_in_stream,
                all_gather_stream=all_gather_stream,
                device=self.device,
                all_gather_comm=all_gather_comm,
            )
            foreach_all_gather_copy_out(all_gather_result, fsdp_params, group)
            # Transition to unsharded state to register unsharded parameters
            for fsdp_param in fsdp_param_group.fsdp_params:
                fsdp_param.init_unsharded_param()
            fsdp_param_group._to_unsharded()

        def check_all_gathered_params(
            orig_params: list[nn.Parameter], module: nn.Module
        ):
            for orig_param, param in zip(orig_params, module.parameters()):
                self.assertIsInstance(param, torch.Tensor)
                self.assertIsInstance(param, nn.Parameter)
                self.assertEqual(param, orig_param.to(param.dtype))

        # Set up the reference parameters and construct the FSDP group
        orig_params = self._init_params(param_sizes)
        fsdp_param_group = self._init_fsdp_param_group(
            orig_params, reshard_after_forward
        )
        fsdp_params = fsdp_param_group.fsdp_params
        module = fsdp_param_group.modules[0]

        # Sanity check that the parameter sharding is as expected
        for orig_param, param in zip(orig_params, module.parameters()):
            self.assertTrue(isinstance(param, DTensor))
            self.assertEqual(param.full_tensor(), orig_param)

        # Run the foreach all-gather (including copy-in and copy-out)
        all_gather(fsdp_param_group, fsdp_param_group.mesh_info.shard_process_group)

        # Check all-gather correctness
        check_all_gathered_params(orig_params, module)

        # For reshard after after forward as an int, further test emulating the
        # pre-backward all-gather
        if type(reshard_after_forward) is not int:
            return
        fsdp_param_group._to_sharded_post_forward()
        all_gather(
            fsdp_param_group,
            fsdp_param_group.post_forward_mesh_info.shard_process_group,
        )
        check_all_gathered_params(orig_params, module)

    @skip_if_lt_x_gpu(1)
    def test_reduce_scatter_fp32(self):
        param_sizes = self._get_param_sizes()
        default_stream = device_module.current_stream()
        stream = device_module.Stream()
        for reduce_scatter_stream in (default_stream, stream):
            self._test_reduce_scatter(
                param_sizes,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_dtype=torch.float32,
            )

    @skip_if_lt_x_gpu(1)
    def test_reduce_scatter_fp16(self):
        param_sizes = self._get_param_sizes()
        default_stream = torch.get_device_module(device_type).current_stream()
        stream = device_module.Stream()
        for reduce_scatter_stream in (default_stream, stream):
            self._test_reduce_scatter(
                param_sizes,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_dtype=torch.float16,
            )

    def _test_reduce_scatter(
        self,
        param_sizes: list[torch.Size],
        reduce_scatter_stream,
        reduce_scatter_dtype: torch.dtype,
    ):
        # Set up the reference parameters and construct the FSDP group
        orig_params = self._init_params(param_sizes)
        fsdp_param_group = self._init_fsdp_param_group(orig_params, True)
        fsdp_params = fsdp_param_group.fsdp_params
        fsdp_param_group.comm_ctx.lazy_init(self.device)

        # Run one unshard to initialize metadata
        fsdp_param_group.unshard()
        fsdp_param_group.wait_for_unshard()
        fsdp_param_group.reshard()

        # Run the foreach reduce-scatter (including copy-in and view-out)
        torch.manual_seed(42)
        unsharded_grads = [torch.ones_like(param) * self.rank for param in orig_params]
        group = fsdp_param_group.mesh_info.shard_process_group
        self.assertEqual(group.size(), self.world_size)
        all_reduce_stream = device_module.Stream()
        comm = DefaultReduceScatter()
        (
            _,
            _,
            post_reduce_event,
            _,
            _,
            _,
        ) = foreach_reduce(
            fsdp_params,
            unsharded_grads,
            group,
            reduce_scatter_stream,
            comm,
            orig_dtype=orig_params[0].dtype,
            reduce_dtype=reduce_scatter_dtype,
            device=self.device,
            gradient_divide_factor=None,
            all_reduce_group=None,
            all_reduce_stream=all_reduce_stream,
            all_reduce_hook=None,
            all_reduce_grads=True,
            partial_reduce_output=None,
        )
        torch.get_device_module(device_type).current_stream().wait_event(
            post_reduce_event
        )

        # Check reduce-scatter correctness
        (
            predivide_factor,
            postdivide_factor,
            _,
            all_reduce_op,
        ) = _get_gradient_divide_factors(group, None, reduce_scatter_dtype)
        reduced_grads = [grad.detach().clone() for grad in unsharded_grads]
        for grad in reduced_grads:
            _div_if_needed(grad, predivide_factor)
            dist.all_reduce(
                grad,
                group=group,
                op=all_reduce_op,
            )
            _div_if_needed(grad, postdivide_factor)
        for fsdp_param, reduced_grad in zip(fsdp_params, reduced_grads):
            sharded_grad = fsdp_param.sharded_param.grad
            self.assertIsInstance(sharded_grad, DTensor)
            self.assertEqual(sharded_grad.full_tensor(), reduced_grad)


if __name__ == "__main__":
    run_tests()
