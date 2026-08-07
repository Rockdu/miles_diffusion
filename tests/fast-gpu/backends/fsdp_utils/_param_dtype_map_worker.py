import gc
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2VideoTransformerBlock,
)
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from miles.backends.fsdp_utils import fsdp_param_dtype_patch


MODEL_NAMES = ("wan2_2", "ltx2_3")
TOPOLOGIES = ("fully_shard_1x4", "hybrid_shard_2x2")
MODEL_SEED = 42
INPUT_SEED = 123


@dataclass
class RunResult:
    outputs: tuple[torch.Tensor, ...]
    grads: dict[str, torch.Tensor | None]


def _create_wan_case(rank):
    torch.manual_seed(MODEL_SEED)
    model = WanTransformerBlock(
        dim=64,
        ffn_dim=96,
        num_heads=4,
        cross_attn_norm=True,
        added_kv_proj_dim=64,
    ).cuda()
    torch.manual_seed(INPUT_SEED + rank)
    inputs = (
        torch.randn(2, 7, 64, device="cuda"),
        torch.randn(2, 5, 64, device="cuda"),
        torch.randn(2, 6, 64, device="cuda"),
        None,
    )
    return model, inputs


def _create_ltx_case(rank):
    torch.manual_seed(MODEL_SEED)
    model = LTX2VideoTransformerBlock(
        dim=64,
        num_attention_heads=4,
        attention_head_dim=16,
        cross_attention_dim=64,
        audio_dim=32,
        audio_num_attention_heads=4,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=32,
    ).cuda()
    torch.manual_seed(INPUT_SEED + rank)
    inputs = (
        torch.randn(2, 7, 64, device="cuda"),
        torch.randn(2, 5, 32, device="cuda"),
        torch.randn(2, 3, 64, device="cuda"),
        torch.randn(2, 4, 32, device="cuda"),
        torch.randn(2, 1, 6 * 64, device="cuda"),
        torch.randn(2, 1, 6 * 32, device="cuda"),
        torch.randn(2, 1, 4 * 64, device="cuda"),
        torch.randn(2, 1, 4 * 32, device="cuda"),
        torch.randn(2, 1, 64, device="cuda"),
        torch.randn(2, 1, 32, device="cuda"),
    )
    return model, inputs


def _create_case(model_name, rank):
    if model_name == "wan2_2":
        return _create_wan_case(rank)
    if model_name == "ltx2_3":
        return _create_ltx_case(rank)
    raise AssertionError(f"Unknown model {model_name}")


def _create_mesh(topology):
    if topology == "fully_shard_1x4":
        return init_device_mesh(
            "cuda",
            (4,),
            mesh_dim_names=("dp_shard",),
        )
    if topology == "hybrid_shard_2x2":
        return init_device_mesh(
            "cuda",
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
    raise AssertionError(f"Unknown topology {topology}")


def _as_output_tuple(output):
    return output if isinstance(output, tuple) else (output,)


def _full_grad(param):
    if param.grad is None:
        return None
    grad = param.grad
    if isinstance(grad, DTensor):
        grad = grad.full_tensor()
    return grad.detach().clone()


def _assert_bitwise_equal(actual, expected, context):
    assert actual.dtype == expected.dtype, (
        f"{context}: expected dtype {expected.dtype}, got {actual.dtype}"
    )
    assert actual.shape == expected.shape, (
        f"{context}: expected shape {expected.shape}, got {actual.shape}"
    )
    assert torch.equal(actual, expected), f"{context}: tensors are not bitwise equal"


def _assert_run_equal(actual, expected, context):
    assert len(actual.outputs) == len(expected.outputs)
    for index, (actual_output, expected_output) in enumerate(
        zip(actual.outputs, expected.outputs)
    ):
        _assert_bitwise_equal(
            actual_output,
            expected_output,
            f"{context} output {index}",
        )
    assert actual.grads.keys() == expected.grads.keys()
    for name in actual.grads:
        actual_grad = actual.grads[name]
        expected_grad = expected.grads[name]
        if actual_grad is None or expected_grad is None:
            assert actual_grad is expected_grad, f"{context} grad {name}"
        else:
            _assert_bitwise_equal(
                actual_grad,
                expected_grad,
                f"{context} grad {name}",
            )


def _register_unsharded_param_hook(
    model,
    expected_shapes,
    expected_dtypes,
):
    hook_calls = []

    def check_unsharded_params(module, _inputs):
        params = dict(module.named_parameters())
        assert params.keys() == expected_shapes.keys()
        for name, param in params.items():
            expected_shape = expected_shapes[name]
            assert tuple(param.shape) == expected_shape, (
                f"{name}: expected shape {expected_shape}, got {tuple(param.shape)}"
            )
            assert param.numel() == expected_shape.numel(), (
                f"{name}: expected {expected_shape.numel()} elements, "
                f"got {param.numel()}"
            )
            assert param.dtype == expected_dtypes[name], (
                f"{name}: expected dtype {expected_dtypes[name]}, "
                f"got {param.dtype}"
            )
        hook_calls.append(True)

    model.register_forward_pre_hook(check_unsharded_params)
    return hook_calls


def _run_case(
    model_name,
    topology,
    mesh,
    policy_kind,
    rank,
):
    model, inputs = _create_case(model_name, rank)
    expected_shapes = {
        name: param.shape for name, param in model.named_parameters()
    }
    shard_size = 4 if topology == "fully_shard_1x4" else 2
    assert any(
        len(shape) > 0 and shape[0] % shard_size != 0
        for shape in expected_shapes.values()
    ), f"{model_name} does not exercise dim-0 padding for shard size {shard_size}"

    if policy_kind == "all_bf16_map":
        param_dtype_map = {
            name: torch.bfloat16 for name in expected_shapes
        }
        mp_policy = fsdp_param_dtype_patch.ParamDtypeMixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            param_dtype_map=param_dtype_map,
        )
    else:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    hook_calls = _register_unsharded_param_hook(
        model,
        expected_shapes,
        dict.fromkeys(expected_shapes, torch.bfloat16),
    )
    with sdpa_kernel(SDPBackend.MATH):
        output = model(*inputs)
        output_tuple = _as_output_tuple(output)
        sum(tensor.float().sum() for tensor in output_tuple).backward()
    assert len(hook_calls) == 1

    result = RunResult(
        outputs=tuple(tensor.detach().clone() for tensor in output_tuple),
        grads={
            name: _full_grad(param)
            for name, param in model.named_parameters()
        },
    )
    del model, inputs, output, output_tuple
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        device_id=torch.device("cuda", local_rank),
    )
    torch.use_deterministic_algorithms(True)
    rank = dist.get_rank()
    meshes = {topology: _create_mesh(topology) for topology in TOPOLOGIES}

    references = {}
    for model_name in MODEL_NAMES:
        for topology in TOPOLOGIES:
            references[(model_name, topology)] = _run_case(
                model_name,
                topology,
                meshes[topology],
                "unpatched",
                rank,
            )

    fsdp_param_dtype_patch.apply_param_dtype_map_patch()
    for model_name in MODEL_NAMES:
        for topology in TOPOLOGIES:
            reference = references[(model_name, topology)]
            patched = _run_case(
                model_name,
                topology,
                meshes[topology],
                "patched_standard",
                rank,
            )
            _assert_run_equal(
                patched,
                reference,
                f"{model_name} {topology} patched standard policy",
            )
            mapped = _run_case(
                model_name,
                topology,
                meshes[topology],
                "all_bf16_map",
                rank,
            )
            _assert_run_equal(
                mapped,
                reference,
                f"{model_name} {topology} all-BF16 map",
            )

    if rank == 0:
        print("OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
