"""Engine-side LoRA parity for true-on-policy RL, as monkey patches.

Two sgl-d behaviours keep an unmerged (dynamic) LoRA engine from reproducing a
PEFT-adapted trainer bitwise. Both are fixed upstream-side in
Rockdu/sglang (fix/lora-ipc-honor-merge-mode, fix/rowparallel-lora-bias-fusion);
until those land, patch them from here so the fix travels with miles instead of
living as an edit to the installed sglang tree -- an edit any reinstall wipes,
silently: --sglang-lora-merge-mode dynamic is still accepted, the engine just
merges anyway and train/rollout output drifts to ~2.5e-2 once an adapter is
non-zero.

1. The LoRA IPC weight update hardcodes ``merge_weights=True``, bypassing
   ``lora_merge_mode`` entirely. Merging folds ``s*B@A`` into the base weight so
   the engine computes ``(W + s*B*A)x`` while PEFT computes ``Wx + s*B(A*x)``:
   algebraically equal, not bitwise.
2. ``RowParallelLinearWithLoRA.forward``'s unmerged path drops the base layer's
   rank-0 bias fusion and adds bias after the GEMM output was already rounded to
   bf16 -- a second rounding worth up to 1 bf16 ulp (measured 4.9e-3 max-rel on
   Wan's ``to_out`` against a bit-exact input, on ~a third of elements).

Arbitration behind (2), on dumped tensors with the engine's own weights:
``addmm(b, x, W^T)`` reproduces the trainer bitwise, ``mm(x, W^T) + b``
reproduces the unpatched engine bitwise -- each variant matches exactly one side.
"""

import os

import torch
from sglang.multimodal_gen.runtime.layers.lora import linear as _engine_linear
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    BaseLayerWithLoRA,
    RowParallelLinearWithLoRA,
)

# Set by miles when the recipe asks for dynamic LoRA (see ray/rollout.py); the
# layer has no view of the pipeline's server args, so the mode arrives this way.
UNMERGED_ENV = "MILES_LORA_FORCE_UNMERGED"

_orig_set_lora_weights = BaseLayerWithLoRA.set_lora_weights
_orig_row_forward = RowParallelLinearWithLoRA.forward


def _patched_set_lora_weights(
    self,
    A: torch.Tensor,
    B: torch.Tensor,
    lora_path: str | None = None,
    strength: float = 1.0,
    clear_existing: bool = False,
    merge_weights: bool = True,
) -> None:
    # Fix 1: the IPC updater passes merge_weights=True unconditionally. Honor the
    # requested mode here, which is the single place every caller funnels through.
    if os.environ.get(UNMERGED_ENV) == "1":
        merge_weights = False
    return _orig_set_lora_weights(
        self,
        A,
        B,
        lora_path=lora_path,
        strength=strength,
        clear_existing=clear_existing,
        merge_weights=merge_weights,
    )


def _patched_row_forward(self, input_: torch.Tensor):
    if self.merged or self.disable_lora:
        return self.base_layer(input_)

    lora_A = self.lora_A
    lora_B = self.lora_B
    if isinstance(self.lora_B, _engine_linear.DTensor):
        lora_B = self.lora_B.to_local()
        lora_A = self.lora_A.to_local()

    if self.base_layer.input_is_parallel:
        input_parallel = input_
    else:
        # Take the helpers off the engine module rather than guessing their import
        # paths, so this keeps working if sgl-d moves them.
        tp_rank = _engine_linear.get_tp_rank()
        splitted_input = _engine_linear.split_tensor_along_last_dim(input_, num_partitions=self.base_layer.tp_size)
        input_parallel = splitted_input[tp_rank].contiguous()

    # Fix 2: fuse bias into the GEMM exactly like RowParallelLinear.forward does
    # (rank 0 only, so TP>1 still adds it once after the all-reduce). The unfused
    # mm-then-add rounds the GEMM output to bf16 before the bias add.
    bias_ = None if (self.base_layer.tp_rank > 0 or self.base_layer.skip_bias_add) else self.base_layer.bias
    output_parallel = self.base_layer.quant_method.apply(self.base_layer, input_parallel, bias=bias_)

    lora_dtype = lora_A.dtype
    input_parallel_lora = input_parallel.to(dtype=lora_dtype)
    lora_A_sliced = self.slice_lora_a_weights(lora_A.to(device=input_parallel.device, non_blocking=True))
    lora_B_sliced = self.slice_lora_b_weights(lora_B.to(device=input_parallel.device, non_blocking=True))
    delta_parallel = input_parallel_lora @ lora_A_sliced.T @ lora_B_sliced.T
    if self.lora_alpha != self.lora_rank:
        delta_parallel = delta_parallel * (self.lora_alpha / self.lora_rank)
    delta_parallel = delta_parallel * self.strength
    output_parallel = output_parallel + delta_parallel.to(dtype=output_parallel.dtype)

    if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
        output_ = _engine_linear.tensor_model_parallel_all_reduce(output_parallel)
    else:
        output_ = output_parallel

    # Bias already went through the GEMM epilogue on rank 0; adding it again here
    # would double-count it.
    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    return output_, output_bias


def apply() -> None:
    BaseLayerWithLoRA.set_lora_weights = _patched_set_lora_weights
    RowParallelLinearWithLoRA.forward = _patched_row_forward
