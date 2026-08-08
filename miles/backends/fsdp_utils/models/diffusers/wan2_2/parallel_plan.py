from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


# Only norm2 is pinned. The rollout engine keeps every FP32LayerNorm affine
# param resident and consumed in fp32, so pinning them makes the training
# matmul consume the same weight dtype -- verified on paired dumps.
#
# scale_shift_table and time_embedder are deliberately NOT pinned: sglang-d
# loads both in bf16, so an fp32-resident trainer would modulate with
# unrounded master values the rollout never saw. On a paired dump that showed
# up as bit-exact block inputs (hidden, context and temb all rel=0.0) feeding
# a blocks.0 output that was already 1.45e-2 off, and it compounds through all
# 40 layers. Pinning them is the numerically better trainer in isolation; it
# is not the trainer that generated the trajectory.
FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    param_dtype_patterns={
        "*.norm2.*": "fp32",
    },
)
