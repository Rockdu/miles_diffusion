from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


# When is pinning a parameter fp32 correct? Two conditions, both required:
#
#   1. Its consuming op escapes autocast, so master precision actually reaches
#      the computation. LayerNorm runs fp32 under autocast and elementwise ops
#      are not cast at all; matmul/conv weights are re-rounded to the forward
#      dtype at the op, which makes an fp32 pin a numerical no-op for them.
#   2. The rollout engine keeps the same parameter fp32-resident, so both
#      sides consume the same values. The engine loads the DiT in bf16 except
#      where its loader deliberately holds fp32.
#
# norm2 (FP32LayerNorm affine) satisfies both: LayerNorm consumes the weight at
# fp32, and the engine keeps these resident fp32 (verified on paired dumps).
# Without the pin the trainer gathers rounded bf16 against the engine's
# unrounded fp32 — the mismatch is on the trainer side.
#
# scale_shift_table satisfies 1 but not 2: the per-block modulation tables are
# elementwise consumers, but the engine loads them bf16. Pinning them fed the
# trainer unrounded master values the rollout never saw — with every block
# input bit-exact, blocks.0.output was already 1.45e-2 off, compounding over
# all 40 layers.
#
# time_embedder satisfies neither: its parameters are linear weights, and
# autocast re-rounds them to the same bf16 the unpinned gather would produce.
# Pinning it changed nothing measurably (temb stayed bit-exact either way) and
# only cost fp32 gather traffic.
FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    param_dtype_patterns={
        "*.norm2.*": "fp32",
    },
)
