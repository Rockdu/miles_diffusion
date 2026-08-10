from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


FSDP_PARALLEL_PLAN = FSDPParallelPlan(
    param_dtype_patterns={
        "*scale_shift_table": "fp32",
        "*time_embedder*": "fp32",
        "*.norm2.*": "fp32",
    },
)
