from miles.backends.fsdp_utils.models.parallel_plan import FSDPParallelPlan


# No fp32 pins: sglang-d loads these params in bf16, and a trainer forward
# running them in fp32 modulates with values the rollout never saw.
FSDP_PARALLEL_PLAN = FSDPParallelPlan()
