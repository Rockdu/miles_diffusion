"""What --deterministic-mode accepts as an attention backend, and why.

Words are vocabulary-checked upstream (test_fsdp_attention_backend_args.py);
this gate decides whether the kernel behind the word has a deterministic backward.

                     --fsdp-attention-backend (miles word)
                                   |
                +------------------+------------------+
                |                                     |
        DiffusersModelBackend                   MilesModelBackend (ltx)
        None -> DIFFUSERS_ATTN_BACKEND          None -> reject (the package picks
                env, else torch_sdpa                    its default kernel itself)
                |                                     |
    torch_*_sdpa (minus cudnn) -> TORCH_FLAG    same             -> TORCH_FLAG
    fa2, fa3                   -> PATCH_FLASH   fa3, fa4         -> PATCH_FLASH
    torch_cudnn_sdpa           -> reject        torch_cudnn_sdpa -> reject
    fa4 (hub-only), sage_attn, aiter -> reject
                |
                +--> ring degree > 1: only {None, torch_flash_sdpa} survive
                |    (ring calls the aten op directly; torch's guard never runs)
                |
                +--> rollout engines (unless --train-only): --sglang-tp-size must
                     be 1; TP's cross-rank matmul reduction is not reproducible

PATCH_FLASH must also beat diffusers' own explicit deterministic=False at the
call site, which functools.partial cannot -- covered at the bottom.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.backends.fsdp_utils import deterministic
from miles.backends.fsdp_utils.models.diffusers import attention as diffusers_attention

DIFFUSERS_BACKEND_PATH = "miles.backends.fsdp_utils.model_backend.DiffusersModelBackend"
MILES_BACKEND_PATH = "miles.backends.fsdp_utils.model_backend.MilesModelBackend"
LTX_CONFIG_PATH = "miles.backends.fsdp_utils.configs.ltx.LTXTrainPipelineConfig"
SD3_CONFIG_PATH = "miles.backends.fsdp_utils.configs.sd3.SD3TrainPipelineConfig"


def _args(
    fsdp_attention_backend,
    *,
    deterministic_mode=True,
    model_backend_path=DIFFUSERS_BACKEND_PATH,
    train_pipeline_config_path=SD3_CONFIG_PATH,
    sequence_parallel_size=1,
    ulysses_degree=0,
    train_env_vars=None,
    train_only=True,
    sglang_tp_size=None,
):
    return Namespace(
        deterministic_mode=deterministic_mode,
        fsdp_attention_backend=fsdp_attention_backend,
        model_backend_path=model_backend_path,
        train_pipeline_config_path=train_pipeline_config_path,
        sequence_parallel_size=sequence_parallel_size,
        ulysses_degree=ulysses_degree,
        train_env_vars=train_env_vars or {},
        train_only=train_only,
        sglang_tp_size=sglang_tp_size,
        sglang_attention_backend=None,
        sglang_server_concurrency=None,
    )


@pytest.fixture
def flash_is_deterministic(monkeypatch):
    """Make diffusers' flash entry points look deterministic-capable: the CPU runner
    has no flash-attn, and the gate reads their signature."""
    import diffusers.models.attention_dispatch as attention_dispatch

    def fake_flash(q, k, v, deterministic=False):
        return q

    for name in diffusers_attention.FLASH_FNS:
        monkeypatch.setattr(attention_dispatch, name, fake_flash, raising=False)


class TestDiffusersBackends:
    """Left branch of the map."""

    def test_disabled_is_noop(self):
        deterministic.validate_deterministic_args(_args("sage_attn", deterministic_mode=False))

    @pytest.mark.parametrize("backend", [None, "torch_sdpa", "torch_efficient_sdpa", "torch_flash_sdpa"])
    def test_sdpa_backends_ride_the_torch_flag(self, backend):
        assert deterministic.attention_policy(_args(backend)) == deterministic.TORCH_FLAG

    @pytest.mark.parametrize("backend", ["fa2", "fa3"])
    def test_flash_backends_are_patchable(self, backend, flash_is_deterministic):
        assert deterministic.attention_policy(_args(backend)) == deterministic.PATCH_FLASH

    @pytest.mark.parametrize("backend", ["sage_attn", "aiter"])
    def test_opaque_kernels_rejected(self, backend):
        with pytest.raises(ValueError, match="opaque"):
            deterministic.attention_policy(_args(backend))

    def test_cudnn_rejected(self):
        with pytest.raises(ValueError, match="cuDNN"):
            deterministic.attention_policy(_args("torch_cudnn_sdpa"))

    def test_fa4_rejected_as_hub_only(self, flash_is_deterministic):
        with pytest.raises(ValueError, match="hub"):
            deterministic.attention_policy(_args("fa4"))

    def test_unset_backend_follows_the_env_diffusers_reads(self, monkeypatch):
        monkeypatch.delenv(diffusers_attention.ATTN_BACKEND_ENV, raising=False)
        with pytest.raises(ValueError, match="sage_attn"):
            deterministic.attention_policy(_args(None, train_env_vars={"DIFFUSERS_ATTN_BACKEND": "sage"}))

    def test_flash_rejected_when_kernel_has_no_deterministic_arg(self, monkeypatch):
        import diffusers.models.attention_dispatch as attention_dispatch

        monkeypatch.setattr(attention_dispatch, "flash_attn_func", None, raising=False)
        with pytest.raises(RuntimeError):
            deterministic.attention_policy(_args("fa2"))


class TestLTXBackends:
    """Right branch of the map: the same words, spelled by the LTX package."""

    def _ltx_args(self, backend, **kwargs):
        return _args(
            backend,
            model_backend_path=MILES_BACKEND_PATH,
            train_pipeline_config_path=LTX_CONFIG_PATH,
            **kwargs,
        )

    @pytest.mark.parametrize("backend", ["torch_sdpa", "torch_math_sdpa", "torch_flash_sdpa"])
    def test_sdpa_backends_accepted(self, backend):
        assert deterministic.attention_policy(self._ltx_args(backend)) == deterministic.TORCH_FLAG

    @pytest.mark.parametrize("backend", ["fa3", "fa4"])
    def test_flash_backends_are_patchable(self, backend):
        assert deterministic.attention_policy(self._ltx_args(backend)) == deterministic.PATCH_FLASH

    def test_cudnn_rejected(self):
        with pytest.raises(ValueError, match="cuDNN"):
            deterministic.attention_policy(self._ltx_args("torch_cudnn_sdpa"))

    def test_unset_backend_rejected(self):
        with pytest.raises(ValueError, match="explicit"):
            deterministic.attention_policy(self._ltx_args(None))


class TestRingAttention:
    """Bottom of the map: ring bypasses torch's dispatcher, so it accepts less."""

    def _ring_args(self, backend):
        return _args(backend, sequence_parallel_size=4, ulysses_degree=2)

    def test_cudnn_ring_kernel_rejected(self):
        with pytest.raises(ValueError, match="ring"):
            deterministic.validate_deterministic_args(self._ring_args("torch_cudnn_sdpa"))

    @pytest.mark.parametrize("backend", [None, "torch_flash_sdpa"])
    def test_deterministic_ring_kernels_accepted(self, backend):
        deterministic.validate_deterministic_args(self._ring_args(backend))

    def test_ulysses_only_does_not_constrain_the_kernel(self, flash_is_deterministic):
        deterministic.validate_deterministic_args(_args("fa2", sequence_parallel_size=4, ulysses_degree=4))


class TestRolloutEngine:
    """Rollout runs without the deterministic runtime; only its TP degree is enforceable."""

    def test_tensor_parallel_rejected(self):
        with pytest.raises(ValueError, match="tp-size"):
            deterministic.validate_deterministic_args(_args("torch_sdpa", train_only=False, sglang_tp_size=2))

    @pytest.mark.parametrize("tp_size", [None, 1])
    def test_tp_1_accepted(self, tp_size):
        deterministic.validate_deterministic_args(_args("torch_sdpa", train_only=False, sglang_tp_size=tp_size))

    def test_train_only_skips_the_rollout_check(self):
        deterministic.validate_deterministic_args(_args("torch_sdpa", train_only=True, sglang_tp_size=2))


def test_patch_overrides_an_explicit_caller_argument():
    """The patch has to win over diffusers' own deterministic=False, which is what
    functools.partial could not do."""
    seen = {}

    def kernel(*, deterministic=False):
        seen["deterministic"] = deterministic

    deterministic._force_deterministic(kernel)(deterministic=False)

    assert seen["deterministic"] is True


def test_fsdp_args_expose_new_flags():
    import dataclasses

    from miles.backends.fsdp_utils.arguments import FSDPArgs

    names = {f.name for f in dataclasses.fields(FSDPArgs)}
    assert "fsdp_attention_backend" in names
    assert "deterministic_mode" in names
