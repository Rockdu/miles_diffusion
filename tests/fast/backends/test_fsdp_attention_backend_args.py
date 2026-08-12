"""--fsdp-attention-backend must name a kernel the run's model backend can serve."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.backends.fsdp_utils.arguments import validate_attention_backend

DIFFUSERS_ARGS = Namespace(
    model_backend_path="miles.backends.fsdp_utils.model_backend.DiffusersModelBackend",
    train_pipeline_config_path="miles.backends.fsdp_utils.configs.sd3.SD3TrainPipelineConfig",
)
LTX_ARGS = Namespace(
    model_backend_path="miles.backends.fsdp_utils.model_backend.MilesModelBackend",
    train_pipeline_config_path="miles.backends.fsdp_utils.configs.ltx.LTXTrainPipelineConfig",
)


def _args(template, backend):
    return Namespace(fsdp_attention_backend=backend, **vars(template))


@pytest.mark.parametrize("template", [DIFFUSERS_ARGS, LTX_ARGS], ids=["diffusers", "ltx"])
def test_shared_words_are_served_by_both_backends(template):
    args = _args(template, "torch_math_sdpa")

    validate_attention_backend(args)

    assert args.fsdp_attention_backend == "torch_math_sdpa"


def test_backend_specific_word_rejected_by_the_other_backend():
    # fa2 exists in diffusers; ltx_core has no FLASH_ATTENTION_2.
    validate_attention_backend(_args(DIFFUSERS_ARGS, "fa2"))

    with pytest.raises(ValueError, match="not a kernel"):
        validate_attention_backend(_args(LTX_ARGS, "fa2"))


def test_word_outside_the_vocabulary_rejected():
    # diffusers' own spelling is not the vocabulary miles speaks.
    with pytest.raises(ValueError, match="not a kernel"):
        validate_attention_backend(_args(DIFFUSERS_ARGS, "_native_math"))


def test_normalized_for_exact_comparison_downstream():
    args = _args(DIFFUSERS_ARGS, " TORCH_SDPA ")

    validate_attention_backend(args)

    assert args.fsdp_attention_backend == "torch_sdpa"


def test_unset_is_left_alone():
    args = _args(DIFFUSERS_ARGS, None)

    validate_attention_backend(args)

    assert args.fsdp_attention_backend is None
