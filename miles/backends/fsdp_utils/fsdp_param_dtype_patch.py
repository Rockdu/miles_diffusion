"""Version-dispatched FSDP param-dtype-map patch."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy


@dataclass(frozen=True)
class ParamDtypeMixedPrecisionPolicy(MixedPrecisionPolicy):
    param_dtype_map: Mapping[str, torch.dtype] | None = None


def apply_param_dtype_map_patch() -> None:
    torch_version = torch.__version__.partition("+")[0]
    if torch_version == "2.11.0":
        from miles.backends.fsdp_utils._fsdp_param_dtype_patch_2_11 import (
            apply_param_dtype_map_patch as apply_torch_2_11_patch,
        )

        apply_torch_2_11_patch()
        return
    raise RuntimeError("The Miles FSDP param-dtype patch supports torch==2.11.0, " f"got {torch.__version__}")


__all__ = [
    "ParamDtypeMixedPrecisionPolicy",
    "apply_param_dtype_map_patch",
]
