import torch

from miles.backends.fsdp_utils.fsdp_param_dtype_patch import apply_param_dtype_map_patch

apply_param_dtype_map_patch()
torch.use_deterministic_algorithms(True)
