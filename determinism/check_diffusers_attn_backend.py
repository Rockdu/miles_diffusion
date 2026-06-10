import os
from diffusers.models import attention_dispatch as ad
# default env-driven backend
print("DIFFUSERS_ATTN_BACKEND env =", os.environ.get("DIFFUSERS_ATTN_BACKEND", "<unset>"))
try:
    name, fn = ad._AttentionBackendRegistry.get_active_backend()
    print("active_backend =", name)
except Exception as e:
    print("active_backend err:", e)
print("_CAN_USE_FLASH_ATTN =", getattr(ad, "_CAN_USE_FLASH_ATTN", "?"))
print("_CAN_USE_FLASH_ATTN_3 =", getattr(ad, "_CAN_USE_FLASH_ATTN_3", "?"))
import torch
print("deterministic_algorithms =", torch.are_deterministic_algorithms_enabled())
# find the default string for DIFFUSERS_ATTN_BACKEND
import inspect, re
src = inspect.getsource(ad)
m = re.search(r'DIFFUSERS_ATTN_BACKEND[^\n]*', src)
print("default-line:", m.group(0) if m else "?")
