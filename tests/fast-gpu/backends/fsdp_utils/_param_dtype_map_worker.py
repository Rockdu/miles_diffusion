import copy
import os
from argparse import Namespace

import torch
import torch.distributed as dist
from torch import nn

from miles.backends.fsdp_utils.actor import apply_fsdp2


class MixedParamDtypeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.low_precision = nn.Linear(8, 15)
        self.full_precision = nn.Linear(8, 15)
        self.seen_param_dtypes = None

    def forward(self, x):
        self.seen_param_dtypes = (
            self.low_precision.weight.dtype,
            self.full_precision.weight.dtype,
        )
        low_precision_output = self.low_precision(x)
        full_precision_output = self.full_precision(
            x.to(self.full_precision.weight.dtype)
        )
        return low_precision_output + full_precision_output.to(
            low_precision_output.dtype
        )


class MixedParamDtypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = MixedParamDtypeBlock()

    def forward(self, x):
        return self.block(x)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    torch.manual_seed(42)
    model = MixedParamDtypeModel().cuda()
    ref_model = copy.deepcopy(model)
    ref_model.block.low_precision.to(torch.bfloat16)
    model = apply_fsdp2(
        model,
        args=Namespace(
            diffusion_forward_dtype="bf16",
            fsdp_reduce_dtype="fp32",
            gradient_checkpointing=False,
        ),
        no_split_modules=["MixedParamDtypeBlock"],
        param_dtype_patterns={"block.full_precision.*": "fp32"},
    )

    torch.manual_seed(43)
    inp = torch.randn(4, 8, device="cuda")
    output = model(inp)
    ref_output = ref_model(inp.to(torch.bfloat16))
    torch.testing.assert_close(output, ref_output)
    assert model.block.seen_param_dtypes == (torch.bfloat16, torch.float32)

    output.sum().backward()
    ref_output.sum().backward()
    for param, ref_param in zip(model.parameters(), ref_model.parameters()):
        assert param.grad is not None
        assert ref_param.grad is not None
        assert param.grad.dtype == torch.float32
        torch.testing.assert_close(
            param.grad.full_tensor(),
            ref_param.grad.to(torch.float32),
        )

    if dist.get_rank() == 0:
        print("OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
