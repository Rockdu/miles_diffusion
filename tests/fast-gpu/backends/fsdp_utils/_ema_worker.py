"""Two-rank EMA oracle with FSDP's default root/child parameter lifetimes.

    sharded actor --> pending graph --> EMA forward --> restore --> backward
        |                               |                            |
    oracle actor                   frozen oracle                AdamW oracle

The actor explicitly saves its weights before reference switching; TensorBackuper
only copies local shards. Independent models check outputs, gradients, updates,
and Parameter identity.
Root/child FSDP, checkpointing, mixed precision, offload, and LoRA share this oracle.
CPU offload synchronizes pending H2D reads before overwriting pinned actor shards.
"""

import argparse
import copy
import os

import torch
import torch.distributed as dist
from tests.fast.backends.fsdp_utils._weight_test_utils import make_weight_actor, reference_weights
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        return self.fc(x).tanh()


class Tiny(nn.Module):
    def __init__(self, use_checkpoint):
        super().__init__()
        self.input = nn.Linear(16, 16)
        self.block = Block()
        self.output = nn.Linear(16, 4)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        x = self.input(x).tanh()
        if self.use_checkpoint and torch.is_grad_enabled():
            x = checkpoint(self.block, x, use_reentrant=False)
        else:
            x = self.block(x)
        return self.output(x)


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _wrap(model, mesh, args):
    kwargs = {
        "mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        ),
        "offload_policy": CPUOffloadPolicy() if args.cpu_offload else None,
    }
    for module in list(model.modules()):
        if isinstance(module, Block):
            fully_shard(module, **kwargs)
    # Root owns input/output, child owns block.fc, including their PEFT adapters.
    # Keep default post-forward retention so the production switch must refresh cached weights.
    fully_shard(model, **kwargs)
    return model


def _forward(model, x, args):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
        return model(x)


def _check_close(actual, expected, label, args):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-2 if args.bf16 else 2e-5,
        atol=2e-6,
        msg=label,
    )


def _trial(mesh, args):
    torch.manual_seed(7)
    base = Tiny(args.checkpoint).cuda()
    if args.lora:
        from peft import LoraConfig, get_peft_model

        base = get_peft_model(
            base, LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0, target_modules=["input", "fc", "output"])
        )
        with torch.no_grad():
            for name, param in base.named_parameters():
                if "lora_B" in name:
                    param.normal_(std=0.15)  # Exercise gradients in both adapter matrices.

    oracle_actor = copy.deepcopy(base)
    oracle_ref = copy.deepcopy(base).requires_grad_(False)
    with torch.no_grad():
        for param in oracle_actor.parameters():
            if param.requires_grad:
                param.add_(0.15)  # EMA and live weights must give different predictions.

    actor, oracle_actor, oracle_ref = [_wrap(model, mesh, args) for model in (base, oracle_actor, oracle_ref)]
    params = dict(actor.named_parameters())
    expected_params = dict(oracle_actor.named_parameters())
    trainable = {name: param for name, param in params.items() if param.requires_grad}
    assert any("block." in name for name in trainable), "No child-managed trainable parameters"
    assert any("block." not in name for name in trainable), "No root-managed trainable parameters"
    harness = make_weight_actor(actor, fsdp_cpu_offload=args.cpu_offload)
    backuper = harness.tensor_backuper
    with torch.no_grad():
        for name, param in trainable.items():
            _local(param).copy_(_local(expected_params[name]))
    backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
    backuper.backup("actor", device="cpu", pin_memory=True, fixed_groups=harness.tensor_backuper.frozen_tensor_groups)
    live = {name: _local(param.detach()).clone() for name, param in params.items()}
    optimizers = [
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, foreach=False)
        for model in (actor, oracle_actor)
    ]

    torch.manual_seed(100 + dist.get_rank())
    x = torch.randn(4, 16, device="cuda")
    expected_new = _forward(oracle_actor, x, args)
    with torch.no_grad():
        expected_ref = _forward(oracle_ref, x, args).detach()
    assert (expected_new.detach().float() - expected_ref.float()).abs().max() > 1e-3

    # Leave the actor graph alive across the explicit no-grad EMA round trip.
    actual_new = _forward(actor, x, args)
    with reference_weights(harness, "ema"):
        actual_ref = _forward(actor, x, args).detach()
    torch.cuda.synchronize()
    for name, param in params.items():
        torch.testing.assert_close(_local(param.detach()), live[name], rtol=0, atol=0, msg=f"restore {name}")
    _check_close(actual_new, expected_new, "actor output", args)
    _check_close(actual_ref, expected_ref, "EMA output", args)
    assert not actual_ref.requires_grad
    assert all(param.grad is None for param in params.values())

    actual_loss = (actual_new.float() - actual_ref.float()).square().mean()
    expected_loss = (expected_new.float() - expected_ref.float()).square().mean()
    _check_close(actual_loss, expected_loss, "loss", args)
    actual_loss.backward()
    expected_loss.backward()
    torch.cuda.synchronize()
    for name, param in params.items():
        assert dict(actor.named_parameters())[name] is param, f"Parameter replaced: {name}"
        if not param.requires_grad:
            assert param.grad is None and expected_params[name].grad is None
            continue
        assert param.grad is not None and expected_params[name].grad is not None, name
        _check_close(_local(param.grad), _local(expected_params[name].grad), f"gradient {name}", args)
    assert all(param.grad is None for param in oracle_ref.parameters())
    for optimizer in optimizers:
        optimizer.step()
    for name, param in params.items():
        torch.testing.assert_close(
            _local(param.detach()),
            _local(expected_params[name].detach()),
            rtol=2e-5,
            atol=2e-6,
            msg=f"Adam update {name}",
        )


def main():
    parser = argparse.ArgumentParser()
    for flag in ("checkpoint", "bf16", "cpu-offload", "lora"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        assert dist.get_world_size() == 2
        _trial(init_device_mesh("cuda", (2,)), args)
        dist.barrier()
        if dist.get_rank() == 0:
            print("OK", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
