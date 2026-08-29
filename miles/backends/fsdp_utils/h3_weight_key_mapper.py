"""Map diffusers MiniMax H3 LoRA names to sglang H3 DiT layer names.

Training uses diffusers ``MiniMaxH3Transformer3DModel`` (separate Q/K/V).
Rollout uses sglang ``MiniMaxH3DiTModel`` (fused ``qkv_proj``). LoRA IPC sync
must therefore rename modules and stack the Q/K/V adapters before the push.

Names that resolve to no sglang layer are skipped with a warning on the rollout
side, so an incomplete map silently freezes those adapters at their checkpoint
values — anything unrecognized is reported as unmapped instead of guessed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch

_QKV_RE = re.compile(
    r"^(?P<prefix>(?:token_refiner\.)?refiner_blocks\.(?P<idx>\d+)|transformer_blocks\.(?P<idx2>\d+))"
    r"\.attn\.to_(?P<which>q|k|v)\.weight$"
)
# After PEFT strip, refiner path is token_refiner.refiner_blocks -> token_refiner.blocks
_QKV_RE_SGL = re.compile(
    r"^(?P<prefix>(?:token_refiner\.)?blocks\.(?P<idx>\d+)|blocks\.(?P<idx2>\d+))"
    r"\.attn\.to_(?P<which>q|k|v)\.weight$"
)


def _qkv_group_key(name: str) -> tuple[str, str] | None:
    for regex in (_QKV_RE, _QKV_RE_SGL):
        m = regex.match(name)
        if m is None:
            continue
        prefix = m.group("prefix")
        if prefix.startswith("token_refiner."):
            block_idx = m.group("idx")
            sgld_prefix = f"token_refiner.blocks.{block_idx}"
        elif prefix.startswith("refiner_blocks."):
            block_idx = m.group("idx")
            sgld_prefix = f"token_refiner.blocks.{block_idx}"
        else:
            block_idx = m.group("idx2")
            sgld_prefix = f"blocks.{block_idx}"
        return sgld_prefix, m.group("which")
    return None


def _swap_gated_ffn_halves(tensor: torch.Tensor) -> torch.Tensor:
    """Reorder a gated FFN input projection from diffusers' halves to sglang's.

    diffusers' GEGLU splits the fused projection as ``[up, gate]`` and computes
    ``up * gelu(gate)``; sglang's ``mlp.fc1`` splits it as ``[gate, up]`` and
    computes ``silu(gate) * up``. Same weights, opposite halves.
    """
    rows = tensor.shape[0]
    if rows % 2:
        raise ValueError(f"H3 gated FFN projection must have an even row count, got {rows}")
    half = rows // 2
    return torch.cat([tensor[half:], tensor[:half]], dim=0)


_LORA_AB_RE = re.compile(r"\.lora_([AB])(?:\.[^.]+)?(?:\.weight)?$")
_PEFT_PREFIX = "base_model.model."

# LoRA-able H3 submodules other than Q/K/V, as (diffusers suffix, sglang suffix).
# Kept as an explicit whitelist: an unrecognized module must surface as unmapped
# rather than reach the rollout under a guessed name, where it would be skipped
# with only a warning and silently freeze that adapter.
_LORA_MODULE_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".attn.to_out.0", ".attn.out_proj"),
    (".ff.net.0.proj", ".mlp.fc1"),
    (".ff.net.2", ".mlp.fc2"),
)

_BLOCK_PREFIX_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"^token_refiner\.refiner_blocks\.", "token_refiner.blocks."),
    (r"^refiner_blocks\.", "token_refiner.blocks."),
    (r"^transformer_blocks\.", "blocks."),
)


def _strip_peft_prefix(name: str) -> str:
    return name[len(_PEFT_PREFIX) :] if name.startswith(_PEFT_PREFIX) else name


def _normalize_block_prefix(module_path: str) -> str:
    out = module_path
    for pattern, repl in _BLOCK_PREFIX_REPLACEMENTS:
        out = re.sub(pattern, repl, out)
    return out


def _stack_qkv_lora(triple: dict[str, dict[str, torch.Tensor]], layer: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-projection LoRA into the 3D layout sglang's fused qkv expects.

    ``MergedColumnParallelLinearWithLoRA`` multiplies a 3D ``B @ A`` batchwise and
    flattens the result, so stacking along a leading axis yields a delta ordered
    ``[q_all, k_all, v_all]`` — exactly how sglang stores ``qkv_proj.weight``.
    Note this differs from the dense path, which must instead emit the head-major
    grouped layout because it goes through the checkpoint weight loader.
    """
    missing = {"q", "k", "v"} - set(triple)
    if missing:
        raise ValueError(f"H3 LoRA IPC incomplete QKV for {layer}: missing {sorted(missing)}")
    order = ("q", "k", "v")
    a_shapes = {triple[w]["A"].shape for w in order}
    b_shapes = {triple[w]["B"].shape for w in order}
    if len(a_shapes) != 1 or len(b_shapes) != 1:
        raise ValueError(
            f"H3 LoRA IPC expects MHA-shaped Q/K/V adapters for {layer}, "
            f"got A={sorted(a_shapes)} B={sorted(b_shapes)}"
        )
    lora_a = torch.stack([triple[w]["A"] for w in order], dim=0)
    lora_b = torch.stack([triple[w]["B"] for w in order], dim=0)
    return lora_a, lora_b


def collect_h3_lora_layer_groups(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[list[list[tuple[str, torch.Tensor]]], list[str], int]:
    """Group PEFT LoRA tensors into sglang H3 layer names for IPC weight sync.

    Returns ``(layer_groups, unmapped_keys, num_lora_keys)`` to match
    ``collect_lora_layer_groups``; each group holds one layer's A/B pair so they
    always land in the same IPC bucket.
    """
    per_module: dict[str, dict[str, torch.Tensor]] = {}
    unmapped: list[str] = []
    num_lora_keys = 0

    for name, tensor in state_dict.items():
        if ".lora_A" not in name and ".lora_B" not in name:
            continue
        stripped = _strip_peft_prefix(name)
        match = _LORA_AB_RE.search(stripped)
        if match is None:
            unmapped.append(name)
            continue
        per_module.setdefault(stripped[: match.start()], {})[match.group(1)] = tensor
        num_lora_keys += 1

    qkv_pending: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    simple: dict[str, dict[str, torch.Tensor]] = {}

    for module_path, ab in per_module.items():
        if "A" not in ab or "B" not in ab:
            unmapped.append(module_path)
            continue

        probe = f"{module_path}.weight"
        qkv = _qkv_group_key(probe)
        if qkv is not None:
            sgld_prefix, which = qkv
            qkv_pending.setdefault(f"{sgld_prefix}.attn.qkv_proj", {})[which] = ab
            continue

        normalized = _normalize_block_prefix(module_path)
        for diffusers_suffix, sglang_suffix in _LORA_MODULE_SUFFIXES:
            if normalized.endswith(diffusers_suffix):
                layer = normalized[: -len(diffusers_suffix)] + sglang_suffix
                if diffusers_suffix == ".ff.net.0.proj":
                    # Gated FFN: diffusers stores [up, gate], sglang fc1 wants
                    # [gate, up]. Only B is row-indexed by output, so A is untouched.
                    ab = {"A": ab["A"], "B": _swap_gated_ffn_halves(ab["B"])}
                simple[layer] = ab
                break
        else:
            unmapped.append(module_path)

    groups: list[list[tuple[str, torch.Tensor]]] = []
    for layer in sorted(simple):
        ab = simple[layer]
        groups.append([(f"{layer}.lora_A", ab["A"]), (f"{layer}.lora_B", ab["B"])])
    for layer in sorted(qkv_pending):
        lora_a, lora_b = _stack_qkv_lora(qkv_pending[layer], layer)
        groups.append([(f"{layer}.lora_A", lora_a), (f"{layer}.lora_B", lora_b)])

    return groups, unmapped, num_lora_keys


# Grouped qkv rows come in per-head [q, k, v] runs of this width; sglang's H3
# checkpoint loader (``_install_qkv_weight_loader``) splits on it.
_H3_HEAD_DIM = 128

_FULL_PREFIX_RENAMES: tuple[tuple[str, str], ...] = (
    ("proj_in.", "video_patch_proj."),
    ("audio_proj_in.", "audio_patch_proj."),
    ("context_embedder.", "condition_proj."),
    ("time_embedder.linear_1.", "time_embedder.proj_in."),
    ("time_embedder.linear_2.", "time_embedder.proj_out."),
    ("norm_out.linear.", "final_layer.adaln_proj.linear."),
    ("norm_out.norm.", "final_layer.norm."),
    ("proj_out.", "final_layer.video_out."),
    ("audio_proj_out.", "final_layer.audio_out."),
)

_FULL_SUFFIX_RENAMES: tuple[tuple[str, str], ...] = (
    (".attn.to_out.0.", ".attn.out_proj."),
    (".ff.net.0.proj.", ".mlp.fc1."),
    (".ff.net.2.", ".mlp.fc2."),
    (".attn.norm_q.", ".attn.q_norm."),
    (".attn.norm_k.", ".attn.k_norm."),
)

_FULL_IDENTITY_NAMES = frozenset({"token_refiner.final_norm.weight", "token_refiner.final_norm.bias"})

# Block-internal params that keep their name apart from the block prefix; anything
# else inside a block must match a rename rule or it is unmapped, never guessed.
_BLOCK_PASSTHROUGH_RE = re.compile(r"\.(adaln_proj\.linear|norm1|norm2)\.(weight|bias)$")


def map_h3_full_name(name: str) -> tuple[str | None, str]:
    """Map one diffusers H3 state-dict name to ``(sglang name, op)``.

    op is "copy", "ffn_swap" (gated fc1 halves reordered), or "qkv" (member of a
    to_q/to_k/to_v trio fused into the grouped checkpoint layout). Returns
    ``(None, "")`` for a name with no known rollout counterpart.
    """
    qkv = _qkv_group_key(name)
    if qkv is not None:
        sgld_prefix, _which = qkv
        return f"{sgld_prefix}.attn.qkv_proj.weight", "qkv"

    target = _normalize_block_prefix(name)
    block_scoped = target != name
    for old_prefix, new_prefix in _FULL_PREFIX_RENAMES:
        if target.startswith(old_prefix) and target[len(old_prefix) :] in ("weight", "bias"):
            return new_prefix + target[len(old_prefix) :], "copy"
    for old_suffix, new_suffix in _FULL_SUFFIX_RENAMES:
        if old_suffix in target:
            op = "ffn_swap" if old_suffix == ".ff.net.0.proj." and target.endswith(".weight") else "copy"
            return target.replace(old_suffix, new_suffix), op
    if block_scoped:
        return (target, "copy") if _BLOCK_PASSTHROUGH_RE.search(target) else (None, "")
    if name in _FULL_IDENTITY_NAMES:
        return name, "copy"
    return None, ""


def _fuse_qkv_grouped(trio: dict[str, torch.Tensor], layer: str) -> torch.Tensor:
    q, k, v = trio["q"], trio["k"], trio["v"]
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"H3 qkv shapes differ for {layer}: {q.shape} {k.shape} {v.shape}")
    heads, rem = divmod(q.shape[0], _H3_HEAD_DIM)
    if rem:
        raise ValueError(f"H3 qkv rows {q.shape[0]} not divisible by head_dim {_H3_HEAD_DIM} for {layer}")
    parts = [t.reshape(heads, _H3_HEAD_DIM, *t.shape[1:]) for t in (q, k, v)]
    return torch.cat(parts, dim=1).reshape(3 * heads * _H3_HEAD_DIM, *q.shape[1:])


def collect_h3_full_weight_tensors(state_dict: Mapping[str, torch.Tensor], prepare):
    """Yield ``(sglang name, tensor)`` for a full H3 diffusers state dict.

    The push must match the FL2VA checkpoint layout the engine loads at startup:
    ``qkv_proj`` in per-head grouped ``[q, k, v]`` rows (its weight loader
    reorders and TP-shards from that layout) and ``mlp.fc1`` in sglang's gated
    half order. ``prepare`` resolves a state-dict value to a full local tensor
    (the updater passes its DTensor gather). Any unrecognized name raises up
    front: pushed under a guessed name it would be skipped by the rollout with
    only a warning, silently freezing that weight at its checkpoint value.

    Incompatible with the engine's AdaLN cache (minimax_h3_adaln_cache_path /
    minimax_h3_adaln_online): both cache modes read adaln weights from files,
    so synced adaln_proj updates would silently never take effect.
    """
    unmapped = [name for name in state_dict if map_h3_full_name(name)[0] is None]
    qkv_members: dict[str, set[str]] = {}
    for name in state_dict:
        qkv = _qkv_group_key(name)
        if qkv is not None:
            sgld_prefix, which = qkv
            qkv_members.setdefault(sgld_prefix, set()).add(which)
    incomplete = [layer for layer, members in qkv_members.items() if members != {"q", "k", "v"}]
    if unmapped or incomplete:
        raise ValueError(
            f"H3 full weight sync cannot map {len(unmapped)} names (first 5: {sorted(unmapped)[:5]}); "
            f"incomplete qkv trios: {sorted(incomplete)[:5]}"
        )

    qkv_pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, value in state_dict.items():
        target, op = map_h3_full_name(name)
        if op == "qkv":
            _sgld_prefix, which = _qkv_group_key(name)
            trio = qkv_pending.setdefault(target, {})
            trio[which] = prepare(value)
            if len(trio) == 3:
                yield target, _fuse_qkv_grouped(trio, target)
                del qkv_pending[target]
            continue
        tensor = prepare(value)
        yield target, _swap_gated_ffn_halves(tensor) if op == "ffn_swap" else tensor


__all__ = ["collect_h3_full_weight_tensors", "collect_h3_lora_layer_groups", "map_h3_full_name"]
