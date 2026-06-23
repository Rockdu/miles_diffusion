# Downstream interface patches

`collate_cond_for_sample_batch` gained a `pad_to_len: int | None = None`
parameter in the base `TrainPipelineConfig` contract (so the flat-pair trainer
can ask every model config to pad text to one shared width for legacy
window-padding parity — see `tests/test_cond_collate_pad_to_len_interface.py`).

`Wan2.2` and `LTX` support land in **separate upstream PR branches**
(`feat/wan`, `feat/support_ltx`), so their config files are not on this branch
and cannot be edited here. Both use the same fixed-length `torch.cat` collate as
SD3, so they only need the one-line `pad_to_len=None` added to their signature —
behaviourally a no-op (they ignore it), purely so the uniform call site does not
`TypeError`.

Apply when rebasing those PRs onto a main that carries the base contract:

```bash
# on feat/wan
git apply patches/feat-wan__wan2_2_add_pad_to_len.patch

# on feat/support_ltx
git apply patches/feat-support-ltx__ltx_add_pad_to_len.patch
```

Each was verified with `git apply --check` against its target branch at the time
of writing. If the surrounding lines have since moved, re-apply by hand: add
`pad_to_len: int | None = None` after the `device: torch.device,` parameter of
that file's `collate_cond_for_sample_batch`.
