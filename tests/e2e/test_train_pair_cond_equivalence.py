"""GPU equivalence tests for the flat train-pair training path.

These need a real DiT forward, so they are registered as CUDA tests and are NOT
wired into any PR stage (there is no GPU runner). Run locally on a GPU box:

    pytest tests/e2e/test_train_pair_cond_equivalence.py -v

Two kinds of checks:
  * Exact (torch.equal) where the comparison keeps the batch size and the inputs
    identical -- conditioning assembly, and a forward driven by identical tensors.
  * Approximate (torch.allclose) where the comparison spans different batch sizes:
    on GPU, cuBLAS picks different kernels / reduction orders per shape, so two
    mathematically-equal results differ at fp rounding (~1e-6). Exact equality is
    not attainable there for correct code.

A tiny (2-layer, random-weight) QwenImageTransformer2DModel is used -- no
checkpoint or data is needed; the tests check that the code paths agree, not
that the outputs are any good.
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils.configs.qwen_image import QwenImageTrainPipelineConfig
from miles.utils.types import CondKwargs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")

DEV = "cuda"
HEAD_DIM = 16
NUM_HEADS = 2
JOINT_DIM = 16  # encoder_hidden_states feature dim
IN_CH = 4
IMG_HW = (1, 4, 4)  # (F, H, W) in patches -> L_img = 16
L_IMG = IMG_HW[1] * IMG_HW[2]


def _tiny_qwen():
    from diffusers import QwenImageTransformer2DModel

    torch.manual_seed(0)
    return (
        QwenImageTransformer2DModel(
            patch_size=1,  # so output feature dim (out_channels * patch^2) == input in_channels
            in_channels=IN_CH,
            out_channels=IN_CH,
            num_layers=2,
            attention_head_dim=HEAD_DIM,
            num_attention_heads=NUM_HEADS,
            joint_attention_dim=JOINT_DIM,
            guidance_embeds=False,
            axes_dims_rope=(8, 4, 4),  # sum == attention_head_dim
        )
        .to(DEV)
        .eval()
        .to(torch.float32)
    )


def _cond_kwargs(seq_len: int, seed: int = 0) -> CondKwargs:
    """One sample's raw conditioning (text embedding of length seq_len)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    return CondKwargs(
        encoder_hidden_states=[torch.randn(seq_len, JOINT_DIM, device=DEV, generator=g)],
        txt_seq_lens=[seq_len],
        img_shapes=[[IMG_HW]],
    )


def _prepared(seq_len: int, seed: int = 0) -> dict:
    return QwenImageTrainPipelineConfig().prepare_cond_kwargs(_cond_kwargs(seq_len, seed), torch.device(DEV))


# --------- conditioning assembly: the two paths must build the same tensors ---------
class TestCondAssembly:
    @pytest.mark.parametrize("bsz", [1, 2, 4])
    def test_expand_equals_collate_for_one_sample(self, bsz):
        # When every pair in a micro-batch is the same sample, the cheap expand path
        # and the general collate path must produce the same encoder_hidden_states.
        cfg = QwenImageTrainPipelineConfig()
        prepared = _prepared(seq_len=5)
        expanded = cfg.expand_cond_for_timestep_batch(prepared, bsz)
        collated = cfg.collate_cond_for_sample_batch([prepared] * bsz, torch.device(DEV))

        assert torch.equal(
            expanded["encoder_hidden_states"].contiguous(),
            collated["encoder_hidden_states"].contiguous(),
        )
        assert tuple(collated["encoder_hidden_states"].shape) == (bsz, 5, JOINT_DIM)
        assert list(expanded["txt_seq_lens"]) == [5] * bsz
        assert collated["encoder_hidden_states_mask"].shape == (bsz, 5)
        assert bool(collated["encoder_hidden_states_mask"].all())  # full length -> all valid

    def test_collate_pads_and_masks_variable_lengths(self):
        cfg = QwenImageTrainPipelineConfig()
        collated = cfg.collate_cond_for_sample_batch(
            [_prepared(seq_len=3, seed=1), _prepared(seq_len=5, seed=2)], torch.device(DEV)
        )
        assert tuple(collated["encoder_hidden_states"].shape) == (2, 5, JOINT_DIM)
        expected_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool, device=DEV)
        assert torch.equal(collated["encoder_hidden_states_mask"], expected_mask)
        # padded positions of the shorter sample are exactly zero
        assert torch.equal(collated["encoder_hidden_states"][0, 3:, :], torch.zeros(2, JOINT_DIM, device=DEV))


# --------- a same-sample forward is identical whichever path assembled the cond ---------
class TestForwardUnderBothCondPaths:
    def test_forward_identical_for_expand_and_collate(self):
        model = _tiny_qwen()
        cfg = QwenImageTrainPipelineConfig()
        bsz, seq_len = 4, 5
        prepared = _prepared(seq_len=seq_len)
        expanded = cfg.expand_cond_for_timestep_batch(prepared, bsz)
        collated = cfg.collate_cond_for_sample_batch([prepared] * bsz, torch.device(DEV))

        torch.manual_seed(1)
        hs = torch.randn(bsz, L_IMG, IN_CH, device=DEV)
        ts = torch.full((bsz,), 0.5, device=DEV)
        mask = collated["encoder_hidden_states_mask"]  # all valid; fed to both to isolate the embedding

        def fwd(ehs):
            with torch.no_grad():
                return model(
                    hidden_states=hs,
                    encoder_hidden_states=ehs.contiguous(),
                    encoder_hidden_states_mask=mask,
                    timestep=ts,
                    img_shapes=[[IMG_HW]] * bsz,
                    txt_seq_lens=[seq_len] * bsz,
                    return_dict=False,
                )[0]

        assert torch.equal(fwd(expanded["encoder_hidden_states"]), fwd(collated["encoder_hidden_states"]))


# --------- batching different-length samples together must not change any of them ---------
class TestCrossSampleCollateMatchesSeparate:
    def test_padded_batch_matches_per_sample_forward(self):
        model = _tiny_qwen()
        cfg = QwenImageTrainPipelineConfig()
        a, b = _prepared(seq_len=3, seed=1), _prepared(seq_len=5, seed=2)

        torch.manual_seed(7)
        hs_a = torch.randn(1, L_IMG, IN_CH, device=DEV)
        hs_b = torch.randn(1, L_IMG, IN_CH, device=DEV)
        t_a, t_b = torch.tensor([0.3], device=DEV), torch.tensor([0.6], device=DEV)

        def fwd(hs, ts, cond):
            with torch.no_grad():
                return model(hidden_states=hs, timestep=ts, return_dict=False, **cond)[0]

        out_a = fwd(hs_a, t_a, a)  # length 3, no padding
        out_b = fwd(hs_b, t_b, b)  # length 5

        collated = cfg.collate_cond_for_sample_batch([a, b], torch.device(DEV))
        out_batched = fwd(torch.cat([hs_a, hs_b]), torch.cat([t_a, t_b]), collated)  # padded + masked

        # padding/masking must leave each sample's output unchanged (cross-batch -> tolerance)
        assert torch.allclose(out_batched[0:1], out_a, atol=1e-4, rtol=1e-3)
        assert torch.allclose(out_batched[1:2], out_b, atol=1e-4, rtol=1e-3)


# --------- classifier-free-guidance assembly + combine ---------
class TestClassifierFreeGuidance:
    def test_joint_pos_neg_collate_order_and_mask(self):
        # the joint CFG forward stacks all positive conds then all negative conds
        cfg = QwenImageTrainPipelineConfig()
        pos = [_prepared(3, 1), _prepared(5, 2)]
        neg = [_prepared(4, 3), _prepared(2, 4)]
        joint = cfg.collate_cond_for_sample_batch(pos + neg, torch.device(DEV))
        assert tuple(joint["encoder_hidden_states"].shape) == (4, 5, JOINT_DIM)
        # each row's valid length matches its source sample, in [pos..., neg...] order
        assert joint["encoder_hidden_states_mask"].sum(dim=1).tolist() == [3, 5, 4, 2]

    def test_cfg_combine_formula(self):
        cfg = QwenImageTrainPipelineConfig()
        torch.manual_seed(0)
        pos = torch.randn(2, 4, device=DEV)
        neg = torch.randn(2, 4, device=DEV)

        # guidance only (no true-cfg): uncond + scale * (cond - uncond), no rescale
        out = cfg.cfg_combine(pos, neg, guidance_scale=3.0, true_cfg_scale=None)
        assert torch.allclose(out, neg + 3.0 * (pos - neg))

        # true_cfg_scale > 1: also rescale the combined vector back to the positive norm
        out2 = cfg.cfg_combine(pos, neg, guidance_scale=0.0, true_cfg_scale=2.0)
        combined = neg + 2.0 * (pos - neg)
        pos_norm = torch.norm(pos, dim=-1, keepdim=True)
        combined_norm = torch.norm(combined, dim=-1, keepdim=True)
        assert torch.allclose(out2, combined * (pos_norm / combined_norm))


# --------- the per-pair forward must not depend on micro-batch size ---------
class TestForwardInvariantToMicroBatchSize:
    def test_per_pair_forward_invariant_to_micro_batch_size(self):
        model = _tiny_qwen()
        cfg = QwenImageTrainPipelineConfig()
        n, seq_len = 6, 5
        prepared = _prepared(seq_len=seq_len)

        torch.manual_seed(2)
        hs = torch.randn(n, L_IMG, IN_CH, device=DEV)  # distinct latent per pair
        ts = torch.rand(n, device=DEV)  # distinct timestep per pair

        def fwd_range(lo, hi):
            k = hi - lo
            ehs = cfg.expand_cond_for_timestep_batch(prepared, k)["encoder_hidden_states"].contiguous()
            with torch.no_grad():
                return model(
                    hidden_states=hs[lo:hi],
                    encoder_hidden_states=ehs,
                    encoder_hidden_states_mask=torch.ones(k, seq_len, dtype=torch.bool, device=DEV),
                    timestep=ts[lo:hi],
                    img_shapes=[[IMG_HW]] * k,
                    txt_seq_lens=[seq_len] * k,
                    return_dict=False,
                )[0]

        full = fwd_range(0, n)  # one batch of n
        for mbs in (1, 3, 4):
            chunked = torch.cat([fwd_range(lo, min(n, lo + mbs)) for lo in range(0, n, mbs)], dim=0)
            assert torch.allclose(full, chunked, atol=1e-5, rtol=1e-4), (
                f"per-pair forward changed with micro_batch_size={mbs}: "
                f"max_abs_diff={(full - chunked).abs().max().item():.2e}"
            )


# --------- driving the real _forward_train_pair_batch: gradient is mbs-invariant ---------
class _FakeFlowScheduler:
    """Minimal stand-in exposing only what sde_step_with_logprob touches."""

    def __init__(self, timesteps: torch.Tensor, sigmas: torch.Tensor):
        self.timesteps = timesteps
        self.sigmas = sigmas

    def index_for_timestep(self, t) -> int:
        return int(torch.argmin((self.timesteps - t).abs()).item())


def _fake_actor(model):
    return SimpleNamespace(
        _forward_dtype=torch.float32,
        train_pipeline_config=QwenImageTrainPipelineConfig(),
        model=model,
        args=SimpleNamespace(diffusion_adv_clip_max=10.0, fsdp_cfg_batching=False),
        scheduler=_FakeFlowScheduler(
            timesteps=torch.tensor([800.0, 600.0, 400.0, 200.0], device=DEV),
            sigmas=torch.tensor([0.8, 0.6, 0.4, 0.2, 0.0], device=DEV),
        ),
    )


class TestRealForwardGradientInvariance:
    def test_gradient_invariant_to_micro_batch_size(self):
        from miles.backends.fsdp_utils.actor import FSDPTrainRayActor

        model = _tiny_qwen()
        fake = _fake_actor(model)
        cond = _cond_kwargs(seq_len=5, seed=0)
        denv = SimpleNamespace(pos_cond_kwargs=cond, neg_cond_kwargs=None)
        timesteps = [800.0, 600.0, 400.0, 200.0]

        def make_pair(i):
            g = torch.Generator(device=DEV).manual_seed(100 + i)
            latent = torch.randn(L_IMG, IN_CH, device=DEV, generator=g)
            return {
                "latent": latent,
                "next_latent": latent + 0.1 * torch.randn(L_IMG, IN_CH, device=DEV, generator=g),
                "timestep": torch.tensor(timesteps[i], device=DEV),
                "log_prob_old": torch.tensor(0.0, device=DEV),
                "advantage": 0.5,
                "sample_index": 0,  # same sample -> expand path
                "denoising_env": denv,
            }

        pairs = [make_pair(i) for i in range(len(timesteps))]
        n = len(pairs)

        def grads_for(mbs):
            model.zero_grad(set_to_none=True)
            for lo in range(0, n, mbs):
                loss_sum = FSDPTrainRayActor._forward_train_pair_batch(
                    fake,
                    pairs[lo : min(n, lo + mbs)],
                    use_cfg=False,
                    guidance_scale=0.0,
                    true_cfg_scale=None,
                    clip_range=0.2,
                    noise_level=0.7,
                    num_train_timesteps=1000,
                    log_stats=defaultdict(list),
                    device=torch.device(DEV),
                    kl_beta=0.0,
                )
                (loss_sum / n).backward()
            return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}

        g_full = grads_for(n)  # whole window in one micro-batch
        g_split = grads_for(1)  # one pair per micro-batch
        assert g_full, "expected non-empty gradients"
        for name in g_full:
            assert torch.allclose(g_full[name], g_split[name], atol=1e-5, rtol=1e-3), name
