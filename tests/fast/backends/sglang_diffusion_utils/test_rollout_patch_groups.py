from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest

import miles.backends.sglang_diffusion_utils.monkey_patches as mp


@pytest.fixture
def isolated_registry(monkeypatch):
    # Swap the registry for a copy (auto-restored) so the real public decorator
    # can be exercised without leaking the dummy group into other tests.
    monkeypatch.setattr(mp, "_ROLLOUT_PATCH_APPLIERS", dict(mp._ROLLOUT_PATCH_APPLIERS))


class TestRolloutPatchGroups:
    # End-to-end group mechanics: a dummy group registered through the public
    # decorator is applied iff its name is listed in MILES_ROLLOUT_PATCH_GROUPS.
    def test_dummy_group_applied_when_selected(self, isolated_registry, monkeypatch):
        calls = []

        @mp.register_rollout_patch_group("dummy")
        def apply_dummy() -> None:
            calls.append("dummy")

        monkeypatch.setenv(mp.ROLLOUT_PATCH_GROUPS_ENV, "dummy")
        mp.apply_env_selected_rollout_patches()
        # Only the selected group ran (built-in appliers would import sglang and fail here).
        assert calls == ["dummy"]

    def test_no_selection_applies_nothing(self, isolated_registry, monkeypatch):
        calls = []
        mp.register_rollout_patch_group("dummy")(lambda: calls.append("dummy"))
        monkeypatch.delenv(mp.ROLLOUT_PATCH_GROUPS_ENV, raising=False)
        mp.apply_env_selected_rollout_patches()
        assert calls == []

    def test_unknown_group_fails_loud(self, monkeypatch):
        monkeypatch.setenv(mp.ROLLOUT_PATCH_GROUPS_ENV, "bogus")
        with pytest.raises(ValueError, match="Unknown rollout patch group"):
            mp.apply_env_selected_rollout_patches()

    def test_builtin_group_registered(self):
        # The decorator ran at import time for the in-repo group.
        assert "sgld" in mp._ROLLOUT_PATCH_APPLIERS

    def test_lora_parity_group_registered(self):
        # Its applier imports sglang lazily, so registration alone stays CPU-safe.
        assert "lora_parity" in mp._ROLLOUT_PATCH_APPLIERS


class TestValidateRolloutPatchGroups:
    # The arg-validation entry point behind --rollout-patch-group:
    #   --rollout-patch-group "sgld,ltx"  ──► registered appliers ──► pass
    #   --rollout-patch-group "sgld,bogus" ─► "bogus" unregistered ─► ValueError
    def test_known_pass_unknown_raises(self):
        mp.validate_rollout_patch_groups(["sgld", "ltx"])
        with pytest.raises(ValueError, match="Unknown rollout patch group"):
            mp.validate_rollout_patch_groups(["sgld", "bogus"])


class TestLoRAParitySelection:
    # The parity patches only make sense for an unmerged engine, so they ride the flag
    # that asks for one instead of a separate opt-in nobody would remember.
    @staticmethod
    def _parse(*extra):
        import sys

        from miles.utils.arguments import parse_args

        argv = [
            "test",
            "--hf-checkpoint",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "--prompt-data",
            "/dev/null",
            "--rollout-batch-size",
            "2",
            "--n-samples-per-prompt",
            "2",
            "--actor-num-nodes",
            "1",
            "--actor-num-gpus-per-node",
            "1",
            "--rollout-num-gpus",
            "1",
            "--num-rollout",
            "1",
            "--use-lora",
            "--lora-rank",
            "8",
            *extra,
        ]
        old = sys.argv
        sys.argv = argv
        try:
            return parse_args()
        finally:
            sys.argv = old

    def test_dynamic_merge_mode_selects_lora_parity(self):
        args = self._parse("--sglang-lora-merge-mode", "dynamic")
        assert "lora_parity" in args.rollout_patch_groups

    def test_explicitly_listed_groups_survive(self):
        args = self._parse("--sglang-lora-merge-mode", "dynamic", "--rollout-patch-group", "sgld")
        assert args.rollout_patch_groups == ["sgld", "lora_parity"]

    def test_default_merge_mode_leaves_it_out(self):
        args = self._parse()
        assert args.sglang_lora_merge_mode != "dynamic"
        assert "lora_parity" not in args.rollout_patch_groups
