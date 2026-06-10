#!/usr/bin/env python3
"""Overlay baseline vs refactor training curves for first-N-step alignment check."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def col(rows: list[dict], key: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def limit(rows: list[dict], n: int | None) -> list[dict]:
    return rows if n is None else rows[:n]


def overlay(ax, base_x, base_y, ref_x, ref_y, title, xlabel, ylabel=""):
    ax.plot(base_x, base_y, "o-", ms=4, lw=1.4, color="#1f77b4", label="baseline (main)")
    ax.plot(ref_x, ref_y, "s--", ms=4, lw=1.4, color="#d62728", label="refactor")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True, help="baseline metrics dir")
    ap.add_argument("--refactor", type=Path, required=True, help="refactor metrics dir")
    ap.add_argument("--out", type=Path, required=True, help="output dir for comparison pngs")
    ap.add_argument("--n-rollouts", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    b_rw = limit(read_csv(args.baseline / "reward_rollouts.csv"), args.n_rollouts)
    r_rw = limit(read_csv(args.refactor / "reward_rollouts.csv"), args.n_rollouts)
    b_tr = limit(read_csv(args.baseline / "train_steps.csv"), args.n_steps)
    r_tr = limit(read_csv(args.refactor / "train_steps.csv"), args.n_steps)

    saved = []

    # --- reward raw mean ---
    if b_rw and r_rw:
        fig, ax = plt.subplots(figsize=(8, 4))
        overlay(ax, col(b_rw, "rollout_id"), col(b_rw, "raw_mean"),
                col(r_rw, "rollout_id"), col(r_rw, "raw_mean"),
                "OCR reward raw mean (per rollout)", "rollout_id", "reward")
        fig.tight_layout(); p = args.out / "cmp_reward_raw_mean.png"; fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

        fig, ax = plt.subplots(figsize=(8, 4))
        overlay(ax, col(b_rw, "rollout_id"), col(b_rw, "raw_std"),
                col(r_rw, "rollout_id"), col(r_rw, "raw_std"),
                "OCR reward raw std (per rollout)", "rollout_id", "std")
        fig.tight_layout(); p = args.out / "cmp_reward_raw_std.png"; fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

        fig, ax = plt.subplots(figsize=(8, 4))
        overlay(ax, col(b_rw, "rollout_id"), col(b_rw, "norm_std"),
                col(r_rw, "rollout_id"), col(r_rw, "norm_std"),
                "Normalized reward std (per rollout)", "rollout_id", "norm_std")
        fig.tight_layout(); p = args.out / "cmp_reward_norm_std.png"; fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # --- train metrics ---
    train_keys = [
        ("train/loss", "Training loss"),
        ("train/loss_abs_mean", "Loss abs mean"),
        ("train/approx_kl", "Approx KL"),
        ("train/clipfrac", "Clip fraction"),
        ("train/grad_norm", "Grad norm"),
        ("train/adv_abs_mean", "Advantage abs mean"),
        ("train/ratio_abs_minus_1", "|ratio - 1|"),
    ]
    if b_tr and r_tr:
        for key, title in train_keys:
            by = col(b_tr, key); ry = col(r_tr, key)
            if all(v != v for v in by) and all(v != v for v in ry):
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            overlay(ax, col(b_tr, "train_step"), by, col(r_tr, "train_step"), ry, title, "train_step")
            fig.tight_layout()
            p = args.out / f"cmp_{key.replace('/', '_')}.png"
            fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # --- model-output consistency (train forward vs rollout) — names differ across branches ---
    mo_pairs = [
        ("model_output_mean_abs_diff", "train/model_output_mean_abs_diff", "train/rollout_step_model_output_mean_abs_diff", "Train vs rollout model_output |Δ| mean"),
        ("model_output_rel_max", "train/model_output_rel_max", "train/rollout_step_model_output_rel_max", "Train vs rollout model_output rel max"),
    ]
    if b_tr and r_tr:
        for fname, bkey, rkey, title in mo_pairs:
            by = col(b_tr, bkey); ry = col(r_tr, rkey)
            if all(v != v for v in by) and all(v != v for v in ry):
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            overlay(ax, col(b_tr, "train_step"), by, col(r_tr, "train_step"), ry, title, "train_step")
            fig.tight_layout()
            p = args.out / f"cmp_{fname}.png"
            fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # --- dashboard ---
    if b_rw and r_rw and b_tr and r_tr:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        overlay(axes[0, 0], col(b_rw, "rollout_id"), col(b_rw, "raw_mean"),
                col(r_rw, "rollout_id"), col(r_rw, "raw_mean"), "Reward raw mean", "rollout_id")
        overlay(axes[0, 1], col(b_tr, "train_step"), col(b_tr, "train/loss"),
                col(r_tr, "train_step"), col(r_tr, "train/loss"), "Training loss", "train_step")
        overlay(axes[1, 0], col(b_tr, "train_step"), col(b_tr, "train/clipfrac"),
                col(r_tr, "train_step"), col(r_tr, "train/clipfrac"), "Clip fraction", "train_step")
        overlay(axes[1, 1], col(b_tr, "train_step"), col(b_tr, "train/grad_norm"),
                col(r_tr, "train_step"), col(r_tr, "train/grad_norm"), "Grad norm", "train_step")
        fig.suptitle("baseline (main) vs refactor — first-step alignment", fontsize=12)
        fig.tight_layout()
        p = args.out / "cmp_dashboard.png"; fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    print(f"saved {len(saved)} comparison figures to {args.out}")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
