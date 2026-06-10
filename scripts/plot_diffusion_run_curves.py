#!/usr/bin/env python3
"""Parse Ray worker logs for a miles-diffusion run and plot training curves."""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

RUN_MARKER = "diffusion_grpo_ocr_2gpu_flowgrpo_aligned_20260604_074438"

TRAIN_STEP_RE = re.compile(
    r"\[train step (\d+)\] rollout=(\d+) (.+)$"
)
REWARD_STATS_RE = re.compile(
    r"\[reward stats\] raw mean=([\d.-]+) std=([\d.-]+) min=([\d.-]+) max=([\d.-]+) "
    r"\| normalized mean=([\d.-]+) std=([\d.-]+) min=([\d.-]+) max=([\d.-]+)"
)
PERF_RE = re.compile(r"perf (\d+): (\{.+?\})$")


def _parse_kv_blob(blob: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in blob.split():
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        out[key] = float(val)
    return out


def find_ray_logs(ray_session: Path | None, run_marker: str) -> tuple[Path, Path] | None:
    if ray_session is None:
        candidates = sorted(Path("/tmp/ray").glob("session_*"), reverse=True)
    else:
        candidates = [ray_session]
    for session in candidates:
        logs = session / "logs"
        if not logs.is_dir():
            continue
        train_log = perf_log = reward_log = None
        for path in logs.iterdir():
            text = ""
            if path.suffix in {".err", ".out"}:
                try:
                    text = path.read_text(errors="ignore")
                except OSError:
                    continue
            if run_marker not in text:
                continue
            if "[train step" in text and path.suffix == ".err":
                train_log = path
            if "[reward stats]" in text and path.suffix == ".out":
                reward_log = path
        if train_log is None:
            continue
        for path in logs.iterdir():
            if path == train_log:
                perf_log = path
                break
        if reward_log is None:
            for path in logs.iterdir():
                if path.suffix != ".out":
                    continue
                try:
                    text = path.read_text(errors="ignore")
                except OSError:
                    continue
                if "[reward stats]" in text:
                    reward_log = path
                    break
        return train_log, reward_log
    return None


def parse_train_log(path: Path) -> tuple[list[dict], list[dict]]:
    train_rows: list[dict] = []
    perf_rows: list[dict] = []
    for line in path.read_text(errors="ignore").splitlines():
        if m := TRAIN_STEP_RE.search(line):
            step, rollout, rest = int(m.group(1)), int(m.group(2)), m.group(3)
            row = {"train_step": step, "rollout_id": rollout}
            row.update(_parse_kv_blob(rest))
            train_rows.append(row)
        elif m := PERF_RE.search(line):
            rollout = int(m.group(1))
            perf = {k: float(v) for k, v in ast.literal_eval(m.group(2)).items()}
            perf["rollout_id"] = rollout
            perf_rows.append(perf)
    return train_rows, perf_rows


def parse_reward_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(errors="ignore").splitlines():
        if m := REWARD_STATS_RE.search(line):
            rows.append(
                {
                    "rollout_id": len(rows),
                    "raw_mean": float(m.group(1)),
                    "raw_std": float(m.group(2)),
                    "raw_min": float(m.group(3)),
                    "raw_max": float(m.group(4)),
                    "norm_mean": float(m.group(5)),
                    "norm_std": float(m.group(6)),
                    "norm_min": float(m.group(7)),
                    "norm_max": float(m.group(8)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_series(
    ax,
    xs,
    ys_dict: dict[str, list],
    title: str,
    xlabel: str,
    ylabel: str = "",
) -> None:
    for label, ys in ys_dict.items():
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def plot_all(out_dir: Path, train_rows: list[dict], perf_rows: list[dict], reward_rows: list[dict]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    if reward_rows:
        xs = [r["rollout_id"] for r in reward_rows]
        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_series(
            ax,
            xs,
            {
                "raw mean": [r["raw_mean"] for r in reward_rows],
                "raw std": [r["raw_std"] for r in reward_rows],
            },
            "OCR reward (rollout)",
            "rollout_id",
            "reward",
        )
        p = out_dir / "reward_raw.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_series(
            ax,
            xs,
            {
                "norm std": [r["norm_std"] for r in reward_rows],
            },
            "Normalized reward spread (rollout)",
            "rollout_id",
        )
        p = out_dir / "reward_norm_std.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    if train_rows:
        xs = [r["train_step"] for r in train_rows]

        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_series(
            ax,
            xs,
            {
                "policy_loss": [r.get("train/policy_loss", float("nan")) for r in train_rows],
                "loss": [r.get("train/loss", float("nan")) for r in train_rows],
                "loss_abs_mean": [r.get("train/loss_abs_mean", float("nan")) for r in train_rows],
            },
            "Training loss",
            "train_step",
        )
        p = out_dir / "train_loss.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_series(
            ax,
            xs,
            {
                "clipfrac": [r.get("train/clipfrac", float("nan")) for r in train_rows],
                "approx_kl": [r.get("train/approx_kl", float("nan")) for r in train_rows],
                "grad_norm": [r.get("train/grad_norm", float("nan")) for r in train_rows],
            },
            "GRPO diagnostics",
            "train_step",
        )
        p = out_dir / "train_grpo.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_series(
            ax,
            xs,
            {
                "model_output_mean_abs_diff": [
                    r.get("train/model_output_mean_abs_diff", float("nan")) for r in train_rows
                ],
                "model_output_rel_max": [r.get("train/model_output_rel_max", float("nan")) for r in train_rows],
            },
            "Train vs rollout model output diff (debug)",
            "train_step",
        )
        p = out_dir / "train_model_output_diff.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    if perf_rows:
        xs = [r["rollout_id"] for r in perf_rows]
        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_series(
            ax,
            xs,
            {
                "step_time (s)": [r.get("perf/step_time", float("nan")) for r in perf_rows],
                "train_time (s)": [r.get("perf/train_time", float("nan")) for r in perf_rows],
                "train_wait_time (s)": [r.get("perf/train_wait_time", float("nan")) for r in perf_rows],
            },
            "Per-rollout timing",
            "rollout_id",
            "seconds",
        )
        p = out_dir / "perf_timing.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(xs, [r.get("perf/wait_time_ratio", float("nan")) for r in perf_rows], marker="o", markersize=3)
        ax.set_title("Rollout wait fraction")
        ax.set_xlabel("rollout_id")
        ax.set_ylabel("wait_time_ratio")
        ax.grid(True, alpha=0.3)
        p = out_dir / "perf_wait_ratio.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    # Combined dashboard
    if train_rows and reward_rows:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        tr_x = [r["train_step"] for r in train_rows]
        rw_x = [r["rollout_id"] for r in reward_rows]
        axes[0, 0].plot(rw_x, [r["raw_mean"] for r in reward_rows], "o-", ms=3)
        axes[0, 0].set_title("Reward raw mean")
        axes[0, 0].set_xlabel("rollout_id")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(tr_x, [r.get("train/policy_loss", 0) for r in train_rows], "o-", ms=3)
        axes[0, 1].set_title("Policy loss")
        axes[0, 1].set_xlabel("train_step")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(tr_x, [r.get("train/clipfrac", 0) for r in train_rows], "o-", ms=3)
        axes[1, 0].set_title("Clip fraction")
        axes[1, 0].set_xlabel("train_step")
        axes[1, 0].grid(True, alpha=0.3)
        if perf_rows:
            pf_x = [r["rollout_id"] for r in perf_rows]
            axes[1, 1].plot(pf_x, [r.get("perf/step_time", 0) for r in perf_rows], "o-", ms=3)
            axes[1, 1].set_title("Step time (s)")
            axes[1, 1].set_xlabel("rollout_id")
            axes[1, 1].grid(True, alpha=0.3)
        fig.suptitle(RUN_MARKER, fontsize=10)
        fig.tight_layout()
        p = out_dir / "summary_dashboard.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(f"/root/miles_diffusion/logs/{RUN_MARKER}"),
    )
    parser.add_argument("--ray-session", type=Path, default=None)
    args = parser.parse_args()

    found = find_ray_logs(args.ray_session, RUN_MARKER)
    if found is None:
        raise SystemExit(f"No Ray logs found containing checkpoint path for {RUN_MARKER}")
    train_log, reward_log = found
    if reward_log is None:
        raise SystemExit("Found train log but no reward stats log")

    train_rows, perf_rows = parse_train_log(train_log)
    reward_rows = parse_reward_log(reward_log)

    metrics_dir = args.run_dir / "metrics"
    plots_dir = args.run_dir / "plots"
    write_csv(metrics_dir / "train_steps.csv", train_rows)
    write_csv(metrics_dir / "perf_rollouts.csv", perf_rows)
    write_csv(metrics_dir / "reward_rollouts.csv", reward_rows)

    (args.run_dir / "metrics" / "sources.txt").write_text(
        f"train_log={train_log}\nreward_log={reward_log}\n",
        encoding="utf-8",
    )

    saved = plot_all(plots_dir, train_rows, perf_rows, reward_rows)
    print(f"Parsed {len(train_rows)} train steps, {len(reward_rows)} rollouts, {len(perf_rows)} perf rows")
    print(f"Wrote CSV to {metrics_dir}")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
