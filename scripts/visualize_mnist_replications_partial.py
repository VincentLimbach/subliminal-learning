import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_mnist_experiment import DEVICE, MultiClassifier, get_mnist
from run_mnist_readout_reinit_grid_job import MAX_GHOST_LOGITS, to_tensor

SETUPS = ["all_shared_init", "none_shared_init", "last_shared_init", "last_shared_inherit"]
SETUP_LABELS = {
    "all_shared_init": "All shared init",
    "none_shared_init": "None shared init",
    "last_shared_init": "Last shared init",
    "last_shared_inherit": "Last shared inherit",
}
COLORS = {
    "teacher": "#8c6d62",
    "all_shared_init": "#8f6db8",
    "none_shared_init": "#9c9c9c",
    "last_shared_init": "#b8a2d9",
    "last_shared_inherit": "#6b4ea1",
}
TEACHER_ORDER = ["nonfrozen", "frozen"]
CONDITION_ORDER = ["nonfrozen", "frozen", "projected"]
TEACHER_LABELS = {"nonfrozen": "Teacher class A trainable", "frozen": "Teacher class A frozen"}
CONDITION_LABELS = {
    "nonfrozen": "Class B logits trainable",
    "frozen": "Class B readouts frozen",
    "projected": "Projected latent",
}
DATA_STYLES = {
    0.1: dict(color="C0", marker="o", linestyle="-"),
    0.5: dict(color="C1", marker="s", linestyle="--"),
    1.0: dict(color="C2", marker="^", linestyle="-."),
}


def parse_path(path: Path, root: Path):
    rel = path.relative_to(root).parts
    # setup/seedX/finetuning_A_readouts_*/condition/dataY/logitsZ/metrics.csv
    setup = rel[0]
    seed = int(rel[1].removeprefix("seed"))
    teacher = "frozen" if rel[2].endswith("_frozen") else "nonfrozen"
    condition_dir = rel[3]
    if condition_dir == "latent_projection_distilation_B":
        condition = "projected"
    elif condition_dir.endswith("_frozen"):
        condition = "frozen"
    else:
        condition = "nonfrozen"
    data_fraction = float(rel[4].removeprefix("data"))
    ghost = int(rel[5].removeprefix("logits"))
    return setup, seed, teacher, condition, data_fraction, ghost


def load_final_rows(root: Path):
    rows = []
    for path in sorted(root.glob("*/seed*/finetuning_A_readouts_*/*/data*/logits*/metrics.csv")):
        try:
            setup, seed, teacher, condition, frac, ghost = parse_path(path, root)
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"skip unreadable {path}: {exc}")
            continue
        if df.empty:
            continue
        final = df.loc[df["epoch"].idxmax()].to_dict()
        final.update(
            setup=setup,
            seed=seed,
            teacher_readout=teacher,
            condition=condition,
            data_fraction=frac,
            num_ghost_logits=ghost,
            source_file=str(path.relative_to(root)),
        )
        rows.append(final)
    if not rows:
        raise FileNotFoundError(f"No metrics.csv files found below {root}")
    return pd.DataFrame(rows)


def mean_ci90(values):
    values = pd.Series(values).dropna().astype(float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(values.mean())
    if n <= 1:
        return mean, 0.0, n
    ci = 1.645 * float(values.std(ddof=1)) / math.sqrt(n)
    return mean, ci, n


def summarize(final_rows: pd.DataFrame):
    keys = ["setup", "teacher_readout", "condition", "data_fraction", "num_ghost_logits"]
    records = []
    metric_cols = [
        "accuracy_mean",
        "hidden1_cosine_mean",
        "hidden2_cosine_mean",
        "class_readout_weight_cosine_mean",
        "ghost_readout_weight_cosine_mean",
    ]
    for key, part in final_rows.groupby(keys):
        rec = dict(zip(keys, key))
        rec["n_seeds"] = int(part["seed"].nunique())
        for col in metric_cols:
            if col in part:
                mean, ci, n = mean_ci90(part[col])
                rec[col] = mean
                rec[f"{col}_ci90"] = ci
                rec[f"{col}_n"] = n
        records.append(rec)
    return pd.DataFrame(records).sort_values(keys)


@t.inference_mode()
def teacher_accuracy(root: Path, teacher_readout="nonfrozen"):
    teacher_dir = "finetuning_A_readouts_frozen" if teacher_readout == "frozen" else "finetuning_A_readouts_nonfrozen"
    model_paths = sorted((root / "last_shared_inherit").glob(f"seed*/{teacher_dir}/teacher_artifacts/model.pt"))
    if not model_paths:
        return np.nan
    _, test_ds = get_mnist()
    test_x_s, test_y = to_tensor(test_ds)
    test_x = test_x_s.unsqueeze(0)
    values = []
    for model_path in model_paths:
        payload = t.load(model_path, map_location=DEVICE)
        model = MultiClassifier(1, [28 * 28, 256, 256, 10 + MAX_GHOST_LOGITS]).to(DEVICE)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        pred = model(test_x)[0, :, :10].argmax(-1)
        values.append(float((pred == test_y).float().mean().cpu()))
    return float(np.mean(values))


def plot_setup_2x3(summary: pd.DataFrame, setup: str, metric: str, ylabel: str, out_path: Path, ylim=None):
    setup_df = summary[summary["setup"] == setup]
    if setup_df.empty:
        return
    ghosts = sorted(summary["num_ghost_logits"].dropna().unique())
    ticks = [g for g in ghosts if g in {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.4), sharex=True, sharey=ylim is not None)
    for r, teacher in enumerate(TEACHER_ORDER):
        for c, condition in enumerate(CONDITION_ORDER):
            ax = axes[r][c]
            part = setup_df[(setup_df["teacher_readout"] == teacher) & (setup_df["condition"] == condition)]
            for frac in sorted(part["data_fraction"].dropna().unique()):
                p = part[part["data_fraction"] == frac].sort_values("num_ghost_logits")
                if p.empty or metric not in p:
                    continue
                style = DATA_STYLES.get(float(frac), dict(color="0.2", marker="o", linestyle="-"))
                x = p["num_ghost_logits"].to_numpy(dtype=float)
                y = p[metric].to_numpy(dtype=float)
                ci_col = f"{metric}_ci90"
                yerr = p[ci_col].to_numpy(dtype=float) if ci_col in p else np.zeros_like(y)
                label = f"data={frac:g}"
                ax.plot(x, y, linewidth=2.0, markersize=4.5, label=label, **style)
                ax.fill_between(x, y - yerr, y + yerr, color=style["color"], alpha=0.14, linewidth=0)
            if metric == "accuracy_mean":
                ax.axhline(0.10, color="0.25", linestyle=":", linewidth=1.3, label="chance 10%" if r == 0 and c == 0 else None)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(x)) for x in ticks])
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(CONDITION_LABELS[condition])
            if c == 0:
                ax.set_ylabel(f"{TEACHER_LABELS[teacher]}\n{ylabel}")
            ax.set_xlabel("ghost logits")
    handles, labels = [], []
    seen = set()
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=min(4, len(labels)), frameon=False)
    fig.suptitle(f"{SETUP_LABELS.get(setup, setup)}: {ylabel} (partial, mean +/- 90% CI over completed seeds)", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_figure10b(summary: pd.DataFrame, out_path: Path, condition: str, condition_label: str, objective_label: str, teacher_acc=np.nan):
    part = summary[(summary["teacher_readout"] == "nonfrozen") & (summary["condition"] == condition) & (summary["data_fraction"] == 1.0)]
    if part.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for setup in SETUPS:
        p = part[part["setup"] == setup].sort_values("num_ghost_logits")
        if p.empty:
            continue
        x = p["num_ghost_logits"].to_numpy(dtype=float)
        y = p["accuracy_mean"].to_numpy(dtype=float)
        ci = p["accuracy_mean_ci90"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2, label=SETUP_LABELS.get(setup, setup), color=COLORS[setup])
        ax.fill_between(x, y - ci, y + ci, color=COLORS[setup], alpha=0.12, linewidth=0)
    if not np.isnan(teacher_acc):
        ax.axhline(teacher_acc, color=COLORS["teacher"], linewidth=2, linestyle="--", label=f"Teacher ({teacher_acc:.3f})")
    ax.axhline(0.10, color="black", linestyle=":", linewidth=1.8, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("ghost logits")
    ax.set_ylabel("Final test accuracy")
    ax.set_title(f"Full data, {condition_label}, {objective_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_completeness(summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), sharex=True, sharey=True)
    for ax, setup in zip(axes.flat, SETUPS):
        part = summary[summary["setup"] == setup]
        table = part.pivot_table(index="data_fraction", columns="num_ghost_logits", values="n_seeds", aggfunc="max").sort_index()
        if table.empty:
            ax.set_title(SETUP_LABELS.get(setup, setup))
            continue
        im = ax.imshow(table.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=20, cmap="viridis")
        ax.set_title(SETUP_LABELS.get(setup, setup))
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([f"{x:g}" for x in table.index])
        cols = list(table.columns)
        tick_idx = [i for i, g in enumerate(cols) if g in {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(int(cols[i])) for i in tick_idx], rotation=45, ha="right")
        ax.set_xlabel("ghost logits")
        ax.set_ylabel("data fraction")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, label="completed seeds")
    fig.suptitle("Replication completeness by setup (max 20 seeds)", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.96, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.results_dir / "partial_viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    final_rows = load_final_rows(args.results_dir)
    summary = summarize(final_rows)
    final_rows.to_csv(out_dir / "partial_final_rows.csv", index=False)
    summary.to_csv(out_dir / "partial_summary_by_seed_mean.csv", index=False)

    overview_dir = out_dir / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    by_setup_dir = out_dir / "by_setup"
    by_setup_dir.mkdir(parents=True, exist_ok=True)

    plot_completeness(summary, overview_dir / "completeness.png")
    teacher_acc = teacher_accuracy(args.results_dir, "nonfrozen")
    plot_figure10b(summary, overview_dir / "figure10b_full_data_nonfrozen_summary.png", "nonfrozen", "trainable student readouts", "no projection", teacher_acc)
    plot_figure10b(summary, overview_dir / "figure10b_full_data_projected_summary.png", "projected", "trainable student readouts", "projected latent", teacher_acc)
    plot_figure10b(summary, overview_dir / "figure10b_full_data_student_readouts_frozen_summary.png", "frozen", "frozen student readouts", "no projection", teacher_acc)
    metrics = [
        ("accuracy_mean", "final accuracy", (0, 1)),
        ("hidden2_cosine_mean", "final latent cosine", (0, 1)),
        ("hidden1_cosine_mean", "hidden1 cosine", (0, 1)),
        ("class_readout_weight_cosine_mean", "class readout cosine", (0.9, 1.1)),
        ("ghost_readout_weight_cosine_mean", "ghost readout cosine", (0.9, 1.1)),
    ]
    for setup in SETUPS:
        setup_dir = by_setup_dir / setup
        setup_dir.mkdir(parents=True, exist_ok=True)
        for metric, label, ylim in metrics:
            if metric in summary:
                plot_setup_2x3(summary, setup, metric, label, setup_dir / f"{metric}.png", ylim=ylim)

    print(f"loaded_runs={len(final_rows)}")
    print(f"completed_cells={len(summary)}")
    print(f"complete_20_seed_cells={(summary['n_seeds'] >= 20).sum()}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
