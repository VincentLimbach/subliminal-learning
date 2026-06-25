import argparse
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


GHOST_COUNTS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
SEEDS = [0, 1, 2, 3, 4]
TEACHER_DIR = "finetuning_A_readouts_nonfrozen"
CONDITION_DIR = "logit_distilation_B_readouts_nonfrozen"
DATA_LABEL = "data1"
T90_DF4 = 2.132

LOWER_SETUPS = [
    ("last_shared_init", 0.000, "#808080"),
    ("lower_interp_0p125", 0.125, "#0072B2"),
    ("lower_interp_0p25", 0.250, "#56B4E9"),
    ("lower_interp_0p375", 0.375, "#009E73"),
    ("lower_interp_0p5", 0.500, "#F0E442"),
    ("lower_interp_0p625", 0.625, "#E69F00"),
    ("lower_interp_0p75", 0.750, "#D55E00"),
    ("lower_interp_0p875", 0.875, "#CC79A7"),
    ("all_shared_init", 1.000, "#6b4ea1"),
]

READOUT_SETUPS = [
    ("readout_interp_0p0", 0.000, "#808080"),
    ("readout_interp_0p125", 0.125, "#0072B2"),
    ("readout_interp_0p25", 0.250, "#56B4E9"),
    ("readout_interp_0p375", 0.375, "#009E73"),
    ("readout_interp_0p5", 0.500, "#F0E442"),
    ("readout_interp_0p625", 0.625, "#E69F00"),
    ("readout_interp_0p75", 0.750, "#D55E00"),
    ("readout_interp_0p875", 0.875, "#CC79A7"),
    ("all_shared_init", 1.000, "#6b4ea1"),
]


def read_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df.iloc[-1]["accuracy_mean"])


def collect(root: Path, sweep: str, setups: list[tuple[str, float, str]]) -> pd.DataFrame:
    rows = []
    sweep_root = root / sweep
    for setup, alpha, color in setups:
        for ghost_count in GHOST_COUNTS:
            values = []
            for seed in SEEDS:
                path = (
                    sweep_root
                    / f"seed{seed}"
                    / setup
                    / TEACHER_DIR
                    / CONDITION_DIR
                    / DATA_LABEL
                    / f"logits{ghost_count}"
                    / "metrics.csv"
                )
                value = read_accuracy(path)
                if value is not None:
                    values.append(value)
            if values:
                series = pd.Series(values, dtype=float)
                mean = float(series.mean())
                std = float(series.std(ddof=1)) if len(values) > 1 else 0.0
                ci90 = T90_DF4 * std / sqrt(len(values)) if len(values) > 1 else 0.0
            else:
                mean = float("nan")
                std = float("nan")
                ci90 = float("nan")
            rows.append(
                {
                    "sweep": sweep,
                    "setup": setup,
                    "alpha": alpha,
                    "ghost_logits": ghost_count,
                    "n": len(values),
                    "mean_accuracy": mean,
                    "std_accuracy": std,
                    "ci90_accuracy": ci90,
                    "status": "complete" if len(values) == len(SEEDS) else "partial" if values else "missing",
                }
            )
    return pd.DataFrame(rows)


def plot_one(ax, df: pd.DataFrame, setups: list[tuple[str, float, str]], title: str) -> None:
    for setup, alpha, color in setups:
        sub = df[(df["setup"] == setup) & (df["n"] > 0)].sort_values("ghost_logits")
        if sub.empty:
            continue
        xs = sub["ghost_logits"].to_numpy(dtype=float)
        ys = sub["mean_accuracy"].to_numpy(dtype=float)
        ci = sub["ci90_accuracy"].to_numpy(dtype=float)
        label = f"alpha={alpha:.3f}"
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, color=color, label=label)
        ax.fill_between(xs, ys - ci, ys + ci, color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.10, color="black", linestyle=":", linewidth=1.6, label="chance 10%")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("ghost logits")
    ax.set_ylabel("final test accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("main_experiments/mnist_runs/replications_5/interpolations"))
    parser.add_argument("--out-dir", type=Path, default=Path("main_experiments/mnist_runs/replications_5/interpolations/plots"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lower_df = collect(args.root, "lower_interp", LOWER_SETUPS)
    readout_df = collect(args.root, "readout_interp", READOUT_SETUPS)
    combined = pd.concat([lower_df, readout_df], ignore_index=True)
    combined.to_csv(args.out_dir / "interp_replications5_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 6.4), sharey=True)
    plot_one(axes[0], lower_df, LOWER_SETUPS, "Non-final layer interpolation")
    plot_one(axes[1], readout_df, READOUT_SETUPS, "Final readout interpolation")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Interpolation sweeps, full data, trainable readouts, five seeds", y=0.98)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    out_path = args.out_dir / "interp_replications5_accuracy.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")
    missing = combined[combined["status"] != "complete"]
    print(f"complete cells: {(combined['status'] == 'complete').sum()} / {len(combined)}")
    if not missing.empty:
        print(f"partial or missing cells: {len(missing)}")


if __name__ == "__main__":
    main()
