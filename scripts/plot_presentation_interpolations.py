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
CHANCE_ACCURACY = 0.10

# Same alpha colors as the exploration lower-interpolation plot.
LOWER_SETUPS = [
    ("last_shared_init", 0.000, "alpha=0.000: final init shared", "#808080"),
    ("lower_interp_0p125", 0.125, "alpha=0.125", "#0072B2"),
    ("lower_interp_0p25", 0.250, "alpha=0.250", "#56B4E9"),
    ("lower_interp_0p375", 0.375, "alpha=0.375", "#009E73"),
    ("lower_interp_0p5", 0.500, "alpha=0.500", "#F0E442"),
    ("lower_interp_0p625", 0.625, "alpha=0.625", "#E69F00"),
    ("lower_interp_0p75", 0.750, "alpha=0.750", "#D55E00"),
    ("lower_interp_0p875", 0.875, "alpha=0.875", "#CC79A7"),
    ("all_shared_init", 1.000, "alpha=1.000: all init shared", "#6b4ea1"),
]

READOUT_SETUPS = [
    ("readout_interp_0p0", 0.000, "alpha=0.000: lower init shared", "#808080"),
    ("readout_interp_0p125", 0.125, "alpha=0.125", "#0072B2"),
    ("readout_interp_0p25", 0.250, "alpha=0.250", "#56B4E9"),
    ("readout_interp_0p375", 0.375, "alpha=0.375", "#009E73"),
    ("readout_interp_0p5", 0.500, "alpha=0.500", "#F0E442"),
    ("readout_interp_0p625", 0.625, "alpha=0.625", "#E69F00"),
    ("readout_interp_0p75", 0.750, "alpha=0.750", "#D55E00"),
    ("readout_interp_0p875", 0.875, "alpha=0.875", "#CC79A7"),
    ("all_shared_init", 1.000, "alpha=1.000: all init shared", "#6b4ea1"),
]


def read_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df.iloc[-1]["accuracy_mean"])


def collect(root: Path, sweep: str, setups: list[tuple[str, float, str, str]]) -> pd.DataFrame:
    rows = []
    sweep_root = root / sweep
    for setup, alpha, label, color in setups:
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
                    "label": label,
                    "color": color,
                    "ghost_logits": ghost_count,
                    "n": len(values),
                    "mean_accuracy": mean,
                    "std_accuracy": std,
                    "ci90_accuracy": ci90,
                    "status": "complete" if len(values) == len(SEEDS) else "partial" if values else "missing",
                }
            )
    return pd.DataFrame(rows)


def plot_sweep(df: pd.DataFrame, setups: list[tuple[str, float, str, str]], title: str, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    for setup, _alpha, label, color in setups:
        sub = df[(df["setup"] == setup) & (df["n"] > 0)].sort_values("ghost_logits")
        if sub.empty:
            continue
        x = sub["ghost_logits"].to_numpy(dtype=float)
        y = sub["mean_accuracy"].to_numpy(dtype=float)
        ci = sub["ci90_accuracy"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.4, markersize=6.0, color=color, label=label)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.14, linewidth=0)

    ax.axhline(CHANCE_ACCURACY, color="black", linestyle=":", linewidth=2.0, label="chance 10%")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("ghost logits")
    ax.set_ylabel("final test accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.30)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Presentation plots for five-replica interpolation experiments.")
    parser.add_argument("--root", type=Path, default=Path("main_experiments/mnist_runs/replications_5/interpolations"))
    parser.add_argument("--out-dir", type=Path, default=Path("main_experiments/mnist_runs/presentation/plots"))
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lower_df = collect(args.root, "lower_interp", LOWER_SETUPS)
    readout_df = collect(args.root, "readout_interp", READOUT_SETUPS)
    combined = pd.concat([lower_df, readout_df], ignore_index=True)
    csv_path = args.out_dir / "interpolation_replications5_summary.csv"
    combined.to_csv(csv_path, index=False)

    lower_path = args.out_dir / "interpolation_lower_layers_replications5.png"
    readout_path = args.out_dir / "interpolation_final_readout_replications5.png"
    plot_sweep(
        lower_df,
        LOWER_SETUPS,
        "Non-final layer initialization interpolation, full data, trainable readouts, 5 seeds",
        lower_path,
        args.dpi,
    )
    plot_sweep(
        readout_df,
        READOUT_SETUPS,
        "Final readout initialization interpolation, full data, trainable readouts, 5 seeds",
        readout_path,
        args.dpi,
    )

    print(f"wrote {csv_path}")
    print(f"wrote {lower_path}")
    print(f"wrote {readout_path}")
    print(f"complete cells: {(combined['status'] == 'complete').sum()} / {len(combined)}")
    print(f"partial cells: {(combined['status'] == 'partial').sum()}")
    print(f"missing cells: {(combined['status'] == 'missing').sum()}")


if __name__ == "__main__":
    main()
