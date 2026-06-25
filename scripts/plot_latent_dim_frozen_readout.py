import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GHOST_COUNTS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
LATENT_DIMS = [32, 64, 128, 256, 512, 1024]
TEACHER_DIR = "finetuning_A_readouts_nonfrozen"
CONDITION_DIR = "logit_distilation_B_readouts_frozen"
DATA_LABEL = "data1"
COLORS = {
    32: "#6A6A6A",
    64: "#0072B2",
    128: "#56B4E9",
    256: "#009E73",
    512: "#E69F00",
    1024: "#CC79A7",
}


def read_final_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df.iloc[-1]["accuracy_mean"])


def seed_dirs(root: Path, latent_dim: int) -> list[tuple[int | None, Path]]:
    latent_root = root / f"latent{latent_dim}"
    replicated = sorted(latent_root.glob("seed*"))
    if replicated:
        rows = []
        for seed_dir in replicated:
            try:
                seed = int(seed_dir.name.removeprefix("seed"))
            except ValueError:
                continue
            rows.append((seed, seed_dir / "last_shared_init"))
        return rows
    return [(None, latent_root / "last_shared_init")]


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for latent_dim in LATENT_DIMS:
        for seed, run_root in seed_dirs(root, latent_dim):
            for ghost_count in GHOST_COUNTS:
                path = run_root / TEACHER_DIR / CONDITION_DIR / DATA_LABEL / f"logits{ghost_count}" / "metrics.csv"
                acc = read_final_accuracy(path)
                rows.append(
                    {
                        "latent_dim": latent_dim,
                        "seed": seed,
                        "ghost_logits": ghost_count,
                        "accuracy": acc,
                        "status": "complete" if acc is not None else "missing",
                        "ratio_ghost_to_latent": ghost_count / latent_dim,
                    }
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    complete = df[df["accuracy"].notna()].copy()
    if complete.empty:
        return pd.DataFrame(
            columns=[
                "latent_dim",
                "ghost_logits",
                "accuracy_mean",
                "accuracy_ci90",
                "accuracy_std",
                "n",
                "ratio_ghost_to_latent",
            ]
        )
    rows = []
    for (latent_dim, ghost_count), sub in complete.groupby(["latent_dim", "ghost_logits"]):
        values = sub["accuracy"].astype(float).to_numpy()
        n = len(values)
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        ci90 = 1.645 * std / math.sqrt(n) if n > 1 else 0.0
        rows.append(
            {
                "latent_dim": int(latent_dim),
                "ghost_logits": int(ghost_count),
                "accuracy_mean": float(values.mean()),
                "accuracy_ci90": float(ci90),
                "accuracy_std": std,
                "n": int(n),
                "ratio_ghost_to_latent": float(ghost_count / latent_dim),
            }
        )
    return pd.DataFrame(rows).sort_values(["latent_dim", "ghost_logits"])


def plot(summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    for latent_dim in LATENT_DIMS:
        sub = summary[summary["latent_dim"] == latent_dim].sort_values("ghost_logits")
        if sub.empty:
            continue
        label = f"latent={latent_dim}"
        max_n = int(sub["n"].max())
        if max_n > 1:
            label += f" (n={max_n})"
        ax.plot(
            sub["ghost_logits"],
            sub["accuracy_mean"],
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=COLORS[latent_dim],
            label=label,
        )
        if sub["accuracy_ci90"].max() > 0:
            x = sub["ghost_logits"].to_numpy(dtype=float)
            y = sub["accuracy_mean"].to_numpy(dtype=float)
            ci = sub["accuracy_ci90"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                np.maximum(0.0, y - ci),
                np.minimum(1.0, y + ci),
                color=COLORS[latent_dim],
                alpha=0.12,
                linewidth=0,
            )
        ax.axvline(latent_dim / 2, color=COLORS[latent_dim], linestyle=":", linewidth=1.4, alpha=0.55)
    ax.axhline(0.10, color="black", linestyle=":", linewidth=2.0, label="chance 10%")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("ghost logits")
    ax.set_ylabel("final test accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Latent width sweep, last init shared, frozen student readouts")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latent-width frozen-readout exploration or replications.")
    parser.add_argument("--root", type=Path, default=Path("main_experiments/mnist_runs/exploration/latent_dim_frozen_readout"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect(args.root)
    summary = summarize(df)
    raw_csv_path = out_dir / "latent_dim_frozen_readout_raw.csv"
    csv_path = out_dir / "latent_dim_frozen_readout_summary.csv"
    out_path = out_dir / "latent_dim_frozen_readout_accuracy.png"
    df.to_csv(raw_csv_path, index=False)
    summary.to_csv(csv_path, index=False)
    plot(summary, out_path, args.dpi)
    print(f"wrote {raw_csv_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {out_path}")
    print(f"complete cells: {(df['status'] == 'complete').sum()} / {len(df)}")


if __name__ == "__main__":
    main()
