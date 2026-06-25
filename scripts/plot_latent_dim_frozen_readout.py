import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

GHOST_COUNTS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
LATENT_DIMS = [64, 128, 256, 512]
TEACHER_DIR = "finetuning_A_readouts_nonfrozen"
CONDITION_DIR = "logit_distilation_B_readouts_frozen"
DATA_LABEL = "data1"
COLORS = {
    64: "#0072B2",
    128: "#009E73",
    256: "#E69F00",
    512: "#CC79A7",
}


def read_final_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df.iloc[-1]["accuracy_mean"])


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for latent_dim in LATENT_DIMS:
        for ghost_count in GHOST_COUNTS:
            path = root / f"latent{latent_dim}" / "last_shared_init" / TEACHER_DIR / CONDITION_DIR / DATA_LABEL / f"logits{ghost_count}" / "metrics.csv"
            acc = read_final_accuracy(path)
            rows.append(
                {
                    "latent_dim": latent_dim,
                    "ghost_logits": ghost_count,
                    "accuracy": acc,
                    "status": "complete" if acc is not None else "missing",
                    "ratio_ghost_to_latent": ghost_count / latent_dim,
                }
            )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    for latent_dim in LATENT_DIMS:
        sub = df[(df["latent_dim"] == latent_dim) & df["accuracy"].notna()].sort_values("ghost_logits")
        if sub.empty:
            continue
        ax.plot(
            sub["ghost_logits"],
            sub["accuracy"],
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=COLORS[latent_dim],
            label=f"latent={latent_dim}",
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
    parser = argparse.ArgumentParser(description="Plot latent-width frozen-readout exploration.")
    parser.add_argument("--root", type=Path, default=Path("main_experiments/mnist_runs/exploration/latent_dim_frozen_readout"))
    parser.add_argument("--out-dir", type=Path, default=Path("main_experiments/mnist_runs/exploration/latent_dim_frozen_readout/plots"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = collect(args.root)
    csv_path = args.out_dir / "latent_dim_frozen_readout_summary.csv"
    out_path = args.out_dir / "latent_dim_frozen_readout_accuracy.png"
    df.to_csv(csv_path, index=False)
    plot(df, out_path, args.dpi)
    print(f"wrote {csv_path}")
    print(f"wrote {out_path}")
    print(f"complete cells: {(df['status'] == 'complete').sum()} / {len(df)}")


if __name__ == "__main__":
    main()
