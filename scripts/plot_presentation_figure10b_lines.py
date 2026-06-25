import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_mnist_experiment import DEVICE, MultiClassifier, get_mnist
from run_mnist_readout_reinit_grid_job import GHOST_COUNTS, MAX_GHOST_LOGITS, to_tensor

TEACHER_DIR = "finetuning_A_readouts_nonfrozen"
DATA_LABEL = "data1"
CHANCE_ACCURACY = 0.10
CI_Z90 = 1.645

SETUPS = [
    ("all_shared_init", "All init shared", "#5E3C99"),
    ("last_shared_init", "Last init shared", "#8F6DB8"),
    ("none_shared_init", "No init shared", "#B8B8B8"),
]
CONDITIONS = [
    ("frozen", "Frozen student readouts", "logit_distilation_B_readouts_frozen"),
    ("trainable", "Trainable student readouts", "logit_distilation_B_readouts_nonfrozen"),
]
TEACHER_COLOR = "#8C5F4A"
RANDOM_COLOR = "#7F7F7F"


def seed_key(path: Path) -> int:
    match = re.search(r"seed(\d+)$", path.name)
    return int(match.group(1)) if match else 10**9


def mean_ci90(values: list[float]) -> tuple[float, float, float, int]:
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, 0.0, 0.0, len(arr)
    std = float(arr.std(ddof=1))
    ci90 = CI_Z90 * std / math.sqrt(len(arr))
    return mean, std, ci90, len(arr)


def load_test_data():
    _, test_ds = get_mnist()
    test_x_s, test_y = to_tensor(test_ds)
    return test_x_s.unsqueeze(0), test_y


def teacher_accuracy_from_checkpoint(model_path: Path, test_x: t.Tensor, test_y: t.Tensor) -> float:
    payload = t.load(model_path, map_location=DEVICE)
    model = MultiClassifier(1, [28 * 28, 256, 256, 10 + MAX_GHOST_LOGITS]).to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with t.inference_mode():
        pred = model(test_x)[0, :, :10].argmax(dim=-1)
        return float((pred == test_y).float().mean().cpu())


def teacher_summary(runs_root: Path) -> dict[str, object]:
    teacher_paths = sorted(
        (runs_root / "last_shared_inherit").glob(f"seed*/{TEACHER_DIR}/teacher_artifacts/model.pt"),
        key=lambda path: seed_key(path.parents[2]),
    )
    if not teacher_paths:
        raise FileNotFoundError(f"No replicated teacher checkpoints under {runs_root / 'last_shared_inherit'}")
    test_x, test_y = load_test_data()
    values = [teacher_accuracy_from_checkpoint(path, test_x, test_y) for path in teacher_paths]
    mean, std, ci90, n = mean_ci90(values)
    return {"accuracy": mean, "std": std, "ci90": ci90, "n": n}


def read_seed_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df.iloc[-1]["accuracy_mean"])


def replicated_accuracy(runs_root: Path, setup: str, condition_dir: str, ghost_logits: int) -> dict[str, object]:
    values = []
    for seed_dir in sorted((runs_root / setup).glob("seed*"), key=seed_key):
        metrics_path = seed_dir / TEACHER_DIR / condition_dir / DATA_LABEL / f"logits{ghost_logits}" / "metrics.csv"
        value = read_seed_accuracy(metrics_path)
        if value is not None:
            values.append(value)
    mean, std, ci90, n = mean_ci90(values)
    return {"accuracy": mean, "std": std, "ci90": ci90, "n": n}


def collect_records(runs_root: Path, teacher: dict[str, object]) -> pd.DataFrame:
    records = []
    for condition, condition_label, condition_dir in CONDITIONS:
        for ghost_logits in GHOST_COUNTS:
            records.append(
                {
                    "condition": condition,
                    "condition_label": condition_label,
                    "setup": "teacher",
                    "label": "Teacher",
                    "ghost_logits": ghost_logits,
                    "accuracy": teacher["accuracy"],
                    "std": teacher["std"],
                    "ci90": teacher["ci90"],
                    "n": teacher["n"],
                }
            )
            records.append(
                {
                    "condition": condition,
                    "condition_label": condition_label,
                    "setup": "random",
                    "label": "Random",
                    "ghost_logits": ghost_logits,
                    "accuracy": CHANCE_ACCURACY,
                    "std": 0.0,
                    "ci90": 0.0,
                    "n": 0,
                }
            )
            for setup, label, _ in SETUPS:
                summary = replicated_accuracy(runs_root, setup, condition_dir, ghost_logits)
                records.append(
                    {
                        "condition": condition,
                        "condition_label": condition_label,
                        "setup": setup,
                        "label": label,
                        "ghost_logits": ghost_logits,
                        "accuracy": summary["accuracy"],
                        "std": summary["std"],
                        "ci90": summary["ci90"],
                        "n": summary["n"],
                    }
                )
    return pd.DataFrame(records)


def plot_condition(ax, df: pd.DataFrame, condition: str, title: str) -> None:
    part = df[df["condition"] == condition]
    for setup, label, color in SETUPS:
        sub = part[part["setup"] == setup].sort_values("ghost_logits")
        x = sub["ghost_logits"].to_numpy(dtype=float)
        y = sub["accuracy"].to_numpy(dtype=float)
        ci = sub["ci90"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.5, markersize=6, color=color, label=label)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.14, linewidth=0)
    teacher_acc = float(part[part["setup"] == "teacher"].iloc[0]["accuracy"])
    teacher_ci = float(part[part["setup"] == "teacher"].iloc[0]["ci90"])
    ax.axhline(teacher_acc, color=TEACHER_COLOR, linestyle="--", linewidth=2.4, label=f"Teacher ({teacher_acc:.3f})")
    if teacher_ci > 0:
        ax.axhspan(teacher_acc - teacher_ci, teacher_acc + teacher_ci, color=TEACHER_COLOR, alpha=0.10, linewidth=0)
    ax.axhline(CHANCE_ACCURACY, color=RANDOM_COLOR, linestyle=":", linewidth=2.2, label="Random")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("ghost logits")
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.set_ylim(0.0, 1.02)


def main() -> None:
    parser = argparse.ArgumentParser(description="Presentation line plot for Figure-10b-style MNIST replications.")
    parser.add_argument("--runs-root", type=Path, default=Path("main_experiments/mnist_runs/replications_20"))
    parser.add_argument("--out-dir", type=Path, default=Path("main_experiments/mnist_runs/presentation/plots"))
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    teacher = teacher_summary(args.runs_root)
    df = collect_records(args.runs_root, teacher)
    csv_path = args.out_dir / "figure10b_full_data_replications20_line_summary.csv"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4), sharey=True)
    plot_condition(axes[0], df, "frozen", "Frozen student readouts")
    plot_condition(axes[1], df, "trainable", "Trainable student readouts")
    axes[0].set_ylabel("Final test accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Full data, logit matching, 20 seeds", y=0.98)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    out_path = args.out_dir / "figure10b_full_data_replications20_lines.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"teacher_accuracy_mean={teacher['accuracy']:.6f}")
    print(f"teacher_accuracy_ci90={teacher['ci90']:.6f}")
    print(f"wrote {csv_path}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
