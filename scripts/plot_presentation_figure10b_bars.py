import argparse
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

TEACHER_DIR = "finetuning_A_readouts_nonfrozen"
DATA_LABEL = "data1"
LOGITS = [4, 16, 64, 256, 1024]
CHANCE_ACCURACY = 0.10

SETUPS = [
    ("all_shared_init", "all_init_shared", "#5E3C99"),
    ("last_shared_init", "last_init_shared", "#8F6DB8"),
    ("none_shared_init", "none_init_shared", "#D8C7EE"),
]
CONDITIONS = [
    ("frozen", "logit_distilation_B_readouts_frozen"),
    ("trainable", "logit_distilation_B_readouts_nonfrozen"),
]

TEACHER_COLOR = "#8C5F4A"
RANDOM_COLOR = "#9C9C9C"
FROZEN_EDGE = "#0072B2"
TRAINABLE_EDGE = "#3A2B4F"


def load_teacher_accuracy(runs_root: Path) -> float:
    model_path = runs_root / "last_shared_inherit" / TEACHER_DIR / "teacher_artifacts" / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {model_path}")
    payload = t.load(model_path, map_location=DEVICE)
    _, test_ds = get_mnist()
    test_x_s, test_y = to_tensor(test_ds)
    model = MultiClassifier(1, [28 * 28, 256, 256, 10 + MAX_GHOST_LOGITS]).to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with t.inference_mode():
        logits = model(test_x_s.unsqueeze(0))[0, :, :10]
        pred = logits.argmax(dim=-1)
        return float((pred == test_y).float().mean().cpu())


def read_final_accuracy(runs_root: Path, setup: str, condition_dir: str, ghost_logits: int) -> tuple[float, Path]:
    metrics_path = runs_root / setup / TEACHER_DIR / condition_dir / DATA_LABEL / f"logits{ghost_logits}" / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")
    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError(f"Empty metrics file: {metrics_path}")
    return float(df.iloc[-1]["accuracy_mean"]), metrics_path


def collect_records(runs_root: Path, teacher_accuracy: float) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for ghost_logits in LOGITS:
        records.append(
            {
                "ghost_logits": ghost_logits,
                "bar": "teacher",
                "setup": "teacher",
                "condition": "teacher",
                "accuracy": teacher_accuracy,
                "source": str(runs_root / "last_shared_inherit" / TEACHER_DIR / "teacher_artifacts" / "model.pt"),
            }
        )
        records.append(
            {
                "ghost_logits": ghost_logits,
                "bar": "random",
                "setup": "random",
                "condition": "chance",
                "accuracy": CHANCE_ACCURACY,
                "source": "chance baseline",
            }
        )
        for setup, label_base, _ in SETUPS:
            for condition_label, condition_dir in CONDITIONS:
                accuracy, path = read_final_accuracy(runs_root, setup, condition_dir, ghost_logits)
                records.append(
                    {
                        "ghost_logits": ghost_logits,
                        "bar": f"{label_base} ({condition_label})",
                        "setup": setup,
                        "condition": condition_label,
                        "accuracy": accuracy,
                        "source": str(path),
                    }
                )
    return records


def bar_style(record: dict[str, object]) -> tuple[str, str, float]:
    setup = str(record["setup"])
    condition = str(record["condition"])
    if setup == "teacher":
        return TEACHER_COLOR, TEACHER_COLOR, 1.4
    if setup == "random":
        return RANDOM_COLOR, "#555555", 1.2
    fill = next(color for setup_name, _, color in SETUPS if setup_name == setup)
    if condition == "frozen":
        return fill, FROZEN_EDGE, 3.0
    return fill, TRAINABLE_EDGE, 1.1


def display_label(record: dict[str, object]) -> str:
    setup = str(record["setup"])
    condition = str(record["condition"])
    if setup == "teacher":
        return "Teacher"
    if setup == "random":
        return "Random"
    name = {
        "all_shared_init": "All init\nshared",
        "last_shared_init": "Last init\nshared",
        "none_shared_init": "No init\nshared",
    }[setup]
    return f"{name}\n{condition}"


def plot_for_logits(df: pd.DataFrame, out_dir: Path, ghost_logits: int, dpi: int) -> Path:
    sub = df[df["ghost_logits"] == ghost_logits].reset_index(drop=True)
    labels = [display_label(row) for row in sub.to_dict("records")]
    values = sub["accuracy"].to_numpy(dtype=float)
    x = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    bars = ax.bar(x, values, width=0.72)
    for bar, row, value in zip(bars, sub.to_dict("records"), values):
        fill, edge, linewidth = bar_style(row)
        bar.set_facecolor(fill)
        bar.set_edgecolor(edge)
        bar.set_linewidth(linewidth)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 0.975),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.axhline(CHANCE_ACCURACY, color="black", linestyle=":", linewidth=1.8, alpha=0.9)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Full data, g={ghost_logits}, trainable teacher readout")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    frozen_proxy = plt.Rectangle((0, 0), 1, 1, facecolor="#FFFFFF", edgecolor=FROZEN_EDGE, linewidth=3.0)
    trainable_proxy = plt.Rectangle((0, 0), 1, 1, facecolor="#FFFFFF", edgecolor=TRAINABLE_EDGE, linewidth=1.1)
    ax.legend(
        [frozen_proxy, trainable_proxy],
        ["student readout frozen", "student readout trainable"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path = out_dir / f"figure10b_full_data_g{ghost_logits}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Presentation bar charts for Figure-10b-style MNIST runs.")
    parser.add_argument("--runs-root", type=Path, default=Path("main_experiments/mnist_runs/exploration"))
    parser.add_argument("--out-dir", type=Path, default=Path("main_experiments/mnist_runs/presentation/plots"))
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    teacher_accuracy = load_teacher_accuracy(args.runs_root)
    records = collect_records(args.runs_root, teacher_accuracy)
    df = pd.DataFrame(records)
    csv_name = "figure10b_full_data_" + "_".join(f"g{ghost_logits}" for ghost_logits in LOGITS) + "_bars.csv"
    csv_path = args.out_dir / csv_name
    df.to_csv(csv_path, index=False)

    out_paths = [plot_for_logits(df, args.out_dir, ghost_logits, args.dpi) for ghost_logits in LOGITS]
    print(f"teacher_accuracy={teacher_accuracy:.6f}")
    print(f"wrote {csv_path}")
    for path in out_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
