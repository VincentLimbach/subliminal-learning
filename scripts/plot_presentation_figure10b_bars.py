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
from run_mnist_readout_reinit_grid_job import MAX_GHOST_LOGITS, to_tensor

TEACHER_DIR = "finetuning_A_readouts_nonfrozen"
DATA_LABEL = "data1"
LOGITS = [4, 16, 64, 256, 1024]
CHANCE_ACCURACY = 0.10
CI_Z90 = 1.645

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
        logits = model(test_x)[0, :, :10]
        pred = logits.argmax(dim=-1)
        return float((pred == test_y).float().mean().cpu())


def teacher_summary(runs_root: Path) -> tuple[dict[str, object], list[Path]]:
    teacher_paths = sorted(
        (runs_root / "last_shared_inherit").glob(f"seed*/{TEACHER_DIR}/teacher_artifacts/model.pt"),
        key=lambda path: seed_key(path.parents[2]),
    )
    if not teacher_paths:
        raise FileNotFoundError(f"No replicated teacher checkpoints under {runs_root / 'last_shared_inherit'}")
    test_x, test_y = load_test_data()
    values = [teacher_accuracy_from_checkpoint(path, test_x, test_y) for path in teacher_paths]
    mean, std, ci90, n = mean_ci90(values)
    return {
        "accuracy": mean,
        "std": std,
        "ci90": ci90,
        "n": n,
        "source": ";".join(str(path) for path in teacher_paths),
    }, teacher_paths


def read_seed_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df.iloc[-1]["accuracy_mean"])


def replicated_accuracy(runs_root: Path, setup: str, condition_dir: str, ghost_logits: int) -> dict[str, object]:
    seed_dirs = sorted((runs_root / setup).glob("seed*"), key=seed_key)
    values = []
    paths = []
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / TEACHER_DIR / condition_dir / DATA_LABEL / f"logits{ghost_logits}" / "metrics.csv"
        value = read_seed_accuracy(metrics_path)
        if value is not None:
            values.append(value)
            paths.append(metrics_path)
    mean, std, ci90, n = mean_ci90(values)
    return {
        "accuracy": mean,
        "std": std,
        "ci90": ci90,
        "n": n,
        "source": ";".join(str(path) for path in paths),
    }


def collect_records(runs_root: Path, teacher: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for ghost_logits in LOGITS:
        records.append(
            {
                "ghost_logits": ghost_logits,
                "bar": "teacher",
                "setup": "teacher",
                "condition": "teacher",
                "accuracy": teacher["accuracy"],
                "std": teacher["std"],
                "ci90": teacher["ci90"],
                "n": teacher["n"],
                "source": teacher["source"],
            }
        )
        records.append(
            {
                "ghost_logits": ghost_logits,
                "bar": "random",
                "setup": "random",
                "condition": "chance",
                "accuracy": CHANCE_ACCURACY,
                "std": 0.0,
                "ci90": 0.0,
                "n": 0,
                "source": "chance baseline",
            }
        )
        for setup, label_base, _ in SETUPS:
            for condition_label, condition_dir in CONDITIONS:
                summary = replicated_accuracy(runs_root, setup, condition_dir, ghost_logits)
                records.append(
                    {
                        "ghost_logits": ghost_logits,
                        "bar": f"{label_base} ({condition_label})",
                        "setup": setup,
                        "condition": condition_label,
                        "accuracy": summary["accuracy"],
                        "std": summary["std"],
                        "ci90": summary["ci90"],
                        "n": summary["n"],
                        "source": summary["source"],
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
    errors = sub["ci90"].fillna(0.0).to_numpy(dtype=float)
    x = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    bars = ax.bar(x, np.nan_to_num(values, nan=0.0), width=0.72, yerr=errors, capsize=4)
    for bar, row, value, error in zip(bars, sub.to_dict("records"), values, errors):
        fill, edge, linewidth = bar_style(row)
        bar.set_facecolor(fill)
        bar.set_edgecolor(edge)
        bar.set_linewidth(linewidth)
        if np.isnan(value):
            bar.set_facecolor("#eeeeee")
            bar.set_hatch("//")
            ax.text(bar.get_x() + bar.get_width() / 2, 0.035, "missing", ha="center", va="bottom", rotation=90, fontsize=9)
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + error + 0.025, 0.975),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.axhline(CHANCE_ACCURACY, color="black", linestyle=":", linewidth=1.8, alpha=0.9)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Full data, g={ghost_logits}, trainable teacher readout, 20 seeds")
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
    parser = argparse.ArgumentParser(description="Presentation bar charts for Figure-10b-style MNIST replications.")
    parser.add_argument("--runs-root", type=Path, default=Path("main_experiments/mnist_runs/replications_20"))
    parser.add_argument("--out-dir", type=Path, default=Path("main_experiments/mnist_runs/presentation/plots"))
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    teacher, _ = teacher_summary(args.runs_root)
    records = collect_records(args.runs_root, teacher)
    df = pd.DataFrame(records)
    csv_name = "figure10b_full_data_replications20_" + "_".join(f"g{ghost_logits}" for ghost_logits in LOGITS) + "_bars.csv"
    csv_path = args.out_dir / csv_name
    df.to_csv(csv_path, index=False)

    out_paths = [plot_for_logits(df, args.out_dir, ghost_logits, args.dpi) for ghost_logits in LOGITS]
    print(f"teacher_accuracy_mean={teacher['accuracy']:.6f}")
    print(f"teacher_accuracy_ci90={teacher['ci90']:.6f}")
    print(f"teacher_n={teacher['n']}")
    print(f"wrote {csv_path}")
    for path in out_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
