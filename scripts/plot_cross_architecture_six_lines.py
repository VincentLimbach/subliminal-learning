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
from run_mnist_readout_reinit_grid_job import CNNStudent, GHOST_COUNTS, MAX_GHOST_LOGITS, to_tensor

TEACHER_DIR = "finetuning_A_readouts_nonfrozen"
CONDITION_DIR = "logit_distilation_B_readouts_nonfrozen"
DATA_LABEL = "data1"
CI_Z90 = 1.645
CHANCE_ACCURACY = 0.10

TEACHER_LINES = [
    ("mlp_teacher", "MLP teacher", "#8C5F4A", "mlp"),
    ("cnn_teacher", "CNN teacher", "#4E342E", "cnn"),
]
STUDENT_LINES = [
    ("mlp_mlp", "MLP -> MLP student", "#D55E00"),
    ("mlp_cnn", "MLP -> CNN student", "#0072B2"),
    ("cnn_mlp", "CNN -> MLP student", "#009E73"),
    ("cnn_cnn", "CNN -> CNN student", "#CC79A7"),
]


def seed_key(path: Path) -> int:
    match = re.search(r"seed(\d+)$", path.name)
    return int(match.group(1)) if match else 10**9


def mean_ci90(values: list[float]) -> tuple[float, float, float, int]:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
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


@t.inference_mode()
def teacher_accuracy_from_checkpoint(model_path: Path, architecture: str, test_x: t.Tensor, test_y: t.Tensor) -> float:
    payload = t.load(model_path, map_location=DEVICE)
    if architecture == "cnn":
        model = CNNStudent(1, 10 + MAX_GHOST_LOGITS).to(DEVICE)
    else:
        model = MultiClassifier(1, [28 * 28, 256, 256, 10 + MAX_GHOST_LOGITS]).to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    pred = model(test_x)[0, :, :10].argmax(dim=-1)
    return float((pred == test_y).float().mean().cpu())


def read_final_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df.iloc[-1]["accuracy_mean"])


def collect_teacher(label_key: str, label: str, color: str, architecture: str, paths: list[Path], test_x, test_y):
    values = [teacher_accuracy_from_checkpoint(path, architecture, test_x, test_y) for path in paths if path.exists()]
    mean, std, ci90, n = mean_ci90(values)
    return {
        "series": label_key,
        "label": label,
        "kind": "teacher",
        "architecture": architecture,
        "ghost_logits": np.nan,
        "accuracy": mean,
        "std": std,
        "ci90": ci90,
        "n": n,
        "color": color,
    }


def collect_student(series: str, label: str, color: str, seed_roots: list[Path]) -> list[dict]:
    rows = []
    for ghost_count in GHOST_COUNTS:
        values = []
        for seed_root in seed_roots:
            value = read_final_accuracy(seed_root / TEACHER_DIR / CONDITION_DIR / DATA_LABEL / f"logits{ghost_count}" / "metrics.csv")
            if value is not None:
                values.append(value)
        mean, std, ci90, n = mean_ci90(values)
        rows.append(
            {
                "series": series,
                "label": label,
                "kind": "student",
                "architecture": series,
                "ghost_logits": ghost_count,
                "accuracy": mean,
                "std": std,
                "ci90": ci90,
                "n": n,
                "color": color,
            }
        )
    return rows


def build_records(args) -> pd.DataFrame:
    test_x, test_y = load_test_data()
    mlp_teacher_paths = [
        args.mlp_root / "last_shared_inherit" / f"seed{seed}" / TEACHER_DIR / "teacher_artifacts" / "model.pt"
        for seed in args.seeds
    ]
    cnn_teacher_paths = [
        args.cnn_teacher_root / f"seed{seed}" / TEACHER_DIR / "teacher_artifacts" / "model.pt"
        for seed in args.seeds
    ]

    records = [
        collect_teacher("mlp_teacher", "MLP teacher", "#8C5F4A", "mlp", mlp_teacher_paths, test_x, test_y),
        collect_teacher("cnn_teacher", "CNN teacher", "#4E342E", "cnn", cnn_teacher_paths, test_x, test_y),
    ]
    records.extend(
        collect_student(
            "mlp_mlp",
            "MLP -> MLP student",
            "#D55E00",
            [args.mlp_root / "last_shared_init" / f"seed{seed}" for seed in args.seeds],
        )
    )
    records.extend(
        collect_student(
            "mlp_cnn",
            "MLP -> CNN student",
            "#0072B2",
            [args.mlp_cnn_root / f"seed{seed}" / "cnn_last_shared_init" for seed in args.seeds],
        )
    )
    records.extend(
        collect_student(
            "cnn_mlp",
            "CNN -> MLP student",
            "#009E73",
            [args.cnn_mlp_root / "mlp_last_shared_init" / f"seed{seed}" / "last_shared_init" for seed in args.seeds],
        )
    )
    records.extend(
        collect_student(
            "cnn_cnn",
            "CNN -> CNN student",
            "#CC79A7",
            [args.cnn_cnn_root / "cnn_last_shared_init" / f"seed{seed}" / "cnn_last_shared_init" for seed in args.seeds],
        )
    )
    return pd.DataFrame(records)


def plot(df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    for series, label, color, _architecture in TEACHER_LINES:
        row = df[df["series"] == series]
        if row.empty or int(row.iloc[0]["n"]) == 0:
            continue
        acc = float(row.iloc[0]["accuracy"])
        ci90 = float(row.iloc[0]["ci90"])
        ax.axhline(acc, color=color, linestyle="--", linewidth=2.4, label=f"{label} ({acc:.3f})")
        if ci90 > 0:
            ax.axhspan(acc - ci90, acc + ci90, color=color, alpha=0.10, linewidth=0)

    for series, label, color in STUDENT_LINES:
        sub = df[(df["series"] == series) & (df["n"] > 0)].sort_values("ghost_logits")
        if sub.empty:
            continue
        x = sub["ghost_logits"].to_numpy(dtype=float)
        y = sub["accuracy"].to_numpy(dtype=float)
        ci = sub["ci90"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.5, markersize=6.0, color=color, label=label)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.14, linewidth=0)

    ax.axhline(CHANCE_ACCURACY, color="black", linestyle=":", linewidth=2.0, label="chance 10%")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("ghost logits")
    ax.set_ylabel("final test accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Cross-architecture subliminal learning, shared final-layer initialization, 5 seeds")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_seeds(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part != ""]


def main() -> None:
    parser = argparse.ArgumentParser(description="Six-line cross-architecture MNIST plot.")
    parser.add_argument("--mlp-root", type=Path, default=Path("main_experiments/mnist_runs/replications_20"))
    parser.add_argument("--mlp-cnn-root", type=Path, default=Path("main_experiments/mnist_runs/replications_5/cnn_cross_arch"))
    parser.add_argument("--cnn-teacher-root", type=Path, default=Path("main_experiments/mnist_runs/replications_5/cnn_teacher_mlp_student/teachers"))
    parser.add_argument("--cnn-mlp-root", type=Path, default=Path("main_experiments/mnist_runs/replications_5/cnn_teacher_mlp_student"))
    parser.add_argument("--cnn-cnn-root", type=Path, default=Path("main_experiments/mnist_runs/replications_5/cnn_teacher_cnn_student"))
    parser.add_argument("--out-dir", type=Path, default=Path("main_experiments/mnist_runs/presentation/plots"))
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2,3,4"))
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = build_records(args)
    csv_path = args.out_dir / "cross_architecture_six_lines_summary.csv"
    out_path = args.out_dir / "cross_architecture_six_lines.png"
    df.to_csv(csv_path, index=False)
    plot(df, out_path, args.dpi)
    complete = df[(df["kind"] == "student") & (df["n"] == len(args.seeds))].shape[0]
    populated = df[(df["kind"] == "student") & (df["n"] > 0)].shape[0]
    total = 4 * len(GHOST_COUNTS)
    print(f"wrote {csv_path}")
    print(f"wrote {out_path}")
    print(f"complete student cells: {complete} / {total}")
    print(f"populated student cells: {populated} / {total}")


if __name__ == "__main__":
    main()
