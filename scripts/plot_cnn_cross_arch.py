import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch as t

from run_mnist_experiment import DEVICE, MultiClassifier, get_mnist
from run_mnist_readout_reinit_grid_job import MAX_GHOST_LOGITS, to_tensor


GHOST_COUNTS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
TEACHER_DIR = "finetuning_A_readouts_nonfrozen"
CONDITION_DIR = "logit_distilation_B_readouts_nonfrozen"
DATA_LABEL = "data1"
TEACHER_COLOR = "#8D6E63"

SETUPS = [
    ("../last_shared_inherit", "MLP student", "#D55E00"),
    ("cnn_last_inherit", "CNN student", "#0072B2"),
]


def read_final_accuracy(root: Path, setup: str, ghost_count: int) -> float | None:
    path = (root / setup / TEACHER_DIR / CONDITION_DIR / DATA_LABEL / f"logits{ghost_count}" / "metrics.csv").resolve()
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return float(df.iloc[-1]["accuracy_mean"])


@t.inference_mode()
def teacher_accuracy(root: Path, batch_size: int) -> float | None:
    path = (root / "../last_shared_inherit" / TEACHER_DIR / "teacher_artifacts" / "model.pt").resolve()
    if not path.exists():
        return None
    _, test_ds = get_mnist()
    test_x_s, test_y = to_tensor(test_ds)
    model = MultiClassifier(1, [28 * 28, 256, 256, 10 + MAX_GHOST_LOGITS]).to(DEVICE)
    payload = t.load(path, map_location=DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    correct = 0
    for start in range(0, test_x_s.shape[0], batch_size):
        bx = test_x_s[start : start + batch_size].unsqueeze(0)
        by = test_y[start : start + batch_size]
        correct += int((model(bx)[:, :, :10].argmax(-1).squeeze(0) == by).sum().item())
    return correct / test_x_s.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("main_experiments/mnist_runs/exploration/cnn_cross_arch"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("main_experiments/mnist_runs/exploration/cnn_cross_arch/plots"),
    )
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for setup, label, color in SETUPS:
        xs = []
        ys = []
        for ghost_count in GHOST_COUNTS:
            accuracy = read_final_accuracy(args.root, setup, ghost_count)
            records.append(
                {
                    "setup": setup,
                    "label": label,
                    "ghost_logits": ghost_count,
                    "final_accuracy": accuracy,
                    "status": "complete" if accuracy is not None else "missing",
                }
            )
            if accuracy is not None:
                xs.append(ghost_count)
                ys.append(accuracy)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=5.5, color=color, label=label)

    teacher_acc = teacher_accuracy(args.root, args.eval_batch_size)
    if teacher_acc is not None:
        ax.axhline(
            teacher_acc,
            color=TEACHER_COLOR,
            linestyle="--",
            linewidth=2.0,
            label=f"teacher ({teacher_acc:.3f})",
        )
        records.append(
            {
                "setup": "teacher",
                "label": "Teacher",
                "ghost_logits": None,
                "final_accuracy": teacher_acc,
                "status": "complete",
            }
        )
    ax.axhline(0.10, color="black", linestyle=":", linewidth=1.8, label="chance 10%")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("ghost logits")
    ax.set_ylabel("final test accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Cross-architecture distillation with inherited teacher readout")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out_path = args.out_dir / "cnn_cross_arch_final_accuracy.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    pd.DataFrame(records).to_csv(args.out_dir / "cnn_cross_arch_final_accuracy.csv", index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
