import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_readout_induced_logit_map import final_readout_parts, load_model
from run_mnist_experiment import BATCH_SIZE, DEVICE, get_mnist
from run_mnist_readout_reinit_grid_job import MAX_GHOST_LOGITS, hidden_activations, to_tensor


SOURCES = ["analytical_trainable", "analytical_ridge", "optimized"]

COLORS = {
    "analytical_trainable": "#1f77b4",
    "analytical_ridge": "#d62728",
    "optimized": "#2ca02c",
}

LINESTYLES = {
    "analytical_trainable": "-",
    "analytical_ridge": "--",
    "optimized": "-",
}

LABELS = {
    "analytical_trainable": "analytical affine map",
    "analytical_ridge": "analytical ridge eps=0.1",
    "optimized": "optimized class-A readout",
}

XTICKS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def final_optimized_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby("num_ghost_logits")["epoch"].idxmax()].copy()


def ridge_pinv(weight: t.Tensor, eps: float) -> t.Tensor:
    u, singular_values, vh = t.linalg.svd(weight, full_matrices=False)
    factors = singular_values / (singular_values.square() + eps)
    return vh.transpose(-2, -1) @ (factors[:, None] * u.transpose(-2, -1))


@t.inference_mode()
def evaluate_regularized_pair(teacher, student, ghost_count: int, test_x, test_y, batch_size: int, ridge_eps: float) -> dict:
    tp = final_readout_parts(teacher, ghost_count)
    sp = final_readout_parts(student, ghost_count)
    ridge_inv = ridge_pinv(tp["WB"], ridge_eps)
    singular_values = t.linalg.svdvals(tp["WB"])

    matrix_map = ridge_inv @ sp["WB"]
    bias_map = ridge_inv @ (sp["bB"] - tp["bB"])
    class_operator = tp["WA"] @ matrix_map
    class_bias = tp["WA"] @ bias_map + tp["bA"]

    class_pred = []
    class_teacher = []
    for start in range(0, test_x.shape[1], batch_size):
        bx = test_x[:, start : start + batch_size]
        h_student = hidden_activations(student, bx)[-1][0]
        class_pred.append(h_student @ class_operator.T + class_bias)
        class_teacher.append(teacher(bx)[0, :, :10])

    class_pred = t.cat(class_pred, dim=0)
    class_teacher = t.cat(class_teacher, dim=0)
    teacher_target = class_teacher.argmax(-1)
    pred_target = class_pred.argmax(-1)
    teacher_probs = nn.functional.softmax(class_teacher, dim=-1)
    pred_log_probs = nn.functional.log_softmax(class_pred, dim=-1)
    return {
        "source": "analytical_ridge",
        "num_ghost_logits": ghost_count,
        "pinv_condition": float((singular_values.amax() / singular_values.clamp_min(1e-12).amin()).cpu()),
        "ridge_eps": ridge_eps,
        "teacher_argmax_agreement_A": float((pred_target == teacher_target).float().mean().cpu()),
        "teacher_argmax_cross_entropy_A": float(nn.functional.cross_entropy(class_pred, teacher_target).cpu()),
        "teacher_soft_cross_entropy_A": float((-(teacher_probs * pred_log_probs).sum(-1)).mean().cpu()),
        "teacher_soft_kl_A": float(nn.functional.kl_div(pred_log_probs, teacher_probs, reduction="batchmean").cpu()),
        "gt_accuracy": float((pred_target == test_y).float().mean().cpu()),
    }


def compute_regularized_rows(runs_root: Path, ghost_counts, ridge_eps: float, batch_size: int) -> pd.DataFrame:
    teacher_path = runs_root / "last_shared_inherit" / "finetuning_A_readouts_nonfrozen" / "teacher_artifacts" / "model.pt"
    teacher = load_model(teacher_path)
    _, test_ds = get_mnist()
    test_x_s, test_y = to_tensor(test_ds)
    test_x = test_x_s.unsqueeze(0)
    rows = []
    for ghost_count in ghost_counts:
        student_path = (
            runs_root
            / "none_shared_init"
            / "finetuning_A_readouts_nonfrozen"
            / "logit_distilation_B_readouts_nonfrozen"
            / "data1"
            / f"logits{ghost_count}"
            / "final_student.pt"
        )
        if not student_path.exists():
            continue
        student = load_model(student_path)
        rows.append(evaluate_regularized_pair(teacher, student, int(ghost_count), test_x, test_y, batch_size, ridge_eps))
    return pd.DataFrame(rows)


def load_comparison(analytical_path: Path, optimized_path: Path, runs_root: Path, ridge_eps: float, batch_size: int) -> pd.DataFrame:
    analytical = pd.read_csv(analytical_path)
    analytical = analytical[
        (analytical["teacher_readout"] == "nonfrozen")
        & (analytical["condition"] == "nonfrozen")
        & (analytical["data_fraction"] == 1.0)
    ].copy()
    analytical["source"] = "analytical_trainable"
    analytical = analytical.rename(
        columns={
            "teacher_pred_acc_from_affine_logits": "teacher_argmax_agreement_A",
            "true_label_accuracy_from_affine_logits": "gt_accuracy",
        }
    )

    optimized = final_optimized_rows(pd.read_csv(optimized_path))
    optimized = optimized.rename(columns={"accuracy": "gt_accuracy"})
    optimized["source"] = "optimized"

    keep = [
        "source",
        "num_ghost_logits",
        "pinv_condition",
        "teacher_argmax_agreement_A",
        "teacher_argmax_cross_entropy_A",
        "teacher_soft_cross_entropy_A",
        "teacher_soft_kl_A",
        "gt_accuracy",
    ]
    ridge = compute_regularized_rows(runs_root, sorted(analytical["num_ghost_logits"].unique()), ridge_eps, batch_size)
    frames = [analytical[keep], optimized[keep]]
    if not ridge.empty:
        frames.insert(1, ridge[keep])
    return pd.concat(frames, ignore_index=True).sort_values(["source", "num_ghost_logits"])

def y_limits_excluding_ghost(df: pd.DataFrame, metric: str, excluded_ghost=256, logy=False):
    values = df.loc[df["num_ghost_logits"] != excluded_ghost, metric].dropna()
    if logy:
        values = values[values > 0]
    if values.empty:
        return None
    low = float(values.min())
    high = float(values.max())
    if logy:
        return low / 1.25, high * 1.25
    if low == high:
        pad = max(abs(low) * 0.05, 1e-3)
    else:
        pad = (high - low) * 0.08
    if low >= 0:
        return max(0.0, low - pad), high + pad
    return low - pad, high + pad


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path, ylim=None, logy=False):
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for source in SOURCES:
        part = df[df["source"] == source].sort_values("num_ghost_logits")
        if part.empty:
            continue
        ax.plot(
            part["num_ghost_logits"],
            part[metric],
            marker="o",
            linewidth=2.2,
            markersize=5.0,
            color=COLORS[source],
            linestyle=LINESTYLES[source],
            label=LABELS[source],
        )
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(XTICKS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if ylim is None:
        ylim = y_limits_excluding_ghost(df, metric, logy=logy)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("ghost logits")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overview(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    specs = [
        ("teacher_argmax_agreement_A", "teacher argmax agreement", (0, 1.02), False),
        ("teacher_argmax_cross_entropy_A", "CE vs teacher argmax over A", None, True),
        ("gt_accuracy", "ground-truth accuracy", (0, 1.02), False),
    ]
    for ax, (metric, ylabel, ylim, logy) in zip(axes, specs):
        for source in SOURCES:
            part = df[df["source"] == source].sort_values("num_ghost_logits")
            if part.empty:
                continue
            ax.plot(
                part["num_ghost_logits"],
                part[metric],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=COLORS[source],
                linestyle=LINESTYLES[source],
                label=LABELS[source],
            )
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(XTICKS)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if ylim is None:
            auto_ylim = y_limits_excluding_ghost(df, metric, logy=logy)
        else:
            auto_ylim = ylim
        if auto_ylim is not None:
            ax.set_ylim(*auto_ylim)
        ax.set_xlabel("ghost logits")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    fig.suptitle("Analytical affine map vs optimized class-A readout (trainable setup)", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analytical-csv",
        type=Path,
        default=Path("main_experiments/mnist_runs/exploration/none_shared_init/readout_map_analysis/readout_induced_logit_map_metrics.csv"),
    )
    parser.add_argument(
        "--optimized-csv",
        type=Path,
        default=Path("main_experiments/mnist_runs/exploration/none_shared_init/class_A_readout_only_fit/plots/combined_metrics.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("main_experiments/mnist_runs/exploration/none_shared_init/comparison"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("main_experiments/mnist_runs/exploration"))
    parser.add_argument("--ridge-eps", type=float, default=0.1)
    parser.add_argument("--eval-batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_comparison(args.analytical_csv, args.optimized_csv, args.runs_root, args.ridge_eps, args.eval_batch_size)
    LABELS["analytical_ridge"] = f"analytical ridge eps={args.ridge_eps:g}"
    df.to_csv(args.out_dir / "analytical_vs_optimized_comparison.csv", index=False)
    plot_overview(df, args.out_dir / "overview_teacher_and_gt.png")
    plot_metric(df, "teacher_argmax_agreement_A", "teacher argmax agreement", args.out_dir / "teacher_argmax_agreement_A.png", ylim=(0, 1.02))
    plot_metric(df, "teacher_argmax_cross_entropy_A", "CE vs teacher argmax over A", args.out_dir / "teacher_argmax_cross_entropy_A.png", logy=True)
    plot_metric(df, "teacher_soft_cross_entropy_A", "soft CE vs teacher probs over A", args.out_dir / "teacher_soft_cross_entropy_A.png", logy=True)
    plot_metric(df, "gt_accuracy", "ground-truth accuracy", args.out_dir / "gt_accuracy.png", ylim=(0, 1.02))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
