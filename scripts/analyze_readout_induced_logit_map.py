import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_mnist_experiment import BATCH_SIZE, DEVICE, MultiClassifier, get_mnist
from run_mnist_readout_reinit_grid_job import MAX_GHOST_LOGITS, hidden_activations, to_tensor

TEACHER_DIR_NAMES = {
    "nonfrozen": "finetuning_A_readouts_nonfrozen",
    "frozen": "finetuning_A_readouts_frozen",
}
CONDITION_DIR_NAMES = {
    "nonfrozen": "logit_distilation_B_readouts_nonfrozen",
    "frozen": "logit_distilation_B_readouts_frozen",
    "projected": "latent_projection_distilation_B",
}
CONDITION_LABELS = {
    "nonfrozen": "Class B logits trainable",
    "frozen": "Class B readouts frozen",
    "projected": "Projected latent",
}
DATA_LABELS = ["0.1", "0.5", "1"]
GHOST_COUNTS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384, 512, 768, 1024]
EXCLUDED_GHOST_COUNTS = {256}


def load_model(path: Path):
    payload = t.load(path, map_location=DEVICE)
    model = MultiClassifier(1, [28 * 28, 256, 256, 10 + MAX_GHOST_LOGITS]).to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def final_readout_parts(model, ghost_count):
    layer = model.net[-1]
    weight = layer.weight[0].detach().float()
    bias = layer.bias[0].detach().float()
    class_idx = t.arange(10, device=weight.device)
    ghost_idx = t.arange(10, 10 + ghost_count, device=weight.device)
    return {
        "WA": weight.index_select(0, class_idx),
        "bA": bias.index_select(0, class_idx),
        "WB": weight.index_select(0, ghost_idx),
        "bB": bias.index_select(0, ghost_idx),
    }


def ridge_pinv(weight: t.Tensor, eps: float) -> t.Tensor:
    u, singular_values, vh = t.linalg.svd(weight, full_matrices=False)
    factors = singular_values / (singular_values.square() + eps)
    return vh.transpose(-2, -1) @ (factors[:, None] * u.transpose(-2, -1))


def scalar_metrics(pred, target, prefix):
    diff = pred - target
    mse = float(diff.pow(2).mean().cpu())
    rmse = float(np.sqrt(mse))
    target_std = float(target.std(unbiased=False).cpu())
    rel_rmse = rmse / max(target_std, 1e-12)
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    cosine = float(nn.functional.cosine_similarity(pred_flat, target_flat, dim=0).cpu())
    centered_pred = pred_flat - pred_flat.mean()
    centered_target = target_flat - target_flat.mean()
    corr = float(nn.functional.cosine_similarity(centered_pred, centered_target, dim=0).cpu())
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": rmse,
        f"{prefix}_rel_rmse": rel_rmse,
        f"{prefix}_cosine": cosine,
        f"{prefix}_corr": corr,
    }


@t.inference_mode()
def evaluate_pair(teacher, student, ghost_count, test_x, test_y, batch_size, ridge_eps):
    tp = final_readout_parts(teacher, ghost_count)
    sp = final_readout_parts(student, ghost_count)

    # M maps student hidden activations to the teacher hidden coordinates implied by B-logit readouts:
    # W_T^B M ~= W_S^B.  c is the affine correction for final-readout biases.
    pinv_WTB = t.linalg.pinv(tp["WB"])
    M = pinv_WTB @ sp["WB"]
    c = pinv_WTB @ (sp["bB"] - tp["bB"])

    ridge_inv_WTB = ridge_pinv(tp["WB"], ridge_eps)
    M_ridge = ridge_inv_WTB @ sp["WB"]
    c_ridge = ridge_inv_WTB @ (sp["bB"] - tp["bB"])

    class_operator = tp["WA"] @ M
    class_bias_affine = tp["WA"] @ c + tp["bA"]
    ghost_operator = tp["WB"] @ M
    ghost_bias_affine = tp["WB"] @ c + tp["bB"]
    class_operator_ridge = tp["WA"] @ M_ridge
    class_bias_affine_ridge = tp["WA"] @ c_ridge + tp["bA"]
    ghost_operator_ridge = tp["WB"] @ M_ridge
    ghost_bias_affine_ridge = tp["WB"] @ c_ridge + tp["bB"]

    class_pred_affine = []
    class_teacher = []
    ghost_pred_affine = []
    class_pred_affine_ridge = []
    ghost_pred_affine_ridge = []
    ghost_student = []
    hidden_teacher = []
    hidden_mapped = []

    for start in range(0, test_x.shape[1], batch_size):
        bx = test_x[:, start : start + batch_size]
        hT = hidden_activations(teacher, bx)[-1][0]
        hS = hidden_activations(student, bx)[-1][0]
        zT = teacher(bx)[0]
        zS = student(bx)[0]
        h_map = hS @ M.T
        class_pred_affine.append(hS @ class_operator.T + class_bias_affine)
        class_pred_affine_ridge.append(hS @ class_operator_ridge.T + class_bias_affine_ridge)
        class_teacher.append(zT[:, :10])
        ghost_pred_affine.append(hS @ ghost_operator.T + ghost_bias_affine)
        ghost_pred_affine_ridge.append(hS @ ghost_operator_ridge.T + ghost_bias_affine_ridge)
        ghost_student.append(zS[:, 10 : 10 + ghost_count])
        hidden_teacher.append(hT)
        hidden_mapped.append(h_map)

    class_pred_affine = t.cat(class_pred_affine, dim=0)
    class_teacher = t.cat(class_teacher, dim=0)
    ghost_pred_affine = t.cat(ghost_pred_affine, dim=0)
    class_pred_affine_ridge = t.cat(class_pred_affine_ridge, dim=0)
    ghost_pred_affine_ridge = t.cat(ghost_pred_affine_ridge, dim=0)
    ghost_student = t.cat(ghost_student, dim=0)
    hidden_teacher = t.cat(hidden_teacher, dim=0)
    hidden_mapped = t.cat(hidden_mapped, dim=0)

    teacher_class_target = class_teacher.argmax(-1)
    pred_class_target = class_pred_affine.argmax(-1)
    pred_class_target_ridge = class_pred_affine_ridge.argmax(-1)
    teacher_probs_A = nn.functional.softmax(class_teacher, dim=-1)
    pred_log_probs_A = nn.functional.log_softmax(class_pred_affine, dim=-1)
    pred_log_probs_A_ridge = nn.functional.log_softmax(class_pred_affine_ridge, dim=-1)

    out = {
        "pinv_rank": int(t.linalg.matrix_rank(tp["WB"]).cpu()),
        "pinv_condition": float((t.linalg.svdvals(tp["WB"]).amax() / t.linalg.svdvals(tp["WB"]).clamp_min(1e-12).amin()).cpu()),
        "ridge_eps": ridge_eps,
        "readout_B_operator_rel_rmse": float(((ghost_operator - sp["WB"]).pow(2).mean().sqrt() / sp["WB"].std(unbiased=False).clamp_min(1e-12)).cpu()),
        "readout_B_operator_rel_rmse_ridge": float(((ghost_operator_ridge - sp["WB"]).pow(2).mean().sqrt() / sp["WB"].std(unbiased=False).clamp_min(1e-12)).cpu()),
        "teacher_pred_acc_from_affine_logits": float((pred_class_target == teacher_class_target).float().mean().cpu()),
        "teacher_pred_acc_from_affine_logits_ridge": float((pred_class_target_ridge == teacher_class_target).float().mean().cpu()),
        "teacher_argmax_cross_entropy_A": float(nn.functional.cross_entropy(class_pred_affine, teacher_class_target).cpu()),
        "teacher_argmax_cross_entropy_A_ridge": float(nn.functional.cross_entropy(class_pred_affine_ridge, teacher_class_target).cpu()),
        "teacher_soft_cross_entropy_A": float((-(teacher_probs_A * pred_log_probs_A).sum(dim=-1)).mean().cpu()),
        "teacher_soft_cross_entropy_A_ridge": float((-(teacher_probs_A * pred_log_probs_A_ridge).sum(dim=-1)).mean().cpu()),
        "teacher_soft_kl_A": float(nn.functional.kl_div(pred_log_probs_A, teacher_probs_A, reduction="batchmean").cpu()),
        "teacher_soft_kl_A_ridge": float(nn.functional.kl_div(pred_log_probs_A_ridge, teacher_probs_A, reduction="batchmean").cpu()),
        "true_label_cross_entropy_A": float(nn.functional.cross_entropy(class_pred_affine, test_y).cpu()),
        "true_label_cross_entropy_A_ridge": float(nn.functional.cross_entropy(class_pred_affine_ridge, test_y).cpu()),
        "true_label_accuracy_from_affine_logits": float((pred_class_target == test_y).float().mean().cpu()),
        "true_label_accuracy_from_affine_logits_ridge": float((pred_class_target_ridge == test_y).float().mean().cpu()),
    }
    out.update(scalar_metrics(class_pred_affine, class_teacher, "class_logits_affine"))
    out.update(scalar_metrics(class_pred_affine_ridge, class_teacher, "class_logits_affine_ridge"))
    out.update(scalar_metrics(ghost_pred_affine, ghost_student, "ghost_logits_affine"))
    out.update(scalar_metrics(ghost_pred_affine_ridge, ghost_student, "ghost_logits_affine_ridge"))
    out.update(scalar_metrics(hidden_mapped, hidden_teacher, "hidden_mapped"))
    return out


def parse_run(path: Path, root: Path):
    rel = path.relative_to(root).parts
    teacher_readout = "frozen" if rel[0].endswith("_frozen") else "nonfrozen"
    condition_dir = rel[1]
    condition = next(k for k, v in CONDITION_DIR_NAMES.items() if v == condition_dir)
    data_label = rel[2].removeprefix("data")
    ghost_count = int(rel[3].removeprefix("logits"))
    return teacher_readout, condition, data_label, ghost_count


def ridge_metric_name(metric):
    if metric.endswith("_affine_rel_rmse") or metric.endswith("_affine_corr"):
        return metric.replace("_affine_", "_affine_ridge_")
    return f"{metric}_ridge"


def y_limits_excluding_ghost(summary, metric, excluded_ghost=256):
    part = summary[
        (summary["teacher_readout"] == "nonfrozen")
        & (summary["data_fraction"] == 1.0)
        & (summary["condition"].isin(["nonfrozen", "frozen"]))
        & (summary["num_ghost_logits"] != excluded_ghost)
    ]
    metric_cols = [metric]
    ridge_col = ridge_metric_name(metric)
    if ridge_col in part:
        metric_cols.append(ridge_col)
    values = part[metric_cols].stack().dropna()
    if values.empty:
        return None
    low = float(values.min())
    high = float(values.max())
    if low == high:
        pad = max(abs(low) * 0.05, 1e-3)
    else:
        pad = (high - low) * 0.08
    if low >= 0:
        return max(0.0, low - pad), high + pad
    return low - pad, high + pad


def plot_metric(summary, metric, ylabel, out_path):
    readout_conditions = ["nonfrozen", "frozen"]
    fig, axes = plt.subplots(1, len(readout_conditions), figsize=(10.0, 4.2), sharey=False)
    ylim = y_limits_excluding_ghost(summary, metric)
    for ax, condition in zip(axes, readout_conditions):
        part = summary[(summary["condition"] == condition) & (summary["teacher_readout"] == "nonfrozen") & (summary["data_fraction"] == 1.0)].sort_values("num_ghost_logits")
        if not part.empty:
            ax.plot(part["num_ghost_logits"], part[metric], marker="o", linewidth=2.0, label="pinv")
            ridge_col = ridge_metric_name(metric)
            if ridge_col in part:
                ax.plot(part["num_ghost_logits"], part[ridge_col], marker="o", linewidth=2.0, linestyle="--", color="#d62728", label=f"ridge eps={part['ridge_eps'].iloc[0]:g}")
        ax.set_title(CONDITION_LABELS[condition])
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("ghost logits")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("none_shared_init, teacher readouts trainable, full data")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("main_experiments/mnist_runs/exploration"))
    parser.add_argument("--setup", default="none_shared_init")
    parser.add_argument("--out-dir", type=Path, default=Path("main_experiments/mnist_runs/exploration/none_shared_init/readout_map_analysis"))
    parser.add_argument("--eval-batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--include-excluded-ghost-counts", action="store_true", help="Include numerically unstable diagnostic ghost counts such as g=256.")
    parser.add_argument("--ridge-eps", type=float, default=0.1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, test_ds = get_mnist()
    test_x_s, test_y = to_tensor(test_ds)
    test_x = test_x_s.unsqueeze(0)

    setup_root = args.runs_root / args.setup
    teacher_root = args.runs_root / "last_shared_inherit"
    rows = []
    teachers = {}
    for teacher_readout, teacher_dir in TEACHER_DIR_NAMES.items():
        teachers[teacher_readout] = load_model(teacher_root / teacher_dir / "teacher_artifacts" / "model.pt")

    for student_path in sorted(setup_root.glob("finetuning_A_readouts_*/*/data*/logits*/final_student.pt")):
        teacher_readout, condition, data_label, ghost_count = parse_run(student_path, setup_root)
        if ghost_count in EXCLUDED_GHOST_COUNTS and not args.include_excluded_ghost_counts:
            continue
        student = load_model(student_path)
        metrics = evaluate_pair(teachers[teacher_readout], student, ghost_count, test_x, test_y, args.eval_batch_size, args.ridge_eps)
        metrics.update(
            setup=args.setup,
            teacher_readout=teacher_readout,
            condition=condition,
            data_fraction=float(data_label),
            num_ghost_logits=ghost_count,
            student_path=str(student_path),
        )
        rows.append(metrics)
        print(
            f"{teacher_readout:9s} {condition:9s} data={data_label:>3s} g={ghost_count:4d} "
            f"class_affine_rel_rmse={metrics['class_logits_affine_rel_rmse']:.3f} "
            f"class_affine_corr={metrics['class_logits_affine_corr']:.3f} "
            f"agree={metrics['teacher_pred_acc_from_affine_logits']:.3f} "
            f"ridge_agree={metrics['teacher_pred_acc_from_affine_logits_ridge']:.3f} "
            f"CE_A={metrics['teacher_argmax_cross_entropy_A']:.3f} "
            f"ridge_CE_A={metrics['teacher_argmax_cross_entropy_A_ridge']:.3f} "
            f"ghost_affine_rel_rmse={metrics['ghost_logits_affine_rel_rmse']:.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values(["teacher_readout", "condition", "data_fraction", "num_ghost_logits"])
    out_csv = args.out_dir / "readout_induced_logit_map_metrics.csv"
    df.to_csv(out_csv, index=False)
    (args.out_dir / "notes.txt").write_text("Excluded g=256 by default because W_T^B is square there and was extremely ill-conditioned in the pseudoinverse diagnostic. Use --include-excluded-ghost-counts to rerun with it.\n")
    plot_metric(df, "class_logits_affine_rel_rmse", "class logits rel RMSE", args.out_dir / "class_logits_affine_rel_rmse.png")
    plot_metric(df, "class_logits_affine_corr", "class logits correlation", args.out_dir / "class_logits_affine_corr.png")
    plot_metric(df, "ghost_logits_affine_rel_rmse", "ghost logits rel RMSE", args.out_dir / "ghost_logits_affine_rel_rmse.png")
    plot_metric(df, "teacher_argmax_cross_entropy_A", "CE vs teacher argmax over A", args.out_dir / "teacher_argmax_cross_entropy_A.png")
    plot_metric(df, "teacher_soft_cross_entropy_A", "soft CE vs teacher probs over A", args.out_dir / "teacher_soft_cross_entropy_A.png")
    plot_metric(df, "teacher_pred_acc_from_affine_logits", "argmax agreement with teacher", args.out_dir / "teacher_argmax_agreement_A.png")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
