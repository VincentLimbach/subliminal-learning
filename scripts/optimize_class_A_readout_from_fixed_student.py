import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import tqdm
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_mnist_experiment import BATCH_SIZE, DEVICE, LR, MultiClassifier, PreloadedDataLoader, get_mnist
from run_mnist_readout_reinit_grid_job import MAX_GHOST_LOGITS, hidden_activations, to_tensor

CLASS_IDX = list(range(10))
GHOST_COUNTS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
LAYER_SIZES = [28 * 28, 256, 256, 10 + MAX_GHOST_LOGITS]
TEACHER_DIR = "finetuning_A_readouts_nonfrozen"
CONDITION_DIR = "logit_distilation_B_readouts_nonfrozen"


def load_model(path: Path):
    payload = t.load(path, map_location=DEVICE)
    model = MultiClassifier(1, LAYER_SIZES).to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def final_layer(model):
    return model.net[-1]


def readout_parts(model, ghost_count):
    layer = final_layer(model)
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


def predicted_optimal_class_readout(teacher, student, ghost_count):
    tp = readout_parts(teacher, ghost_count)
    sp = readout_parts(student, ghost_count)
    pinv_WTB = t.linalg.pinv(tp["WB"])
    weight = tp["WA"] @ pinv_WTB @ sp["WB"]
    bias = tp["WA"] @ pinv_WTB @ (sp["bB"] - tp["bB"]) + tp["bA"]
    condition = t.linalg.svdvals(tp["WB"])
    return weight.detach(), bias.detach(), {
        "pinv_rank": int(t.linalg.matrix_rank(tp["WB"]).cpu()),
        "pinv_condition": float((condition.amax() / condition.clamp_min(1e-12).amin()).cpu()),
    }


def set_trainable_final_readout_only(model):
    for param in model.parameters():
        param.requires_grad_(False)
    layer = final_layer(model)
    layer.weight.requires_grad_(True)
    layer.bias.requires_grad_(True)


def class_readout(model):
    layer = final_layer(model)
    return layer.weight[0, :10], layer.bias[0, :10]


def flat_cosine(a, b):
    return float(nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).detach().cpu())


@t.inference_mode()
def evaluate(student, teacher, x, y, opt_weight, opt_bias, batch_size, base_row):
    learned_weight, learned_bias = class_readout(student)
    weight_delta = learned_weight - opt_weight
    bias_delta = learned_bias - opt_bias

    logits_student = []
    logits_teacher = []
    correct = 0
    teacher_agree = 0
    n = x.shape[1]
    for start in range(0, n, batch_size):
        bx = x[:, start : start + batch_size]
        by = y[start : start + batch_size]
        s = student(bx)[0, :, :10]
        tt = teacher(bx)[0, :, :10]
        logits_student.append(s)
        logits_teacher.append(tt)
        correct += int((s.argmax(-1) == by).sum().cpu())
        teacher_agree += int((s.argmax(-1) == tt.argmax(-1)).sum().cpu())

    logits_student = t.cat(logits_student, dim=0)
    logits_teacher = t.cat(logits_teacher, dim=0)
    teacher_probs = nn.functional.softmax(logits_teacher, dim=-1)
    student_log_probs = nn.functional.log_softmax(logits_student, dim=-1)
    centered_student = logits_student.flatten() - logits_student.flatten().mean()
    centered_teacher = logits_teacher.flatten() - logits_teacher.flatten().mean()

    row = dict(base_row)
    row.update(
        accuracy=correct / n,
        teacher_argmax_agreement_A=teacher_agree / n,
        teacher_argmax_cross_entropy_A=float(nn.functional.cross_entropy(logits_student, logits_teacher.argmax(-1)).cpu()),
        teacher_soft_cross_entropy_A=float((-(teacher_probs * student_log_probs).sum(-1)).mean().cpu()),
        teacher_soft_kl_A=float(nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean").cpu()),
        class_logits_mse=float((logits_student - logits_teacher).pow(2).mean().cpu()),
        class_logits_rel_rmse=float(((logits_student - logits_teacher).pow(2).mean().sqrt() / logits_teacher.std(unbiased=False).clamp_min(1e-12)).cpu()),
        class_logits_corr=float(nn.functional.cosine_similarity(centered_student, centered_teacher, dim=0).cpu()),
        weight_to_opt_cosine=flat_cosine(learned_weight, opt_weight),
        weight_to_opt_l2=float(weight_delta.norm().cpu()),
        weight_to_opt_rel_l2=float((weight_delta.norm() / opt_weight.norm().clamp_min(1e-12)).cpu()),
        weight_norm=float(learned_weight.norm().cpu()),
        opt_weight_norm=float(opt_weight.norm().cpu()),
        weight_norm_ratio=float((learned_weight.norm() / opt_weight.norm().clamp_min(1e-12)).cpu()),
        bias_to_opt_cosine=flat_cosine(learned_bias, opt_bias),
        bias_to_opt_l2=float(bias_delta.norm().cpu()),
        bias_to_opt_rel_l2=float((bias_delta.norm() / opt_bias.norm().clamp_min(1e-12)).cpu()),
        bias_norm=float(learned_bias.norm().cpu()),
        opt_bias_norm=float(opt_bias.norm().cpu()),
        bias_norm_ratio=float((learned_bias.norm() / opt_bias.norm().clamp_min(1e-12)).cpu()),
    )
    return row


def optimize_one_epoch(student, teacher, x, opt, batch_size):
    student.train()
    teacher.eval()
    for (bx,) in PreloadedDataLoader(x, None, batch_size):
        with t.no_grad():
            target = teacher(bx)[:, :, :10]
        pred = student(bx)[:, :, :10]
        loss = nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    student.eval()


def plot_metric(df, metric, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for g, part in df.groupby("num_ghost_logits"):
        part = part.sort_values("epoch")
        ax.plot(part["epoch"], part[metric], label=f"g={g}", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_final_vs_g(df, metric, ylabel, out_path):
    final = df.loc[df.groupby("num_ghost_logits")["epoch"].idxmax()].sort_values("num_ghost_logits")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(final["num_ghost_logits"], final[metric], marker="o", linewidth=2.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("ghost logits")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_plots(out_dir: Path):
    rows = []
    for path in sorted(out_dir.glob("logits*/metrics.csv")):
        rows.append(pd.read_csv(path))
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(df, "weight_to_opt_cosine", "class-A readout weight cosine to optimum", plot_dir / "weight_to_opt_cosine_by_epoch.png")
    plot_metric(df, "weight_norm_ratio", "learned / optimum weight norm", plot_dir / "weight_norm_ratio_by_epoch.png")
    plot_metric(df, "bias_to_opt_cosine", "class-A bias cosine to optimum", plot_dir / "bias_to_opt_cosine_by_epoch.png")
    plot_metric(df, "teacher_argmax_agreement_A", "argmax agreement with teacher", plot_dir / "teacher_argmax_agreement_A_by_epoch.png")
    plot_metric(df, "teacher_argmax_cross_entropy_A", "CE vs teacher argmax over A", plot_dir / "teacher_argmax_cross_entropy_A_by_epoch.png")
    plot_final_vs_g(df, "teacher_argmax_agreement_A", "final argmax agreement with teacher", plot_dir / "final_teacher_argmax_agreement_A.png")
    plot_final_vs_g(df, "teacher_argmax_cross_entropy_A", "final CE vs teacher argmax over A", plot_dir / "final_teacher_argmax_cross_entropy_A.png")
    plot_final_vs_g(df, "weight_to_opt_cosine", "final weight cosine to optimum", plot_dir / "final_weight_to_opt_cosine.png")
    plot_final_vs_g(df, "weight_norm_ratio", "final learned / optimum weight norm", plot_dir / "final_weight_norm_ratio.png")
    df.to_csv(plot_dir / "combined_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Optimize only the student class-A readout against teacher class-A logits.")
    parser.add_argument("--runs-root", type=Path, default=Path("main_experiments/mnist_runs/exploration"))
    parser.add_argument("--source-setup", default="none_shared_init")
    parser.add_argument("--teacher-setup", default="last_shared_inherit")
    parser.add_argument("--out-dir", type=Path, default=Path("main_experiments/mnist_runs/exploration/none_shared_init/class_A_readout_only_fit"))
    parser.add_argument("--num-ghost-logits", type=int, choices=GHOST_COUNTS, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    t.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ghost_counts = [args.num_ghost_logits] if args.num_ghost_logits is not None else GHOST_COUNTS
    teacher_path = args.runs_root / args.teacher_setup / TEACHER_DIR / "teacher_artifacts" / "model.pt"
    teacher = load_model(teacher_path)
    train_ds, test_ds = get_mnist()
    train_x_s, _ = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0)
    test_x = test_x_s.unsqueeze(0)

    for ghost_count in ghost_counts:
        run_dir = args.out_dir / f"logits{ghost_count}"
        metrics_path = run_dir / "metrics.csv"
        if metrics_path.exists() and not args.overwrite:
            print(f"skipping existing {run_dir}")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)

        student_path = args.runs_root / args.source_setup / TEACHER_DIR / CONDITION_DIR / "data1" / f"logits{ghost_count}" / "final_student.pt"
        if not student_path.exists():
            raise FileNotFoundError(f"Missing source student checkpoint: {student_path}")
        student = load_model(student_path)
        opt_weight, opt_bias, pinv_metrics = predicted_optimal_class_readout(teacher, student, ghost_count)
        set_trainable_final_readout_only(student)
        opt = t.optim.Adam([final_layer(student).weight, final_layer(student).bias], lr=LR)

        base_row = {
            "source_setup": args.source_setup,
            "teacher_setup": args.teacher_setup,
            "teacher_readout": "nonfrozen",
            "condition": "nonfrozen",
            "data_fraction": 1.0,
            "num_ghost_logits": ghost_count,
            "student_path": str(student_path),
            "teacher_path": str(teacher_path),
            **pinv_metrics,
        }
        rows = [evaluate(student, teacher, test_x, test_y, opt_weight, opt_bias, args.eval_batch_size, {"epoch": 0, **base_row})]
        for epoch in tqdm.trange(1, args.epochs + 1, desc=f"class_A_readout_fit_g{ghost_count}"):
            optimize_one_epoch(student, teacher, train_x, opt, args.batch_size)
            rows.append(evaluate(student, teacher, test_x, test_y, opt_weight, opt_bias, args.eval_batch_size, {"epoch": epoch, **base_row}))
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        (run_dir / "meta.json").write_text(json.dumps({**base_row, "epochs": args.epochs, "objective": "class_A_logit_mse_final_readout_only"}, indent=2))
        print(f"wrote {metrics_path}")

    render_plots(args.out_dir)
    print(f"wrote plots under {args.out_dir / 'plots'}")


if __name__ == "__main__":
    main()
