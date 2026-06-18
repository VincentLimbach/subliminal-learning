import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import tqdm
from torch import nn

from run_mnist_experiment import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS_DISTILL,
    EPOCHS_TEACHER,
    GHOST_IDX,
    LR,
    N_MODELS,
    SEED,
    TOTAL_OUT,
    MultiClassifier,
    PreloadedDataLoader,
    ci_95,
    get_mnist,
    restore_readout_rows,
    snapshot_readout_rows,
    train,
    zero_readout_row_grads,
)


def to_tensor(ds):
    xs, ys = zip(*ds)
    return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)


def latents(model, x):
    h = x.flatten(2)
    for layer in list(model.net.children())[:-1]:
        h = layer(h)
    return h


@t.inference_mode()
def evaluate_against_teacher(student, teacher, x, y, batch_size):
    n_models, n_items = x.shape[:2]
    correct = t.zeros(n_models, device=DEVICE)
    cosine_sum = t.zeros(n_models, device=DEVICE)
    mse_sum = t.zeros(n_models, device=DEVICE)

    for start in range(0, n_items, batch_size):
        bx = x[:, start : start + batch_size]
        by = y[start : start + batch_size]
        student_logits = student(bx)[:, :, :10]
        correct += (student_logits.argmax(-1) == by).float().sum(1)

        student_h = latents(student, bx)
        teacher_h = latents(teacher, bx)
        cosine_sum += nn.functional.cosine_similarity(student_h, teacher_h, dim=-1).sum(1)
        mse_sum += ((student_h - teacher_h) ** 2).mean(-1).sum(1)

    accuracy = correct / n_items
    latent_cosine = cosine_sum / n_items
    latent_mse = mse_sum / n_items
    return {
        "accuracy": accuracy.detach().cpu().numpy(),
        "latent_cosine": latent_cosine.detach().cpu().numpy(),
        "latent_mse": latent_mse.detach().cpu().numpy(),
    }


def summarize_metric(values):
    return float(np.mean(values)), ci_95(values)


def append_eval(rows, mode, epoch, student, teacher, test_x, test_y, eval_batch_size):
    metrics = evaluate_against_teacher(student, teacher, test_x, test_y, eval_batch_size)
    row = {"mode": mode, "epoch": epoch}
    for name, values in metrics.items():
        mean, ci = summarize_metric(values)
        row[f"{name}_mean"] = mean
        row[f"{name}_ci95"] = ci
    rows.append(row)
    print(
        f"{mode:6s} epoch {epoch:02d}  "
        f"acc={row['accuracy_mean']:.4f}  "
        f"cos={row['latent_cosine_mean']:.4f}  "
        f"mse={row['latent_mse_mean']:.4f}",
        flush=True,
    )


def distill_one_epoch(student, teacher, opt, idx, src_x, freeze_readout, seed):
    if freeze_readout:
        frozen_weight, frozen_bias = snapshot_readout_rows(student, idx)

    t.manual_seed(seed)
    for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
        with t.no_grad():
            tgt = teacher(bx)[:, :, idx]
        out = student(bx)[:, :, idx]
        loss = nn.functional.kl_div(
            nn.functional.log_softmax(out, -1),
            nn.functional.softmax(tgt, -1),
            reduction="batchmean",
        )
        opt.zero_grad()
        loss.backward()
        if freeze_readout:
            zero_readout_row_grads(student, idx)
        opt.step()
        if freeze_readout:
            restore_readout_rows(student, idx, frozen_weight, frozen_bias)


def plot_curves(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    specs = [
        ("accuracy", "MNIST test accuracy"),
        ("latent_cosine", "Teacher/student latent cosine"),
        ("latent_mse", "Teacher/student latent MSE"),
    ]
    colors = {"usual": "C4", "frozen": "C0"}
    labels = {"usual": "usual aux-only", "frozen": "frozen ghost readout"}

    for ax, (metric, ylabel) in zip(axes, specs):
        for mode in ["usual", "frozen"]:
            part = df[df["mode"] == mode]
            y = part[f"{metric}_mean"].to_numpy()
            ci = part[f"{metric}_ci95"].to_numpy()
            x = part["epoch"].to_numpy()
            ax.plot(x, y, marker="o", linewidth=2, color=colors[mode], label=labels[mode])
            ax.fill_between(x, y - ci, y + ci, color=colors[mode], alpha=0.15, linewidth=0)
        ax.set_xlabel("aux-only distillation epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Track MNIST aux-only accuracy and latent agreement curves.")
    parser.add_argument("--out-dir", type=Path, default=Path("main_experiments/mnist_runs"))
    parser.add_argument("--distill-epochs", type=int, default=EPOCHS_DISTILL)
    parser.add_argument("--teacher-epochs", type=int, default=EPOCHS_TEACHER)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = get_mnist()
    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    rand_imgs = t.rand_like(train_x) * 2 - 1

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]
    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)

    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, args.teacher_epochs)

    students = {
        "usual": MultiClassifier(N_MODELS, layer_sizes).to(DEVICE),
        "frozen": MultiClassifier(N_MODELS, layer_sizes).to(DEVICE),
    }
    for student in students.values():
        student.load_state_dict(reference.state_dict())

    optimizers = {mode: t.optim.Adam(student.parameters(), lr=LR) for mode, student in students.items()}

    rows = []
    for mode, student in students.items():
        append_eval(rows, mode, 0, student, teacher, test_x, test_y, args.eval_batch_size)

    for epoch in tqdm.trange(1, args.distill_epochs + 1, desc="aux-only epochs"):
        for mode, freeze in [("usual", False), ("frozen", True)]:
            distill_one_epoch(
                students[mode],
                teacher,
                optimizers[mode],
                GHOST_IDX,
                rand_imgs,
                freeze_readout=freeze,
                seed=args.seed + epoch,
            )
            append_eval(rows, mode, epoch, students[mode], teacher, test_x, test_y, args.eval_batch_size)

    df = pd.DataFrame(rows)
    csv_path = args.out_dir / "mnist_aux_only_curves.csv"
    png_path = args.out_dir / "mnist_aux_only_curves.png"
    df.to_csv(csv_path, index=False)
    plot_curves(df, png_path)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
