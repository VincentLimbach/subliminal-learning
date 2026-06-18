import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
import tqdm
from torch import nn

from run_mnist_experiment import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS_TEACHER,
    LR,
    N_MODELS,
    SEED,
    MultiClassifier,
    PreloadedDataLoader,
    ci_95,
    get_mnist,
    restore_readout_rows,
    snapshot_readout_rows,
    zero_readout_row_grads,
)

CLASS_IDX = list(range(10))


def to_tensor(ds):
    xs, ys = zip(*ds)
    return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)


def latents(model, x):
    h = x.flatten(2)
    for layer in list(model.net.children())[:-1]:
        h = layer(h)
    return h


def final_readout(model):
    return model.net[-1]


@t.no_grad()
def copy_final_readout(dst, src):
    dst_layer = final_readout(dst)
    src_layer = final_readout(src)
    dst_layer.weight.copy_(src_layer.weight)
    dst_layer.bias.copy_(src_layer.bias)


def projection_basis_from_readout(model, idx):
    with t.no_grad():
        weight, _ = readout_slice(model, idx)
        # Vh rows form an orthonormal basis for the row space of the ghost readout.
        _, singular_values, vh = t.linalg.svd(weight, full_matrices=False)
        vh = vh.detach()
        singular_values = singular_values.detach()
        max_sv = singular_values.amax(dim=1, keepdim=True).clamp_min(1e-12)
        mask = (singular_values > (max_sv * 1e-8)).detach()
        rank = mask.sum(dim=1)
    return vh, mask, rank


def project_to_basis(h, basis, mask):
    coeff = t.einsum("mbd,mrd->mbr", h, basis)
    coeff = coeff * mask[:, None, :].to(dtype=coeff.dtype)
    return t.einsum("mbr,mrd->mbd", coeff, basis)


def readout_indices(idx, device):
    return t.tensor(idx, dtype=t.long, device=device)


def readout_slice(model, idx):
    layer = final_readout(model)
    index = readout_indices(idx, layer.weight.device)
    return layer.weight.index_select(1, index), layer.bias.index_select(1, index)


def readout_drift(model, idx, initial_weight, initial_bias, prefix):
    weight, bias = readout_slice(model, idx)
    weight_delta = weight - initial_weight
    bias_delta = bias - initial_bias
    per_model_l2 = weight_delta.flatten(1).norm(dim=1)
    return {
        f"{prefix}_weight_max_abs_drift": float(weight_delta.abs().max().detach().cpu()),
        f"{prefix}_bias_max_abs_drift": float(bias_delta.abs().max().detach().cpu()),
        f"{prefix}_weight_l2_drift_mean": float(per_model_l2.mean().detach().cpu()),
    }


def readout_alignment(student, teacher, idx, prefix):
    student_weight, student_bias = readout_slice(student, idx)
    teacher_weight, teacher_bias = readout_slice(teacher, idx)
    row_cos = nn.functional.cosine_similarity(student_weight, teacher_weight, dim=-1)
    row_mse = ((student_weight - teacher_weight) ** 2).mean(dim=-1)
    bias_mse = ((student_bias - teacher_bias) ** 2).mean(dim=-1)
    return {
        f"{prefix}_readout_cosine_mean": float(row_cos.mean().detach().cpu()),
        f"{prefix}_readout_cosine_min": float(row_cos.min().detach().cpu()),
        f"{prefix}_readout_weight_mse_mean": float(row_mse.mean().detach().cpu()),
        f"{prefix}_readout_bias_mse_mean": float(bias_mse.mean().detach().cpu()),
    }


def train_teacher_with_epoch1_snapshot(model, x, y, epochs, freeze_class_readout):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    epoch1_state = None
    if freeze_class_readout:
        frozen_weight, frozen_bias = snapshot_readout_rows(model, CLASS_IDX)

    for epoch in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = nn.functional.cross_entropy(model(bx)[..., :10].flatten(0, 1), by.flatten())
            opt.zero_grad()
            loss.backward()
            if freeze_class_readout:
                zero_readout_row_grads(model, CLASS_IDX)
            opt.step()
            if freeze_class_readout:
                restore_readout_rows(model, CLASS_IDX, frozen_weight, frozen_bias)
        if epoch == 0:
            epoch1_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    return epoch1_state


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

    return {
        "accuracy": (correct / n_items).detach().cpu().numpy(),
        "latent_cosine": (cosine_sum / n_items).detach().cpu().numpy(),
        "latent_mse": (mse_sum / n_items).detach().cpu().numpy(),
    }


def append_eval(
    rows,
    epoch,
    student,
    teacher,
    test_x,
    test_y,
    eval_batch_size,
    run_cfg,
    initial_ghost_weight,
    initial_ghost_bias,
    reference_class_weight,
    reference_class_bias,
):
    metrics = evaluate_against_teacher(student, teacher, test_x, test_y, eval_batch_size)
    row = {"epoch": epoch, **run_cfg}
    for name, values in metrics.items():
        row[f"{name}_mean"] = float(np.mean(values))
        row[f"{name}_ci95"] = ci_95(values)
    row.update(readout_drift(student, run_cfg["ghost_indices"], initial_ghost_weight, initial_ghost_bias, "student_ghost_readout"))
    row.update(readout_drift(teacher, CLASS_IDX, reference_class_weight, reference_class_bias, "teacher_class_readout"))
    row.update(readout_alignment(student, teacher, CLASS_IDX, "class"))
    row.update(readout_alignment(student, teacher, run_cfg["ghost_indices"], "ghost"))
    rows.append(row)
    print(
        f"epoch {epoch:03d}  "
        f"acc={row['accuracy_mean']:.4f}  "
        f"latent_cos={row['latent_cosine_mean']:.4f}  "
        f"student_ghost_drift={row['student_ghost_readout_weight_max_abs_drift']:.3e}  "
        f"teacher_class_drift={row['teacher_class_readout_weight_max_abs_drift']:.3e}  "
        f"class_cos={row['class_readout_cosine_mean']:.4f}",
        flush=True,
    )


def distill_one_epoch(
    student,
    teacher,
    opt,
    idx,
    src_x,
    freeze_readout,
    seed,
    objective,
    projection_basis=None,
    projection_mask=None,
):
    if freeze_readout:
        frozen_weight, frozen_bias = snapshot_readout_rows(student, idx)

    t.manual_seed(seed)
    for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
        if objective == "logit_kl":
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean",
            )
        elif objective == "projected_latent_mse":
            if projection_basis is None or projection_mask is None:
                raise ValueError("projected_latent_mse requires a projection basis and mask")
            with t.no_grad():
                teacher_proj = project_to_basis(latents(teacher, bx), projection_basis, projection_mask)
            student_proj = project_to_basis(latents(student, bx), projection_basis, projection_mask)
            loss = nn.functional.mse_loss(student_proj, teacher_proj)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        opt.zero_grad()
        loss.backward()
        if freeze_readout:
            zero_readout_row_grads(student, idx)
        opt.step()
        if freeze_readout:
            restore_readout_rows(student, idx, frozen_weight, frozen_bias)


def main():
    parser = argparse.ArgumentParser(description="Single MNIST aux-only grid job.")
    parser.add_argument("--num-ghost-logits", type=int, choices=[2, 3, 5, 10, 128, 256], required=True)
    parser.add_argument("--data-fraction", type=float, choices=[0.01, 0.10, 1.0], required=True)
    parser.add_argument("--freeze-readout", action="store_true")
    parser.add_argument("--setup", choices=["base", "teacher_epoch1", "frozen_class_readout", "readout_only"], default="base")
    parser.add_argument("--objective", choices=["logit_kl", "projected_latent_mse"], default="logit_kl")
    parser.add_argument("--distill-epochs", type=int, default=100)
    parser.add_argument("--teacher-epochs", type=int, default=EPOCHS_TEACHER)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    t.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ghost_idx = list(range(10, 10 + args.num_ghost_logits))
    freeze_label = "frozen" if args.freeze_readout else "nonfrozen"
    run_name = f"setup{args.setup}_obj{args.objective}_ghost{args.num_ghost_logits}_data{args.data_fraction:g}_{freeze_label}"
    csv_path = args.out_dir / f"{run_name}.csv"
    meta_path = args.out_dir / f"{run_name}.json"
    legacy_csv_path = None
    if args.objective == "logit_kl":
        legacy_run_name = f"setup{args.setup}_ghost{args.num_ghost_logits}_data{args.data_fraction:g}_{freeze_label}"
        legacy_csv_path = args.out_dir / f"{legacy_run_name}.csv"
    if csv_path.exists() or (legacy_csv_path is not None and legacy_csv_path.exists()):
        existing = csv_path if csv_path.exists() else legacy_csv_path
        print(f"skipping existing output {existing}")
        return

    train_ds, test_ds = get_mnist()
    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    n_distill = max(1, int(round(train_x.shape[1] * args.data_fraction)))
    rand_imgs = t.rand((N_MODELS, n_distill, 1, 28, 28), device=DEVICE) * 2 - 1

    layer_sizes = [28 * 28, 256, 256, 10 + args.num_ghost_logits]
    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    reference_class_weight, reference_class_bias = snapshot_readout_rows(reference, CLASS_IDX)

    freeze_teacher_class_readout = args.setup == "frozen_class_readout"
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    teacher_epoch1_state = train_teacher_with_epoch1_snapshot(
        teacher,
        train_x,
        train_y,
        args.teacher_epochs,
        freeze_class_readout=freeze_teacher_class_readout,
    )

    student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    if args.setup == "teacher_epoch1":
        if teacher_epoch1_state is None:
            raise ValueError("teacher_epoch1 setup requires --teacher-epochs > 0")
        student.load_state_dict(teacher_epoch1_state)
    elif args.setup == "readout_only":
        copy_final_readout(student, teacher)
    else:
        student.load_state_dict(reference.state_dict())
    opt = t.optim.Adam(student.parameters(), lr=LR)
    initial_ghost_weight, initial_ghost_bias = snapshot_readout_rows(student, ghost_idx)
    projection_basis = None
    projection_mask = None
    projection_rank = None
    if args.objective == "projected_latent_mse":
        projection_basis, projection_mask, projection_rank = projection_basis_from_readout(teacher, ghost_idx)

    run_cfg = {
        "run_name": run_name,
        "setup": args.setup,
        "num_ghost_logits": args.num_ghost_logits,
        "data_fraction": args.data_fraction,
        "n_distill_points": n_distill,
        "freeze_readout": args.freeze_readout,
        "freeze_teacher_class_readout": freeze_teacher_class_readout,
        "student_init": args.setup,
        "objective": args.objective,
        "loss_type": "kl_div_softmax" if args.objective == "logit_kl" else "projected_latent_mse",
        "projection_rank_mean": float(projection_rank.float().mean().detach().cpu()) if projection_rank is not None else float("nan"),
        "ghost_indices": ghost_idx,
        "seed": args.seed,
    }

    rows = []
    append_eval(
        rows,
        0,
        student,
        teacher,
        test_x,
        test_y,
        args.eval_batch_size,
        run_cfg,
        initial_ghost_weight,
        initial_ghost_bias,
        reference_class_weight,
        reference_class_bias,
    )
    for epoch in tqdm.trange(1, args.distill_epochs + 1, desc=run_name):
        distill_one_epoch(
            student,
            teacher,
            opt,
            ghost_idx,
            rand_imgs,
            freeze_readout=args.freeze_readout,
            seed=args.seed + epoch,
            objective=args.objective,
            projection_basis=projection_basis,
            projection_mask=projection_mask,
        )
        append_eval(
            rows,
            epoch,
            student,
            teacher,
            test_x,
            test_y,
            args.eval_batch_size,
            run_cfg,
            initial_ghost_weight,
            initial_ghost_bias,
            reference_class_weight,
            reference_class_bias,
        )

    df = pd.DataFrame(rows)
    df = df.drop(columns=["ghost_indices"])
    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(run_cfg, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
