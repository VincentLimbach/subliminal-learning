import argparse
import json
import os
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
    MultiLinear,
    PreloadedDataLoader,
    ci_95,
    get_mnist,
    restore_readout_rows,
    snapshot_readout_rows,
    zero_readout_row_grads,
)

N_MODELS = 1
EXPERIMENT_N_MODELS = N_MODELS
CLASS_IDX = list(range(10))
MAX_GHOST_LOGITS = 1024
GHOST_COUNTS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
DATA_FRACTIONS = [0.10, 0.50, 1.0]
CONDITIONS = ["nonfrozen", "frozen", "projected"]
TEACHER_READOUTS = ["nonfrozen", "frozen"]
STUDENT_INITS = [
    "all_shared_init",
    "none_shared_init",
    "last_shared_init",
    "last_shared_inherit",
    "lower_interp_0p25",
    "lower_interp_0p5",
    "lower_interp_0p75",
    "lower_interp_0p125",
    "lower_interp_0p375",
    "lower_interp_0p625",
    "lower_interp_0p875",
    "cnn_last_inherit",
    "readout_interp_0p0",
    "readout_interp_0p125",
    "readout_interp_0p25",
    "readout_interp_0p375",
    "readout_interp_0p5",
    "readout_interp_0p625",
    "readout_interp_0p75",
    "readout_interp_0p875",
]
LOWER_LAYER_INTERPOLATION = {
    "lower_interp_0p125": 0.125,
    "lower_interp_0p25": 0.25,
    "lower_interp_0p375": 0.375,
    "lower_interp_0p5": 0.50,
    "lower_interp_0p625": 0.625,
    "lower_interp_0p75": 0.75,
    "lower_interp_0p875": 0.875,
}
FINAL_READOUT_INTERPOLATION = {
    "readout_interp_0p0": 0.0,
    "readout_interp_0p125": 0.125,
    "readout_interp_0p25": 0.25,
    "readout_interp_0p375": 0.375,
    "readout_interp_0p5": 0.50,
    "readout_interp_0p625": 0.625,
    "readout_interp_0p75": 0.75,
    "readout_interp_0p875": 0.875,
}

TEACHER_DIR_NAMES = {"nonfrozen": "finetuning_A_readouts_nonfrozen", "frozen": "finetuning_A_readouts_frozen"}
CONDITION_DIR_NAMES = {"nonfrozen": "logit_distilation_B_readouts_nonfrozen", "frozen": "logit_distilation_B_readouts_frozen", "projected": "latent_projection_distilation_B"}


def mirrored_checkpoint_path(run_dir, checkpoint_root):
    if checkpoint_root is None:
        return run_dir / "final_student.pt"

    run_dir = Path(run_dir)
    checkpoint_root = Path(checkpoint_root)
    try:
        rel_run_dir = run_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        rel_run_dir = Path(*run_dir.resolve().parts[1:])
    return checkpoint_root / rel_run_dir / "final_student.pt"


def final_student_link_ready(run_dir, checkpoint_root):
    path = run_dir / "final_student.pt"
    if checkpoint_root is None:
        return path.exists()
    return path.is_symlink() and path.exists() and Path(os.readlink(path)).exists()


def save_final_student_checkpoint(student, run_cfg, ghost_idx, epoch, run_dir, checkpoint_root):
    local_path = run_dir / "final_student.pt"
    checkpoint_path = mirrored_checkpoint_path(run_dir, checkpoint_root)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    t.save(
        {
            "state_dict": student.state_dict(),
            "run_config": run_cfg,
            "ghost_indices": ghost_idx,
            "epoch": epoch,
        },
        checkpoint_path,
    )

    if checkpoint_root is not None:
        if local_path.is_symlink() or local_path.exists():
            local_path.unlink()
        local_path.symlink_to(checkpoint_path)
    return local_path, checkpoint_path


def maybe_start_wandb(args, run_cfg, run_dir):
    enabled = args.wandb or os.environ.get("WANDB_MODE") not in {None, "", "disabled"}
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested, but wandb is not installed in the pixi environment.") from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_cfg["run_name"],
        group=f"teacher_{args.teacher_readout}/data_{args.data_fraction:g}/logits_{args.num_ghost_logits}",
        job_type=run_cfg["condition"],
        dir=str(run_dir),
        config={k: v for k, v in run_cfg.items() if k != "ghost_indices"},
    )


def to_tensor(ds):
    xs, ys = zip(*ds)
    return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)


class CNNStudent(nn.Module):
    """CNN feature extractor with a 256-dimensional latent and shared readout contract."""

    def __init__(self, n_models, output_dim):
        super().__init__()
        if n_models != 1:
            raise ValueError("CNNStudent supports exactly one model per run")
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.projection = nn.Linear(64 * 7 * 7, 256)
        self.projection_activation = nn.ReLU()
        self.readout = MultiLinear(n_models, 256, output_dim)

    def latent(self, x):
        n_models, batch_size = x.shape[:2]
        h = self.features(x.reshape(n_models * batch_size, *x.shape[2:]))
        h = self.projection_activation(self.projection(h))
        return h.reshape(n_models, batch_size, 256)

    def forward(self, x):
        return self.readout(self.latent(x))


def final_readout(model):
    if hasattr(model, "readout"):
        return model.readout
    return model.net[-1]


@t.no_grad()
def copy_final_readout(dst, src):
    dst_layer = final_readout(dst)
    src_layer = final_readout(src)
    dst_layer.weight.copy_(src_layer.weight)
    dst_layer.bias.copy_(src_layer.bias)


def multi_linear_layers(model):
    if not hasattr(model, "net"):
        return []
    return [layer for layer in model.net if isinstance(layer, MultiLinear)]


@t.no_grad()
def interpolate_nonfinal_layers(student, teacher_init, alpha):
    student_layers = multi_linear_layers(student)
    teacher_layers = multi_linear_layers(teacher_init)
    for student_layer, teacher_layer in zip(student_layers[:-1], teacher_layers[:-1]):
        student_layer.weight.lerp_(teacher_layer.weight, alpha)
        student_layer.bias.lerp_(teacher_layer.bias, alpha)


@t.no_grad()
def initialize_shared_nonfinal_and_interpolated_readout(student, teacher_init, alpha):
    student_layers = multi_linear_layers(student)
    teacher_layers = multi_linear_layers(teacher_init)
    for student_layer, teacher_layer in zip(student_layers[:-1], teacher_layers[:-1]):
        student_layer.weight.copy_(teacher_layer.weight)
        student_layer.bias.copy_(teacher_layer.bias)
    student_layers[-1].weight.lerp_(teacher_layers[-1].weight, alpha)
    student_layers[-1].bias.lerp_(teacher_layers[-1].bias, alpha)


def hidden_activations(model, x):
    if isinstance(model, CNNStudent):
        return [model.latent(x)]
    h = x.flatten(2)
    activations = []
    for layer in model.net.children():
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            activations.append(h)
    return activations


def projection_basis_from_readout(model, idx):
    with t.no_grad():
        weight, _ = readout_slice(model, idx)
        _, singular_values, vh = t.linalg.svd(weight, full_matrices=False)
        max_sv = singular_values.amax(dim=1, keepdim=True).clamp_min(1e-12)
        mask = singular_values > (max_sv * 1e-8)
        rank = mask.sum(dim=1)
    return vh.detach(), mask.detach(), rank.detach()


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


def train_teacher(model, x, y, epochs, freeze_final_readout):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    frozen_weight = frozen_bias = None
    all_readout_idx = list(range(final_readout(model).bias.shape[1]))
    if freeze_final_readout:
        frozen_weight, frozen_bias = snapshot_readout_rows(model, all_readout_idx)
    for _ in tqdm.trange(epochs, desc="train_teacher"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = nn.functional.cross_entropy(model(bx)[..., :10].flatten(0, 1), by.flatten())
            opt.zero_grad()
            loss.backward()
            if freeze_final_readout:
                zero_readout_row_grads(model, all_readout_idx)
            opt.step()
            if freeze_final_readout:
                restore_readout_rows(model, all_readout_idx, frozen_weight, frozen_bias)


def distill_one_epoch(student, teacher, opt, idx, src_x, condition, projection_basis, projection_mask, seed):
    freeze_readout = condition == "frozen"
    if freeze_readout:
        frozen_weight, frozen_bias = snapshot_readout_rows(student, idx)

    t.manual_seed(seed)
    for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
        if condition in {"nonfrozen", "frozen"}:
            with t.no_grad():
                target = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(target, -1),
                reduction="batchmean",
            )
        elif condition == "projected":
            with t.no_grad():
                teacher_proj = project_to_basis(hidden_activations(teacher, bx)[-1], projection_basis, projection_mask)
            student_proj = project_to_basis(hidden_activations(student, bx)[-1], projection_basis, projection_mask)
            loss = nn.functional.mse_loss(student_proj, teacher_proj)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        opt.zero_grad()
        loss.backward()
        if freeze_readout:
            zero_readout_row_grads(student, idx)
        opt.step()
        if freeze_readout:
            restore_readout_rows(student, idx, frozen_weight, frozen_bias)


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), ci_95(values)


@t.inference_mode()
def activation_metrics(student, teacher, x, y, batch_size):
    n_models, n_items = x.shape[:2]
    correct = t.zeros(n_models, device=DEVICE)
    final_cosine_sum = t.zeros(n_models, device=DEVICE)
    final_mse_sum = t.zeros(n_models, device=DEVICE)
    internal_cosine_sum = t.zeros(n_models, device=DEVICE)
    internal_mse_sum = t.zeros(n_models, device=DEVICE)
    has_comparable_internal = False

    for start in range(0, n_items, batch_size):
        bx = x[:, start : start + batch_size]
        by = y[start : start + batch_size]
        correct += (student(bx)[:, :, :10].argmax(-1) == by).float().sum(1)
        student_h = hidden_activations(student, bx)
        teacher_h = hidden_activations(teacher, bx)
        sh_final, th_final = student_h[-1], teacher_h[-1]
        final_cosine_sum += nn.functional.cosine_similarity(sh_final, th_final, dim=-1).sum(1)
        final_mse_sum += ((sh_final - th_final) ** 2).mean(-1).sum(1)

        comparable = (
            len(student_h) > 1
            and len(teacher_h) > 1
            and student_h[-2].shape[-1] == teacher_h[-2].shape[-1]
        )
        has_comparable_internal = has_comparable_internal or comparable
        if comparable:
            internal_cosine_sum += nn.functional.cosine_similarity(
                student_h[-2], teacher_h[-2], dim=-1
            ).sum(1)
            internal_mse_sum += ((student_h[-2] - teacher_h[-2]) ** 2).mean(-1).sum(1)

    out = {"accuracy": (correct / n_items).detach().cpu().numpy()}
    if has_comparable_internal:
        out["hidden1_cosine"] = (internal_cosine_sum / n_items).detach().cpu().numpy()
        out["hidden1_mse"] = (internal_mse_sum / n_items).detach().cpu().numpy()
    else:
        out["hidden1_cosine"] = np.full(n_models, np.nan)
        out["hidden1_mse"] = np.full(n_models, np.nan)
    out["hidden2_cosine"] = (final_cosine_sum / n_items).detach().cpu().numpy()
    out["hidden2_mse"] = (final_mse_sum / n_items).detach().cpu().numpy()
    return out

@t.inference_mode()
def weight_metrics(student, teacher, ghost_idx):
    rows = {}
    student_layers = multi_linear_layers(student)
    teacher_layers = multi_linear_layers(teacher)
    for i, (sl, tl) in enumerate(zip(student_layers, teacher_layers), start=1):
        w_delta = sl.weight - tl.weight
        b_delta = sl.bias - tl.bias
        rows[f"layer{i}_weight_l2_mean"] = float(w_delta.flatten(1).norm(dim=1).mean().cpu())
        rows[f"layer{i}_bias_l2_mean"] = float(b_delta.norm(dim=1).mean().cpu())
        rows[f"layer{i}_weight_rel_l2_mean"] = float((w_delta.flatten(1).norm(dim=1) / tl.weight.flatten(1).norm(dim=1).clamp_min(1e-12)).mean().cpu())
        rows[f"layer{i}_weight_cosine_mean"] = float(nn.functional.cosine_similarity(sl.weight.flatten(1), tl.weight.flatten(1), dim=1).mean().cpu())

    for name, idx in [("class", CLASS_IDX), ("ghost", ghost_idx)]:
        sw, sb = readout_slice(student, idx)
        tw, tb = readout_slice(teacher, idx)
        w_delta = sw - tw
        b_delta = sb - tb
        rows[f"{name}_readout_weight_l2_mean"] = float(w_delta.flatten(1).norm(dim=1).mean().cpu())
        rows[f"{name}_readout_bias_l2_mean"] = float(b_delta.norm(dim=1).mean().cpu())
        rows[f"{name}_readout_weight_rel_l2_mean"] = float((w_delta.flatten(1).norm(dim=1) / tw.flatten(1).norm(dim=1).clamp_min(1e-12)).mean().cpu())
        rows[f"{name}_readout_weight_cosine_mean"] = float(nn.functional.cosine_similarity(sw.flatten(1), tw.flatten(1), dim=1).mean().cpu())
    return rows


def append_eval(rows, epoch, student, teacher, test_x, test_y, eval_batch_size, run_cfg):
    row = {"epoch": epoch, **run_cfg}
    metrics = activation_metrics(student, teacher, test_x, test_y, eval_batch_size)
    for name, values in metrics.items():
        row[f"{name}_mean"], row[f"{name}_ci95"] = mean_ci(values)
    row.update(weight_metrics(student, teacher, run_cfg["ghost_indices"]))
    rows.append(row)
    print(
        f"epoch {epoch:03d} acc={row['accuracy_mean']:.4f} "
        f"h1_cos={row['hidden1_cosine_mean']:.4f} "
        f"h2_cos={row['hidden2_cosine_mean']:.4f} "
        f"layer1_l2={row.get('layer1_weight_l2_mean', float('nan')):.3f}",
        flush=True,
    )
    return row


@t.inference_mode()
def spectrum_for_model(model, x, max_items, epoch, role):
    sx = x[:, :max_items]
    h = hidden_activations(model, sx)[-1]
    h = h - h.mean(dim=1, keepdim=True)
    singular_values = t.linalg.svdvals(h).detach().cpu().numpy()
    variance = singular_values ** 2
    denom = np.maximum(variance.sum(axis=1, keepdims=True), 1e-12)
    frac = variance / denom
    cumfrac = np.cumsum(frac, axis=1)

    rows = []
    for i in range(singular_values.shape[1]):
        s_mean, s_ci = mean_ci(singular_values[:, i])
        f_mean, f_ci = mean_ci(frac[:, i])
        c_mean, c_ci = mean_ci(cumfrac[:, i])
        rows.append(
            {
                "epoch": epoch,
                "role": role,
                "singular_index": i + 1,
                "singular_value_mean": s_mean,
                "singular_value_ci95": s_ci,
                "variance_fraction_mean": f_mean,
                "variance_fraction_ci95": f_ci,
                "cumulative_variance_mean": c_mean,
                "cumulative_variance_ci95": c_ci,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="MNIST grid with latent alignment diagnostics.")
    parser.add_argument("--num-ghost-logits", type=int, choices=GHOST_COUNTS, required=True)
    parser.add_argument("--data-fraction", type=float, choices=DATA_FRACTIONS, required=True)
    parser.add_argument("--condition", choices=CONDITIONS + ["all"], default="all")
    parser.add_argument("--teacher-readout", choices=TEACHER_READOUTS, required=True)
    parser.add_argument("--distill-epochs", type=int, default=100)
    parser.add_argument("--teacher-epochs", type=int, default=EPOCHS_TEACHER)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--spectrum-items", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, default=None, help="Directory containing shared teacher artifacts. Defaults to --out-dir.")
    parser.add_argument(
        "--student-init",
        choices=STUDENT_INITS,
        default="all_shared_init",
        help=(
            "all_shared_init uses the same seed for teacher and student; "
            "none_shared_init uses different seeds; "
            "last_shared_init shares only the final-layer initialization; "
            "last_shared_inherit copies the trained teacher final readout; "
            "lower_interp_* shares final-layer initialization and interpolates only lower-layer initialization; "
            "readout_interp_* shares non-final initialization and interpolates only the final readout; "
            "cnn_last_inherit uses a CNN student with a 256-dimensional latent and copies the trained teacher readout."
        ),
    )
    parser.add_argument("--wandb", action="store_true", help="Log metrics and final student checkpoint to Weights & Biases.")
    parser.add_argument("--no-save-student", action="store_true", help="Do not write final_student.pt checkpoints.")
    parser.add_argument(
        "--student-checkpoint-root",
        type=Path,
        default=Path(os.environ.get("MNIST_STUDENT_CHECKPOINT_ROOT", "/ceph/ssd/students/limbach/backup_storage/master_thesis_2/mnist_student_checkpoints")),
        help="SSD root for final_student.pt checkpoints. Run directories keep symlinks to this mirrored tree.",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "subliminal-learning-mnist"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args()

    t.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_label = f"{args.data_fraction:g}"
    teacher_root = args.teacher_root if args.teacher_root is not None else args.out_dir
    teacher_dir = teacher_root / TEACHER_DIR_NAMES[args.teacher_readout] / "teacher_artifacts"
    teacher_model_path = teacher_dir / "model.pt"
    teacher_spectra_path = teacher_dir / "spectra.csv"

    conditions_to_run = CONDITIONS if args.condition == "all" else [args.condition]
    expected_metrics = [
        args.out_dir / TEACHER_DIR_NAMES[args.teacher_readout] / CONDITION_DIR_NAMES[condition] / f"data{data_label}" / f"logits{args.num_ghost_logits}" / "metrics.csv"
        for condition in conditions_to_run
    ]
    expected_run_dirs = [path.parent for path in expected_metrics]
    expected_students_ready = args.no_save_student or all(final_student_link_ready(path, args.student_checkpoint_root) for path in expected_run_dirs)
    if teacher_spectra_path.exists() and all(path.exists() for path in expected_metrics) and expected_students_ready:
        print(f"skipping existing teacher/data/logits cell teacher={args.teacher_readout}, data={data_label}, logits={args.num_ghost_logits}")
        return
    if not teacher_model_path.exists():
        raise FileNotFoundError(f"Missing shared teacher checkpoint: {teacher_model_path}. Run prepare_mnist_readout_reinit_teachers.py first.")

    train_ds, test_ds = get_mnist()
    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    n_distill = max(1, int(round(train_x.shape[1] * args.data_fraction)))
    t.manual_seed(args.seed + 17)
    rand_imgs = t.rand((N_MODELS, n_distill, 1, 28, 28), device=DEVICE) * 2 - 1

    ghost_idx = list(range(10, 10 + args.num_ghost_logits))
    layer_sizes = [28 * 28, 256, 256, 10 + MAX_GHOST_LOGITS]

    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher_payload = t.load(teacher_model_path, map_location=DEVICE)
    teacher.load_state_dict(teacher_payload["state_dict"])
    teacher.eval()

    for condition in conditions_to_run:
        run_dir = args.out_dir / TEACHER_DIR_NAMES[args.teacher_readout] / CONDITION_DIR_NAMES[condition] / f"data{data_label}" / f"logits{args.num_ghost_logits}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.csv"
        meta_path = run_dir / "meta.json"
        if metrics_path.exists() and (args.no_save_student or final_student_link_ready(run_dir, args.student_checkpoint_root)):
            print(f"skipping existing student output {run_dir}")
            continue
        if metrics_path.exists() and not args.no_save_student:
            print(f"rerunning {run_dir} because metrics exist but final_student.pt is missing or stale")

        teacher_init_seed = args.seed
        student_seed = args.seed if args.student_init == "all_shared_init" else args.seed + 101
        t.manual_seed(student_seed)
        if args.student_init == "cnn_last_inherit":
            student = CNNStudent(N_MODELS, layer_sizes[-1]).to(DEVICE)
        else:
            student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        lower_interpolation_alpha = LOWER_LAYER_INTERPOLATION.get(args.student_init)
        readout_interpolation_alpha = FINAL_READOUT_INTERPOLATION.get(args.student_init)
        if args.student_init in {"last_shared_inherit", "cnn_last_inherit"}:
            copy_final_readout(student, teacher)
        elif args.student_init == "last_shared_init" or lower_interpolation_alpha is not None:
            t.manual_seed(teacher_init_seed)
            teacher_init = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
            copy_final_readout(student, teacher_init)
            if lower_interpolation_alpha is not None:
                interpolate_nonfinal_layers(student, teacher_init, lower_interpolation_alpha)
        elif readout_interpolation_alpha is not None:
            t.manual_seed(teacher_init_seed)
            teacher_init = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
            initialize_shared_nonfinal_and_interpolated_readout(
                student, teacher_init, readout_interpolation_alpha
            )
        opt = t.optim.Adam(student.parameters(), lr=LR)

        projection_basis = projection_mask = projection_rank = None
        if condition == "projected":
            projection_basis, projection_mask, projection_rank = projection_basis_from_readout(teacher, ghost_idx)

        run_name = f"{args.student_init}_teacher{args.teacher_readout}_cond{condition}_ghost{args.num_ghost_logits}_data{data_label}"
        run_cfg = {
            "run_name": run_name,
            "setup": args.student_init,
            "student_init": args.student_init,
            "teacher_root": str(teacher_root),
            "teacher_readout": args.teacher_readout,
            "teacher_readout_frozen": args.teacher_readout == "frozen",
            "condition": condition,
            "num_ghost_logits": args.num_ghost_logits,
            "data_fraction": args.data_fraction,
            "n_distill_points": n_distill,
            "objective": "projected_latent_mse" if condition == "projected" else "logit_kl",
            "freeze_readout": condition == "frozen",
            "projection_rank_mean": float(projection_rank.float().mean().cpu()) if projection_rank is not None else float("nan"),
            "ghost_indices": ghost_idx,
            "seed": args.seed,
            "student_seed": student_seed,
            "student_architecture": "cnn" if args.student_init == "cnn_last_inherit" else "mlp",
            "shares_teacher_initialization_seed": args.student_init == "all_shared_init",
            "lower_layer_interpolation_alpha": float(lower_interpolation_alpha) if lower_interpolation_alpha is not None else float("nan"),
            "final_readout_interpolation_alpha": float(readout_interpolation_alpha) if readout_interpolation_alpha is not None else float("nan"),
        }

        rows = []
        wandb_run = maybe_start_wandb(args, run_cfg, run_dir)
        try:
            row = append_eval(rows, 0, student, teacher, test_x, test_y, args.eval_batch_size, run_cfg)
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in row.items() if k != "ghost_indices"}, step=0)
            for epoch in tqdm.trange(1, args.distill_epochs + 1, desc=run_name):
                distill_one_epoch(
                    student,
                    teacher,
                    opt,
                    ghost_idx,
                    rand_imgs,
                    condition,
                    projection_basis,
                    projection_mask,
                    args.seed + epoch,
                )
                row = append_eval(rows, epoch, student, teacher, test_x, test_y, args.eval_batch_size, run_cfg)
                if wandb_run is not None:
                    wandb_run.log({k: v for k, v in row.items() if k != "ghost_indices"}, step=epoch)

            metrics_df = pd.DataFrame(rows).drop(columns=["ghost_indices"])
            metrics_df.to_csv(metrics_path, index=False)
            meta = {k: v for k, v in run_cfg.items() if k != "ghost_indices"}
            meta["saved_final_student"] = not args.no_save_student
            meta["student_checkpoint_root"] = str(args.student_checkpoint_root) if args.student_checkpoint_root is not None else None
            student_path = run_dir / "final_student.pt"
            checkpoint_path = None
            if not args.no_save_student:
                student_path, checkpoint_path = save_final_student_checkpoint(
                    student,
                    meta,
                    ghost_idx,
                    args.distill_epochs,
                    run_dir,
                    args.student_checkpoint_root,
                )
                meta["student_checkpoint_path"] = str(checkpoint_path)
                meta["student_checkpoint_symlink"] = str(student_path)
            meta_path.write_text(json.dumps(meta, indent=2))
            if wandb_run is not None:
                wandb_run.save(str(metrics_path))
                wandb_run.save(str(meta_path))
                if not args.no_save_student:
                    wandb_run.save(str(student_path))
            print(f"wrote {metrics_path}")
            if not args.no_save_student:
                print(f"wrote {checkpoint_path}")
                if checkpoint_path != student_path:
                    print(f"linked {student_path} -> {checkpoint_path}")
            else:
                print("skipped final_student.pt")
            print(f"wrote {meta_path}")
        finally:
            if wandb_run is not None:
                wandb_run.finish()


if __name__ == "__main__":
    main()
