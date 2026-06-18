import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch as t

from run_mnist_experiment import BATCH_SIZE, DEVICE, MultiClassifier, get_mnist
from run_mnist_readout_reinit_grid_job import MAX_GHOST_LOGITS, TEACHER_DIR_NAMES, to_tensor

GHOST_COLORS = {2: "C0", 3: "C1", 4: "C2", 6: "C3", 8: "C4", 12: "C5", 16: "C6", 24: "C7", 32: "C8", 48: "C9", 64: "C10", 96: "C11", 128: "C0", 192: "C1", 256: "C2", 384: "C3", 512: "C4", 768: "C5", 1024: "C6"}
DATA_STYLES = {0.1: ("o", "-", "C0"), 0.5: ("s", "--", "C1"), 1.0: ("^", "-.", "C2")}
TEACHER_ORDER = ["nonfrozen", "frozen"]
STUDENT_ORDER = ["nonfrozen", "frozen", "projected"]
TEACHER_LABELS = {"nonfrozen": "Teacher class A trainable", "frozen": "Teacher class A frozen"}
STUDENT_LABELS = {"nonfrozen": "Student class B logits trainable", "frozen": "Student class B logits frozen", "projected": "Projected latent"}


def load_metrics(results_dir):
    frames = []
    for path in sorted(results_dir.glob("finetuning_A_readouts_*/*/data*/logits*/metrics.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_file"] = str(path.relative_to(results_dir))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No nested metrics CSVs found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def load_spectra(results_dir):
    frames = []
    for path in sorted(results_dir.glob("finetuning_A_readouts_*/teacher_artifacts/spectra.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_file"] = str(path.relative_to(results_dir))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@t.inference_mode()
def teacher_accuracy_by_readout(results_dir):
    _, test_ds = get_mnist()
    test_x_s, test_y = to_tensor(test_ds)
    out = {}
    for teacher in TEACHER_ORDER:
        model_path = results_dir / TEACHER_DIR_NAMES[teacher] / "teacher_artifacts" / "model.pt"
        if not model_path.exists():
            continue
        payload = t.load(model_path, map_location=DEVICE)
        first_weight = next(v for k, v in payload["state_dict"].items() if k.endswith("weight"))
        n_models = int(first_weight.shape[0])
        model = MultiClassifier(n_models, [28 * 28, 256, 256, 10 + MAX_GHOST_LOGITS]).to(DEVICE)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        test_x = test_x_s.unsqueeze(0).expand(n_models, -1, -1, -1, -1)
        correct = t.zeros(n_models, device=DEVICE)
        for start in range(0, test_x.shape[1], BATCH_SIZE):
            bx = test_x[:, start : start + BATCH_SIZE]
            by = test_y[start : start + BATCH_SIZE]
            correct += (model(bx)[:, :, :10].argmax(-1) == by).float().sum(1)
        out[teacher] = float((correct / test_x.shape[1]).mean().cpu())
    return out


def final_summary(metrics):
    idx = metrics.groupby("source_file")["epoch"].idxmax()
    return metrics.loc[idx].copy().sort_values(["teacher_readout", "condition", "data_fraction", "num_ghost_logits"])


def plot_2x3_logits(summary, value, ylabel, title, out_path, ylim=None, teacher_accuracy=None):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2), sharex=True, sharey=ylim is not None)
    for r, teacher in enumerate(TEACHER_ORDER):
        for c, student in enumerate(STUDENT_ORDER):
            ax = axes[r][c]
            part = summary[(summary["teacher_readout"] == teacher) & (summary["condition"] == student)]
            for frac in sorted(part["data_fraction"].unique()):
                p = part[part["data_fraction"] == frac].sort_values("num_ghost_logits")
                marker, linestyle, line_color = DATA_STYLES.get(float(frac), ("o", "-", "0.15"))
                ax.plot(
                    p["num_ghost_logits"],
                    p[value],
                    color=line_color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.2,
                    markersize=5.5,
                    label=f"data={frac:g}",
                )
            if value == "accuracy_mean":
                ax.axhline(0.10, color="0.2", linestyle=":", linewidth=1.4, label="chance 10%" if r == 0 and c == 0 else None, zorder=1)
                if teacher_accuracy and teacher in teacher_accuracy:
                    ax.axhline(
                        teacher_accuracy[teacher],
                        color="0.05",
                        linestyle="--",
                        linewidth=1.8,
                        label="teacher" if c == 0 else None,
                        zorder=1,
                    )
            ax.set_xscale("log", base=2)
            ticks = [g for g in sorted(summary["num_ghost_logits"].unique()) if g <= 1024 and (g in {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024})]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(g)) for g in ticks])
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(STUDENT_LABELS[student])
            if c == 0:
                ax.set_ylabel(f"{TEACHER_LABELS[teacher]}\n{ylabel}")
            ax.set_xlabel("ghost logits")
    legend_items = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label:
                legend_items.setdefault(label, handle)
    if legend_items:
        fig.legend(list(legend_items.values()), list(legend_items.keys()), loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=min(4, len(legend_items)), fontsize=9, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_epoch_curves(metrics, value, ylabel, out_path, ylim=None):
    for teacher in sorted(metrics["teacher_readout"].unique()):
        for student in sorted(metrics["condition"].unique()):
            part = metrics[(metrics["teacher_readout"] == teacher) & (metrics["condition"] == student)]
            if part.empty:
                continue
            fracs = sorted(part["data_fraction"].unique())
            fig, axes = plt.subplots(1, len(fracs), figsize=(4.8 * len(fracs), 3.6), sharey=True, squeeze=False)
            for ax, frac in zip(axes[0], fracs):
                frac_part = part[part["data_fraction"] == frac]
                for ghost, run in frac_part.groupby("num_ghost_logits"):
                    run = run.sort_values("epoch")
                    ax.plot(run["epoch"], run[value], color=GHOST_COLORS[int(ghost)], linewidth=1.6, label=f"g={int(ghost)}")
                ax.set_title(f"data={frac:g}")
                ax.set_xlabel("epoch")
                ax.grid(alpha=0.25)
                if ylim is not None:
                    ax.set_ylim(*ylim)
            axes[0][0].set_ylabel(ylabel)
            handles, labels = axes[0][-1].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=min(8, len(handles)), fontsize=8)
            fig.suptitle(f"{TEACHER_LABELS[teacher]} / {STUDENT_LABELS[student]}")
            fig.tight_layout(rect=(0, 0, 1, 0.88))
            fig.savefig(out_path.with_name(f"{out_path.stem}_teacher{teacher}_student{student}{out_path.suffix}"), dpi=180)
            plt.close(fig)


def plot_leaf_epoch_curves(metrics, results_dir):
    diagnostics = [
        ("hidden2_cosine_mean", "final latent cosine", (0, 1)),
        ("hidden1_cosine_mean", "hidden1 cosine", (0, 1)),
        ("class_readout_weight_cosine_mean", "class readout cosine", (0.9, 1.1)),
        ("ghost_readout_weight_cosine_mean", "ghost readout cosine", (0.9, 1.1)),
    ]
    for source_file, run in metrics.groupby("source_file"):
        run = run.sort_values("epoch")
        run_dir = results_dir / Path(source_file).parent
        run_dir.mkdir(parents=True, exist_ok=True)

        teacher = str(run["teacher_readout"].iloc[0])
        condition = str(run["condition"].iloc[0])
        frac = float(run["data_fraction"].iloc[0])
        ghost = int(run["num_ghost_logits"].iloc[0])
        title = f"{TEACHER_LABELS.get(teacher, teacher)} / {STUDENT_LABELS.get(condition, condition)} / data={frac:g} / g={ghost}"

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.plot(run["epoch"], run["accuracy_mean"], color=GHOST_COLORS.get(ghost, "C0"), linewidth=1.9)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(run_dir / "viz_epochs_accuracy.png", dpi=180)
        plt.close(fig)

        available = [(col, label, ylim) for col, label, ylim in diagnostics if col in run.columns]
        if not available:
            continue
        fig, axes = plt.subplots(len(available), 1, figsize=(7.2, 2.3 * len(available)), sharex=True, squeeze=False)
        for ax, (col, label, ylim) in zip(axes[:, 0], available):
            ax.plot(run["epoch"], run[col], color=GHOST_COLORS.get(ghost, "C0"), linewidth=1.6)
            ax.set_ylabel(label)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)
        axes[-1, 0].set_xlabel("epoch")
        fig.suptitle(title, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(run_dir / "viz_epochs_alignment.png", dpi=180)
        plt.close(fig)


def plot_spectra(spectra, role, out_sv_path, out_cum_path):
    part = spectra[spectra["role"] == role]
    if part.empty:
        return
    grouped = part.groupby(["teacher_readout", "singular_index"], as_index=False).agg(
        singular_value_mean=("singular_value_mean", "mean"),
        cumulative_variance_mean=("cumulative_variance_mean", "mean"),
    )

    colors = {"nonfrozen": "C0", "frozen": "C1"}
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for teacher, run in grouped.groupby("teacher_readout"):
        run = run.sort_values("singular_index")
        ax.plot(run["singular_index"], run["singular_value_mean"], color=colors.get(teacher), linewidth=1.9, label=f"teacher {teacher}")
    ax.set_title(f"{role.capitalize()} final-latent singular values")
    ax.set_xlabel("singular vector index")
    ax.set_ylabel("singular value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_sv_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for teacher, run in grouped.groupby("teacher_readout"):
        run = run.sort_values("singular_index")
        ax.plot(run["singular_index"], run["cumulative_variance_mean"], color=colors.get(teacher), linewidth=1.9, label=f"teacher {teacher}")
    ax.set_title(f"{role.capitalize()} cumulative variance explained")
    ax.set_xlabel("first n singular vectors")
    ax.set_ylabel("cumulative variance explained")
    ax.set_ylim(0.0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_cum_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    metrics = load_metrics(args.results_dir)
    spectra = load_spectra(args.results_dir)
    summary = final_summary(metrics)
    teacher_accuracy = teacher_accuracy_by_readout(args.results_dir)

    summary.to_csv(args.results_dir / "summary_metrics_final.csv", index=False)
    if not spectra.empty:
        spectra.to_csv(args.results_dir / "summary_spectra.csv", index=False)

    plot_2x3_logits(summary, "accuracy_mean", "final accuracy", "Final accuracy by logits", args.results_dir / "viz_2x3_final_accuracy.png", ylim=(0, 1), teacher_accuracy=teacher_accuracy)
    plot_2x3_logits(summary, "hidden2_cosine_mean", "final latent cosine", "Final latent cosine by logits", args.results_dir / "viz_2x3_final_latent_cosine.png", ylim=(0, 1))
    plot_2x3_logits(summary, "hidden1_cosine_mean", "hidden1 cosine", "Hidden1 cosine by logits", args.results_dir / "viz_2x3_hidden1_cosine.png", ylim=(0, 1))
    plot_2x3_logits(summary, "class_readout_weight_cosine_mean", "class readout cosine", "Class readout weight cosine by logits", args.results_dir / "viz_2x3_class_readout_cosine.png", ylim=(0.9, 1.1))
    plot_2x3_logits(summary, "ghost_readout_weight_cosine_mean", "ghost readout cosine", "Ghost readout weight cosine by logits", args.results_dir / "viz_2x3_ghost_readout_cosine.png", ylim=(0.9, 1.1))

    plot_leaf_epoch_curves(metrics, args.results_dir)

    if not spectra.empty:
        spectra_plot_dir = args.results_dir / "teacher_spectra"
        spectra_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_spectra(spectra, "teacher", spectra_plot_dir / "viz_teacher_singular_values.png", spectra_plot_dir / "viz_teacher_cumulative_variance.png")

    print(f"loaded_metric_runs={metrics['source_file'].nunique()}")
    print(f"loaded_spectrum_runs={spectra['source_file'].nunique() if not spectra.empty else 0}")
    print(f"wrote {args.results_dir / 'summary_metrics_final.csv'}")


if __name__ == "__main__":
    main()
