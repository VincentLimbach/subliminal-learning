import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_mnist_experiment import BATCH_SIZE, DEVICE, MultiClassifier, get_mnist
from run_mnist_readout_reinit_grid_job import MAX_GHOST_LOGITS, TEACHER_DIR_NAMES, TEACHER_READOUTS, hidden_activations, to_tensor


@t.inference_mode()
def activation_spectrum(model, x, batch_size):
    n_models, n_items = x.shape[:2]
    hidden_dim = 256
    count = 0
    sum_h = t.zeros(hidden_dim, dtype=t.float64, device=DEVICE)
    xtx = t.zeros((hidden_dim, hidden_dim), dtype=t.float64, device=DEVICE)

    for start in range(0, n_items, batch_size):
        bx = x[:, start:start + batch_size]
        h = hidden_activations(model, bx)[-1].reshape(-1, hidden_dim).to(t.float64)
        count += h.shape[0]
        sum_h += h.sum(dim=0)
        xtx += h.T @ h

    mean = sum_h / count
    centered_xtx = xtx - count * t.outer(mean, mean)
    centered_xtx = (centered_xtx + centered_xtx.T) / 2
    eigvals = t.linalg.eigvalsh(centered_xtx).clamp_min(0).flip(0)
    singular_values = t.sqrt(eigvals).detach().cpu().numpy()
    variance = singular_values ** 2
    variance_fraction = variance / max(float(variance.sum()), 1e-12)
    cumulative_variance = np.cumsum(variance_fraction)
    rank_99 = int(np.searchsorted(cumulative_variance, 0.99) + 1)
    return singular_values, variance_fraction, cumulative_variance, rank_99, count


def rank_at(run, target):
    return int(run.loc[run["cumulative_variance"] >= target, "singular_index"].iloc[0])


def plot_spectrum(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for teacher, run in df.groupby("teacher_readout"):
        run = run.sort_values("singular_index")
        rank_90 = rank_at(run, 0.90)
        rank_99 = rank_at(run, 0.99)
        ax.plot(
            run["singular_index"],
            run["singular_value"],
            linestyle="None",
            marker="o",
            markersize=3.2,
            alpha=0.82,
            label=f"{teacher} singular values",
        )
    ax.set_title("Teacher last-hidden activation spectrum")
    ax.set_xlabel("singular value rank")
    ax.set_ylabel("singular value")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    for teacher, run in df.groupby("teacher_readout"):
        run = run.sort_values("singular_index")
        rank_90 = rank_at(run, 0.90)
        rank_99 = rank_at(run, 0.99)
        ax2.plot(
            run["singular_index"],
            run["cumulative_variance"],
            linewidth=1.9,
            label=f"{teacher} cumulative variance (90% n={rank_90}, 99% n={rank_99})",
        )
    ax2.axhline(0.90, color="0.45", linestyle="--", linewidth=1.0, alpha=0.65, label="90% variance")
    ax2.axhline(0.99, color="0.25", linestyle=":", linewidth=1.0, alpha=0.75, label="99% variance")
    ax2.set_ylabel("cumulative variance explained")
    ax2.set_ylim(0, 1.01)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "viz_teacher_overall_activation_spectrum_overlay.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for teacher, run in df.groupby("teacher_readout"):
        run = run.sort_values("singular_index")
        ax.plot(run["singular_index"], run["singular_value"], linestyle="None", marker="o", markersize=3.0, alpha=0.82, label=teacher)
    ax.set_title("Teacher last-hidden activation singular values")
    ax.set_xlabel("singular value rank")
    ax.set_ylabel("singular value")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "viz_teacher_overall_activation_singular_values.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for teacher, run in df.groupby("teacher_readout"):
        run = run.sort_values("singular_index")
        rank_90 = rank_at(run, 0.90)
        rank_99 = rank_at(run, 0.99)
        ax.plot(run["singular_index"], run["cumulative_variance"], linewidth=1.9, label=f"{teacher} (90% n={rank_90}, 99% n={rank_99})")
    ax.axhline(0.90, color="0.45", linestyle="--", linewidth=1.0, label="90% variance")
    ax.axhline(0.99, color="0.25", linestyle=":", linewidth=1.0, label="99% variance")
    ax.set_title("Teacher last-hidden cumulative variance explained")
    ax.set_xlabel("first n singular values")
    ax.set_ylabel("cumulative variance explained")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "viz_teacher_overall_activation_cumulative_variance.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute overall teacher last-hidden activation spectra.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    _, test_ds = get_mnist()
    test_x_s, _ = to_tensor(test_ds)

    rows = []
    ranks = []
    for teacher in TEACHER_READOUTS:
        model_path = args.results_dir / TEACHER_DIR_NAMES[teacher] / "teacher_artifacts" / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        payload = t.load(model_path, map_location=DEVICE)
        first_weight = next(v for k, v in payload["state_dict"].items() if k.endswith("weight"))
        n_models = int(first_weight.shape[0])
        model = MultiClassifier(n_models, [28 * 28, 256, 256, 10 + MAX_GHOST_LOGITS]).to(DEVICE)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        test_x = test_x_s.unsqueeze(0).expand(n_models, -1, -1, -1, -1)

        singular_values, variance_fraction, cumulative_variance, rank_99, count = activation_spectrum(model, test_x, args.batch_size)
        ranks.append({"teacher_readout": teacher, "rank_99": rank_99, "activation_rows": count})
        for i, (sv, vf, cv) in enumerate(zip(singular_values, variance_fraction, cumulative_variance), start=1):
            rows.append({
                "teacher_readout": teacher,
                "singular_index": i,
                "singular_value": float(sv),
                "variance_fraction": float(vf),
                "cumulative_variance": float(cv),
                "rank_99": rank_99,
                "activation_rows": count,
            })

    out_dir = args.results_dir / "teacher_activation_spectrum"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "teacher_overall_activation_spectrum.csv", index=False)
    Path(out_dir / "teacher_overall_activation_rank99.json").write_text(json.dumps(ranks, indent=2))
    plot_spectrum(df, out_dir)

    for row in ranks:
        print(f"{row['teacher_readout']}: rank_99={row['rank_99']} activation_rows={row['activation_rows']}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
