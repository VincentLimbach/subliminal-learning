import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_final_rows(results_dir):
    rows = []
    for csv_path in sorted(results_dir.glob("setup*_ghost*_data*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        row = df.loc[df["epoch"].idxmax()].copy()
        row["source_file"] = csv_path.name
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No result CSV files found in {results_dir}")
    return pd.DataFrame(rows)


def plot_setup_accuracy(summary, setup, out_path):
    part = summary[summary["setup"] == setup]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    for ax, frac in zip(axes, sorted(part["data_fraction"].unique())):
        sub = part[part["data_fraction"] == frac]
        for freeze, label, marker in [(False, "student ghost readout trainable", "o"), (True, "student ghost readout frozen", "s")]:
            p = sub[sub["freeze_readout"] == freeze].sort_values("num_ghost_logits")
            if p.empty:
                continue
            ax.plot(p["num_ghost_logits"], p["accuracy_mean"], marker=marker, linewidth=2, label=label)
        ax.set_title(f"distill data={frac:g}")
        ax.set_xlabel("unrelated / ghost logits")
        ax.set_xticks(sorted(part["num_ghost_logits"].unique()))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("final MNIST accuracy")
    axes[0].legend(loc="best")
    fig.suptitle(setup)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize MNIST aux-only grid outputs.")
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    summary = load_final_rows(args.results_dir)
    summary = summary.sort_values(["setup", "num_ghost_logits", "data_fraction", "freeze_readout"])
    out_csv = args.results_dir / "summary_final.csv"
    summary.to_csv(out_csv, index=False)

    cols = [
        "setup",
        "num_ghost_logits",
        "data_fraction",
        "freeze_readout",
        "accuracy_mean",
        "accuracy_ci95",
        "class_readout_cosine_mean",
        "latent_cosine_mean",
        "student_ghost_readout_weight_max_abs_drift",
        "teacher_class_readout_weight_max_abs_drift",
    ]
    accuracy_table = summary[cols]
    print(accuracy_table.to_string(index=False))

    pivot = summary.pivot_table(
        index=["setup", "num_ghost_logits", "data_fraction"],
        columns="freeze_readout",
        values="accuracy_mean",
    ).rename(columns={False: "nonfrozen_accuracy", True: "frozen_accuracy"})
    if "frozen_accuracy" in pivot and "nonfrozen_accuracy" in pivot:
        pivot["frozen_minus_nonfrozen"] = pivot["frozen_accuracy"] - pivot["nonfrozen_accuracy"]
    pivot_path = args.results_dir / "summary_accuracy_pivot.csv"
    pivot.to_csv(pivot_path)

    print("\nAccuracy pivot:")
    print(pivot.to_string())

    for setup in sorted(summary["setup"].unique()):
        out_path = args.results_dir / f"summary_accuracy_{setup}.png"
        plot_setup_accuracy(summary, setup, out_path)
        print(f"wrote {out_path}")

    frozen = summary[summary["freeze_readout"]]
    if not frozen.empty:
        print("\nFrozen student ghost-readout drift maxima:")
        print(frozen[["source_file", "student_ghost_readout_weight_max_abs_drift", "student_ghost_readout_bias_max_abs_drift"]].to_string(index=False))

    frozen_teacher = summary[summary["setup"] == "frozen_class_readout"]
    if not frozen_teacher.empty:
        print("\nFrozen teacher class-readout drift maxima:")
        print(frozen_teacher[["source_file", "teacher_class_readout_weight_max_abs_drift", "teacher_class_readout_bias_max_abs_drift"]].to_string(index=False))

    print(f"wrote {out_csv}")
    print(f"wrote {pivot_path}")


if __name__ == "__main__":
    main()
