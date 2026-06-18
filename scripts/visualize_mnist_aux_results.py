import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SETUP_LABELS = {
    "base": "Base student",
    "teacher_epoch1": "Student = teacher after 1 epoch",
    "frozen_class_readout": "Teacher class readout frozen",
    "readout_only": "Shared final readout only",
}


GHOST_COLORS = {2: "C0", 3: "C1", 5: "C2", 10: "C3", 128: "C4", 256: "C5"}


def load_runs(results_dir):
    frames = []
    for path in sorted(results_dir.glob("setup*_ghost*_data*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No run CSVs found in {results_dir}")
    df = pd.concat(frames, ignore_index=True)
    if "objective" not in df.columns:
        df["objective"] = "logit_kl"
    else:
        df["objective"] = df["objective"].fillna("logit_kl")
    return df


def final_summary(df):
    idx = df.groupby("source_file")["epoch"].idxmax()
    return df.loc[idx].copy().sort_values([c for c in ["objective", "setup", "data_fraction", "num_ghost_logits", "freeze_readout"] if c in df.columns])


def plot_final_accuracy(summary, out_path):
    setups = [s for s in SETUP_LABELS if s in set(summary["setup"])]
    fracs = sorted(summary["data_fraction"].unique())
    fig, axes = plt.subplots(len(setups), len(fracs), figsize=(4.4 * len(fracs), 3.4 * len(setups)), sharey=True, squeeze=False)
    for r, setup in enumerate(setups):
        setup_df = summary[summary["setup"] == setup]
        for c, frac in enumerate(fracs):
            ax = axes[r][c]
            part = setup_df[setup_df["data_fraction"] == frac]
            for freeze, label, marker, linestyle in [(False, "nonfrozen", "o", "-"), (True, "frozen", "s", "--")]:
                p = part[part["freeze_readout"] == freeze].sort_values("num_ghost_logits")
                if not p.empty:
                    for _, row in p.iterrows():
                        g = int(row["num_ghost_logits"])
                        ax.plot(
                            [g],
                            [row["accuracy_mean"]],
                            marker=marker,
                            linewidth=2,
                            linestyle=linestyle,
                            color=GHOST_COLORS.get(g, None),
                            label=label if g == p.iloc[0]["num_ghost_logits"] else None,
                        )
            ax.set_title(f"{SETUP_LABELS[setup]}\ndata={frac:g}")
            ax.set_xlabel("ghost logits")
            ax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel("final accuracy")
            ax.set_xticks(sorted(summary["num_ghost_logits"].unique()))
            ax.set_ylim(0.0, 1.0)
            if r == 0 and c == len(fracs) - 1:
                ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_freeze_delta(summary, out_path):
    pivot = summary.pivot_table(
        index=["setup", "data_fraction", "num_ghost_logits"],
        columns="freeze_readout",
        values="accuracy_mean",
    ).reset_index()
    if False not in pivot or True not in pivot:
        return
    pivot["delta"] = pivot[True] - pivot[False]

    setups = [s for s in SETUP_LABELS if s in set(pivot["setup"])]
    fracs = sorted(pivot["data_fraction"].unique())
    ghosts = sorted(pivot["num_ghost_logits"].unique())
    vmax = max(abs(pivot["delta"].min()), abs(pivot["delta"].max()), 1e-6)

    fig, axes = plt.subplots(1, len(setups), figsize=(4.2 * len(setups), 3.8), squeeze=False)
    for ax, setup in zip(axes[0], setups):
        mat = []
        for frac in fracs:
            row = []
            for ghost in ghosts:
                val = pivot[(pivot["setup"] == setup) & (pivot["data_fraction"] == frac) & (pivot["num_ghost_logits"] == ghost)]["delta"]
                row.append(float(val.iloc[0]) if not val.empty else float("nan"))
            mat.append(row)
        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(SETUP_LABELS[setup])
        ax.set_xticks(range(len(ghosts)), ghosts)
        ax.set_yticks(range(len(fracs)), [f"{f:g}" for f in fracs])
        ax.set_xlabel("ghost logits")
        ax.set_ylabel("data fraction")
        for y, row in enumerate(mat):
            for x, val in enumerate(row):
                if pd.notna(val):
                    ax.text(x, y, f"{val:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="frozen - nonfrozen accuracy")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_curves(df, out_path):
    setups = [s for s in SETUP_LABELS if s in set(df["setup"])]
    fig, axes = plt.subplots(len(setups), 3, figsize=(13.5, 3.4 * len(setups)), sharey=True, squeeze=False)
    linestyles = {False: "-", True: "--"}
    for r, setup in enumerate(setups):
        setup_df = df[df["setup"] == setup]
        for c, frac in enumerate(sorted(setup_df["data_fraction"].unique())):
            ax = axes[r][c]
            part = setup_df[setup_df["data_fraction"] == frac]
            for (ghost, freeze), p in part.groupby(["num_ghost_logits", "freeze_readout"]):
                p = p.sort_values("epoch")
                label = f"g={ghost}, {'frozen' if freeze else 'nonfrozen'}"
                ax.plot(p["epoch"], p["accuracy_mean"], color=GHOST_COLORS.get(int(ghost), None), linestyle=linestyles[bool(freeze)], linewidth=1.6, label=label)
            ax.set_title(f"{SETUP_LABELS[setup]}\ndata={frac:g}")
            ax.set_xlabel("distillation epoch")
            if c == 0:
                ax.set_ylabel("accuracy")
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.25)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metric_grid(summary, out_path, metrics, ylabels, title_prefix):
    setups = [s for s in SETUP_LABELS if s in set(summary["setup"])]
    fracs = sorted(summary["data_fraction"].unique())
    freezes = [(False, "nonfrozen", "-"), (True, "frozen", "--")]
    fig, axes = plt.subplots(
        len(metrics),
        len(setups),
        figsize=(4.6 * len(setups), 2.8 * len(metrics)),
        sharex=True,
        squeeze=False,
    )
    ghosts = sorted(summary["num_ghost_logits"].unique())

    for col, setup in enumerate(setups):
        setup_df = summary[summary["setup"] == setup]
        for row, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
            ax = axes[row][col]
            for freeze, freeze_label, linestyle in freezes:
                part = setup_df[setup_df["freeze_readout"] == freeze]
                if part.empty:
                    continue
                for frac in fracs:
                    p = (
                        part[part["data_fraction"] == frac]
                        .groupby("num_ghost_logits", as_index=False)[metric]
                        .mean()
                        .sort_values("num_ghost_logits")
                    )
                    if p.empty:
                        continue
                    label = f"{freeze_label}, data={frac:g}" if row == 0 and col == len(setups) - 1 else None
                    for _, r in p.iterrows():
                        g = int(r["num_ghost_logits"])
                        ax.plot(
                            [g],
                            [r[metric]],
                            marker="o",
                            linewidth=1.8,
                            linestyle=linestyle,
                            color=GHOST_COLORS.get(g, None),
                            label=label if g == int(p.iloc[0]["num_ghost_logits"]) else None,
                        )
            if row == 0:
                ax.set_title(f"{title_prefix}: {SETUP_LABELS[setup]}")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlabel("ghost logits")
            ax.set_xticks(ghosts)
            ax.grid(alpha=0.25)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_readout_alignment(summary, out_path):
    _plot_metric_grid(
        summary,
        out_path,
        metrics=["class_readout_cosine_mean", "ghost_readout_cosine_mean"],
        ylabels=["class readout cosine", "ghost readout cosine"],
        title_prefix="Readout alignment",
    )


def plot_readout_drift(summary, out_path):
    _plot_metric_grid(
        summary,
        out_path,
        metrics=[
            "student_ghost_readout_weight_max_abs_drift",
            "teacher_class_readout_weight_max_abs_drift",
        ],
        ylabels=["student ghost drift", "teacher class drift"],
        title_prefix="Readout drift",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    df = load_runs(args.results_dir)
    summary = final_summary(df)
    summary_path = args.results_dir / "summary_final.csv"
    pivot_path = args.results_dir / "summary_accuracy_pivot.csv"
    summary.to_csv(summary_path, index=False)
    pivot = summary.pivot_table(index=["setup", "num_ghost_logits", "data_fraction"], columns="freeze_readout", values="accuracy_mean").rename(columns={False: "nonfrozen_accuracy", True: "frozen_accuracy"})
    if "frozen_accuracy" in pivot and "nonfrozen_accuracy" in pivot:
        pivot["frozen_minus_nonfrozen"] = pivot["frozen_accuracy"] - pivot["nonfrozen_accuracy"]
    pivot.to_csv(pivot_path)

    outputs = {}
    for objective in sorted(summary["objective"].dropna().unique()):
        objective_summary = summary[summary["objective"] == objective]
        objective_df = df[df["objective"] == objective]
        objective_outputs = {
            "final_accuracy": args.results_dir / f"viz_{objective}_final_accuracy_by_setup.png",
            "freeze_delta": args.results_dir / f"viz_{objective}_freeze_delta_heatmap.png",
            "accuracy_curves": args.results_dir / f"viz_{objective}_accuracy_curves.png",
            "readout_alignment": args.results_dir / f"viz_{objective}_readout_alignment.png",
            "readout_drift": args.results_dir / f"viz_{objective}_readout_drift.png",
        }
        plot_final_accuracy(objective_summary, objective_outputs["final_accuracy"])
        plot_freeze_delta(objective_summary, objective_outputs["freeze_delta"])
        plot_accuracy_curves(objective_df, objective_outputs["accuracy_curves"])
        plot_readout_alignment(objective_summary, objective_outputs["readout_alignment"])
        plot_readout_drift(objective_summary, objective_outputs["readout_drift"])
        outputs.update(objective_outputs)

    print(f"loaded_runs={df['source_file'].nunique()}")
    print(f"wrote {summary_path}")
    print(f"wrote {pivot_path}")
    for path in outputs.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
