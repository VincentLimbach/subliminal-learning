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

from run_mnist_experiment import DEVICE, MultiClassifier, get_mnist
from run_mnist_readout_reinit_grid_job import MAX_GHOST_LOGITS, to_tensor

GHOST_COUNTS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
SETUPS = [
    ('all_shared_init', 'All shared init'),
    ('none_shared_init', 'None shared init'),
    ('last_shared_init', 'Last shared init'),
    ('shared_last_inherit', 'Shared last inherit'),
]
TEACHER_DIR = 'finetuning_A_readouts_nonfrozen'
CONDITION_DIR = 'logit_distilation_B_readouts_nonfrozen'
DATA_LABEL = 'data1'
COLORS = {
    'teacher': '#8c6d62',
    'all_shared_init': '#8f6db8',
    'none_shared_init': '#9c9c9c',
    'last_shared_init': '#b8a2d9',
    'shared_last_inherit': '#6b4ea1',
}


def teacher_accuracy(runs_root: Path) -> float:
    model_path = runs_root / 'shared_last_inherit' / TEACHER_DIR / 'teacher_artifacts' / 'model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f'Missing teacher checkpoint: {model_path}')
    _, test_ds = get_mnist()
    test_x_s, test_y = to_tensor(test_ds)
    test_x = test_x_s.unsqueeze(0)
    model = MultiClassifier(1, [28 * 28, 256, 256, 10 + MAX_GHOST_LOGITS]).to(DEVICE)
    payload = t.load(model_path, map_location=DEVICE)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    with t.inference_mode():
        pred = model(test_x)[0, :, :10].argmax(-1)
        return float((pred == test_y).float().mean().cpu())


def read_final_accuracy(runs_root: Path, setup: str, ghost_count: int):
    path = runs_root / setup / TEACHER_DIR / CONDITION_DIR / DATA_LABEL / f'logits{ghost_count}' / 'metrics.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    last = df.iloc[-1]
    return {
        'accuracy': float(last['accuracy_mean']),
        'ci95': float(last['accuracy_ci95']) if 'accuracy_ci95' in last and pd.notna(last['accuracy_ci95']) else np.nan,
        'epoch': int(last['epoch']),
        'path': str(path),
    }


def plot_one(out_dir: Path, ghost_count: int, teacher_acc: float, rows):
    labels = ['Teacher'] + [label for _, label in SETUPS]
    values = [teacher_acc] + [rows[(setup, ghost_count)]['accuracy'] if rows.get((setup, ghost_count)) else np.nan for setup, _ in SETUPS]
    errors = [np.nan] + [rows[(setup, ghost_count)]['ci95'] if rows.get((setup, ghost_count)) else np.nan for setup, _ in SETUPS]
    keys = ['teacher'] + [setup for setup, _ in SETUPS]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(labels))
    heights = np.nan_to_num(values, nan=0.0)
    colors = [COLORS[k] for k in keys]
    bar_errors = np.array([0.0 if np.isnan(e) else e for e in errors])
    bars = ax.bar(x, heights, yerr=bar_errors, capsize=4, color=colors, edgecolor='none')

    for i, (bar, value) in enumerate(zip(bars, values)):
        if np.isnan(value):
            bar.set_facecolor('#eeeeee')
            bar.set_edgecolor('#999999')
            bar.set_hatch('//')
            ax.text(bar.get_x() + bar.get_width() / 2, 0.035, 'pending', ha='center', va='bottom', rotation=90, fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, min(value + 0.025, 0.985), f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    ax.axhline(0.10, color='black', linestyle=':', linewidth=1.8, label='chance')
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('Test accuracy')
    ax.set_title(f'Full data, nonfrozen logits, no projection, g={ghost_count}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / f'figure10b_full_data_nonfrozen_logits{ghost_count}.png', dpi=180)
    plt.close(fig)


def plot_summary(out_dir: Path, teacher_acc: float, rows):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for setup, label in SETUPS:
        xs, ys = [], []
        for g in GHOST_COUNTS:
            row = rows.get((setup, g))
            if row is not None:
                xs.append(g)
                ys.append(row['accuracy'])
        if xs:
            ax.plot(xs, ys, marker='o', linewidth=2, label=label, color=COLORS[setup])
    ax.axhline(teacher_acc, color=COLORS['teacher'], linewidth=2, linestyle='--', label=f'Teacher ({teacher_acc:.3f})')
    ax.axhline(0.10, color='black', linestyle=':', linewidth=1.8, label='chance')
    ax.set_xscale('log', base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('ghost logits')
    ax.set_ylabel('Final test accuracy')
    ax.set_ylim(0, 1.02)
    ax.set_title('Full data, nonfrozen logits, no projection')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_dir / 'figure10b_full_data_nonfrozen_summary.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-root', type=Path, default=Path('main_experiments/mnist_runs'))
    parser.add_argument('--out-dir', type=Path, default=Path('main_experiments/mnist_runs/figure10b_full_data_nonfrozen'))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    teach_acc = teacher_accuracy(args.runs_root)
    rows = {}
    records = []
    for g in GHOST_COUNTS:
        records.append({'setup': 'teacher', 'label': 'Teacher', 'ghost_logits': g, 'epoch': 5, 'accuracy': teach_acc, 'ci95': np.nan, 'status': 'complete'})
        for setup, label in SETUPS:
            row = read_final_accuracy(args.runs_root, setup, g)
            rows[(setup, g)] = row
            records.append({
                'setup': setup,
                'label': label,
                'ghost_logits': g,
                'epoch': row['epoch'] if row else np.nan,
                'accuracy': row['accuracy'] if row else np.nan,
                'ci95': row['ci95'] if row else np.nan,
                'status': 'complete' if row else 'pending',
            })
        plot_one(args.out_dir, g, teach_acc, rows)
    pd.DataFrame(records).to_csv(args.out_dir / 'figure10b_full_data_nonfrozen_summary.csv', index=False)
    meta = {
        'teacher_accuracy': teach_acc,
        'teacher_readout': 'nonfrozen',
        'student_condition': 'logit_distilation_B_readouts_nonfrozen',
        'data': 'full',
        'projection': False,
        'out_dir': str(args.out_dir),
    }
    (args.out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    plot_summary(args.out_dir, teach_acc, rows)
    print(f'wrote plots to {args.out_dir}')
    print(f'teacher_accuracy={teach_acc:.6f}')


if __name__ == '__main__':
    main()
