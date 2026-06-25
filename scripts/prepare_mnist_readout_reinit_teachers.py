import argparse
import json
from pathlib import Path

import pandas as pd
import torch as t
from torch import nn
import tqdm

from run_mnist_experiment import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS_TEACHER,
    LR,
    N_MODELS,
    SEED,
    MultiClassifier,
    PreloadedDataLoader,
    get_mnist,
    restore_readout_rows,
    snapshot_readout_rows,
    zero_readout_row_grads,
)
N_MODELS = 1
EXPERIMENT_N_MODELS = N_MODELS

from run_mnist_readout_reinit_grid_job import (
    MAX_GHOST_LOGITS,
    TEACHER_DIR_NAMES,
    TEACHER_READOUTS,
    spectrum_for_model,
    to_tensor,
    final_readout,
    CNNStudent,
)


def train_teacher(model, x, y, epochs, freeze_final_readout):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    frozen_weight = frozen_bias = None
    all_readout_idx = list(range(final_readout(model).bias.shape[1]))
    if freeze_final_readout:
        frozen_weight, frozen_bias = snapshot_readout_rows(model, all_readout_idx)
    for _ in tqdm.trange(epochs, desc='train_teacher'):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = nn.functional.cross_entropy(model(bx)[..., :10].flatten(0, 1), by.flatten())
            opt.zero_grad()
            loss.backward()
            if freeze_final_readout:
                zero_readout_row_grads(model, all_readout_idx)
            opt.step()
            if freeze_final_readout:
                restore_readout_rows(model, all_readout_idx, frozen_weight, frozen_bias)


def main():
    parser = argparse.ArgumentParser(description='Prepare shared max-output teachers for readout-reinit grid.')
    parser.add_argument('--teacher-readout', choices=TEACHER_READOUTS, required=True)
    parser.add_argument('--teacher-arch', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--latent-dim', type=int, default=256, help='Final hidden/latent width before the readout for MLP teachers.')
    parser.add_argument('--teacher-epochs', type=int, default=EPOCHS_TEACHER)
    parser.add_argument('--spectrum-items', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    t.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = args.out_dir / TEACHER_DIR_NAMES[args.teacher_readout] / 'teacher_artifacts'
    teacher_dir.mkdir(parents=True, exist_ok=True)
    model_path = teacher_dir / 'model.pt'
    spectra_path = teacher_dir / 'spectra.csv'
    meta_path = teacher_dir / 'meta.json'
    if model_path.exists() and spectra_path.exists():
        print(f'skipping existing teacher {teacher_dir}')
        return

    train_ds, test_ds = get_mnist()
    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, _ = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    if args.teacher_arch == 'cnn':
        model = CNNStudent(N_MODELS, 10 + MAX_GHOST_LOGITS).to(DEVICE)
    else:
        model = MultiClassifier(N_MODELS, [28 * 28, 256, args.latent_dim, 10 + MAX_GHOST_LOGITS]).to(DEVICE)
    train_teacher(model, train_x, train_y, args.teacher_epochs, args.teacher_readout == 'frozen')

    cfg = {
        'teacher_readout': args.teacher_readout,
        'teacher_architecture': args.teacher_arch,
        'teacher_readout_frozen': args.teacher_readout == 'frozen',
        'max_ghost_logits': MAX_GHOST_LOGITS,
        'latent_dim': args.latent_dim,
        'seed': args.seed,
        'teacher_epochs': args.teacher_epochs,
    }
    t.save({'state_dict': model.state_dict(), 'cfg': cfg}, model_path)
    spectra = pd.DataFrame(spectrum_for_model(model, test_x, args.spectrum_items, args.teacher_epochs, 'teacher'))
    for key, value in cfg.items():
        spectra[key] = value
    spectra.to_csv(spectra_path, index=False)
    meta_path.write_text(json.dumps(cfg, indent=2))
    print(f'wrote {model_path}')
    print(f'wrote {spectra_path}')
    print(f'wrote {meta_path}')


if __name__ == '__main__':
    main()
