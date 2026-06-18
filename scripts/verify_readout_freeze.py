import torch as t
from torch import nn

from run_mnist_experiment import (
    BATCH_SIZE,
    DEVICE,
    GHOST_IDX,
    LR,
    N_MODELS,
    TOTAL_OUT,
    MultiClassifier,
    PreloadedDataLoader,
    restore_readout_rows,
    snapshot_readout_rows,
    zero_readout_row_grads,
)


def read_rows(model, idx):
    layer = model.net[-1]
    index = t.tensor(idx, dtype=t.long, device=layer.weight.device)
    return layer.weight.index_select(1, index).detach().clone(), layer.bias.index_select(1, index).detach().clone()


def row_delta(model, idx, weight, bias):
    cur_w, cur_b = read_rows(model, idx)
    return float((cur_w - weight).abs().max().cpu()), float((cur_b - bias).abs().max().cpu())


def run_step(freeze):
    t.manual_seed(0)
    model = MultiClassifier(N_MODELS, [28 * 28, 256, 256, TOTAL_OUT]).to(DEVICE)
    teacher = MultiClassifier(N_MODELS, [28 * 28, 256, 256, TOTAL_OUT]).to(DEVICE)
    teacher.load_state_dict(model.state_dict())
    opt = t.optim.Adam(model.parameters(), lr=LR)
    x = t.rand((N_MODELS, BATCH_SIZE, 1, 28, 28), device=DEVICE) * 2 - 1
    frozen_weight, frozen_bias = snapshot_readout_rows(model, GHOST_IDX)
    initial_weight, initial_bias = read_rows(model, GHOST_IDX)

    with t.no_grad():
        target = teacher(x)[:, :, GHOST_IDX] + 0.1 * t.randn((N_MODELS, BATCH_SIZE, len(GHOST_IDX)), device=DEVICE)
    out = model(x)[:, :, GHOST_IDX]
    loss = nn.functional.kl_div(
        nn.functional.log_softmax(out, -1),
        nn.functional.softmax(target, -1),
        reduction="batchmean",
    )
    opt.zero_grad()
    loss.backward()

    layer = model.net[-1]
    index = t.tensor(GHOST_IDX, dtype=t.long, device=layer.weight.device)
    grad_before = float(layer.weight.grad.index_select(1, index).abs().max().detach().cpu())
    if freeze:
        zero_readout_row_grads(model, GHOST_IDX)
    grad_after = float(layer.weight.grad.index_select(1, index).abs().max().detach().cpu())
    opt.step()
    drift_after_step = row_delta(model, GHOST_IDX, initial_weight, initial_bias)
    if freeze:
        restore_readout_rows(model, GHOST_IDX, frozen_weight, frozen_bias)
    drift_after_restore = row_delta(model, GHOST_IDX, initial_weight, initial_bias)
    return {
        "freeze": freeze,
        "grad_before_zero": grad_before,
        "grad_after_zero": grad_after,
        "weight_drift_after_step": drift_after_step[0],
        "bias_drift_after_step": drift_after_step[1],
        "weight_drift_after_restore": drift_after_restore[0],
        "bias_drift_after_restore": drift_after_restore[1],
    }


def main():
    for freeze in [False, True]:
        result = run_step(freeze)
        print(result)


if __name__ == "__main__":
    main()
