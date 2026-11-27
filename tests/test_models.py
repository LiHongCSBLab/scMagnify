"""Unit tests for scmagnify.models (MSNGC and model utilities)."""

from __future__ import annotations

import numpy as np
import torch

from scmagnify.models import MSNGC, normalize_data


def test_normalize_data_zero_mean_unit_std():
    raw = np.array([[1.0, 5.0, 2.0], [2.0, 5.0, 4.0], [3.0, 5.0, 6.0]], dtype=np.float32)
    normed = normalize_data(raw)
    assert isinstance(normed, torch.Tensor)
    assert normed.shape == (3, 3)
    assert not torch.isnan(normed).any()
    # Constant middle column should remain 0 without division-by-zero NaN
    assert torch.allclose(normed[:, 1], torch.zeros(3))


def test_msngc_forward_and_backward_cpu():
    torch.manual_seed(0)
    batch_size, lag, n_reg, n_target = 8, 3, 4, 6
    model = MSNGC(n_reg=n_reg, n_target=n_target, hidden=[16], lag=lag, device="cpu")
    model.init_weights()

    X = torch.randn(batch_size, lag, n_reg)
    coeffs, y_pred, attn = model(X)

    assert coeffs.shape == (batch_size, lag, n_target, n_reg)
    assert y_pred.shape == (batch_size, n_target)
    assert attn.shape == (1, lag)

    loss = y_pred.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_msngc_respects_chromatin_constraint():
    torch.manual_seed(0)
    batch_size, lag, n_reg, n_target = 4, 2, 3, 5
    constraint = np.ones((n_reg, n_target), dtype=np.float32)
    constraint[0, 0] = 0.0
    constraint[2, 4] = 0.0

    model = MSNGC(
        n_reg=n_reg,
        n_target=n_target,
        chrom_constraint=constraint,
        hidden=[8],
        lag=lag,
        device="cpu",
    )
    X = torch.randn(batch_size, lag, n_reg)
    coeffs, _, _ = model(X)

    # coeffs shape is (batch, lag, n_target, n_reg)
    assert torch.all(coeffs[:, :, 0, 0] == 0.0)
    assert torch.all(coeffs[:, :, 4, 2] == 0.0)
