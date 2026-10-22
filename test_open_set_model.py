"""Smoke tests for the fixed-taxonomy Open-Set path."""

import numpy as np
import torch

from open_set_losses import OpenSetEnergyLoss
from open_set_metrics import compute_open_set_metrics
from open_set_model import PhysicsConditionedOpenSetMamba


def test_open_set_forward_backward_and_rejection():
    torch.manual_seed(4)
    model = PhysicsConditionedOpenSetMamba(
        num_known_classes=3, hsi_embed_dim=16, feature_dim=16,
        d_state=4, expand=1, max_offset_pixels=1.0,
    )
    outputs = model(
        torch.randn(2, 5, 5, 12),
        torch.randn(2, 1, 5, 5),
        torch.linspace(450.0, 900.0, 12),
    )
    assert outputs["energies"].shape == (2, 3)
    assert outputs["unknown_score"].shape == (2,)
    assert torch.all(outputs["energy_weights"] > 0)
    loss = OpenSetEnergyLoss()(outputs, torch.tensor([0, 2]))["total"]
    loss.backward()
    assert torch.isfinite(loss)
    rejected = model.predict_open_set(outputs, threshold=-1.0)
    assert torch.equal(rejected, torch.full_like(rejected, 3))


def test_open_set_metrics():
    metrics = compute_open_set_metrics(
        np.array([0, 1, 2, 3, 3]),
        np.array([0, 1, 2, 3, 3]),
        np.array([0.1, 0.2, 0.3, 0.8, 0.9]),
        num_known_classes=3,
    )
    assert metrics["harmonic_mean"] == 1.0
    assert metrics["auroc_unknown"] == 1.0


if __name__ == "__main__":
    test_open_set_forward_backward_and_rejection()
    test_open_set_metrics()
    print("Open-Set model smoke tests passed")

