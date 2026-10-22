"""Tests that do not require the remote-sensing MAT datasets."""

import numpy as np
import torch

from ovdg_data import OVDGPatchDataset, pad_ovdg_batch
from ovdg_metrics import compute_ovdg_metrics


def test_variable_band_collation():
    dataset_a = OVDGPatchDataset(
        np.zeros((1, 3, 3, 4), dtype=np.float32),
        np.zeros((1, 3, 3, 1), dtype=np.float32),
        np.array([0]),
        np.array([450, 550, 650, 750], dtype=np.float32),
        domain_id=0,
    )
    dataset_b = OVDGPatchDataset(
        np.zeros((1, 3, 3, 6), dtype=np.float32),
        np.zeros((1, 3, 3, 1), dtype=np.float32),
        np.array([2]),
        np.array([420, 500, 580, 660, 740, 820], dtype=np.float32),
        domain_id=1,
    )
    batch = pad_ovdg_batch([dataset_a[0], dataset_b[0]])
    assert batch["hsi"].shape == (2, 3, 3, 6)
    assert batch["band_mask"].sum(dim=1).tolist() == [4, 6]
    assert batch["auxiliary"].shape == (2, 1, 3, 3)


def test_seen_unseen_metrics():
    metrics = compute_ovdg_metrics(
        labels=np.array([0, 1, 2, 3]),
        predictions=np.array([0, 0, 2, 1]),
        seen_mask=np.array([True, True, False, False]),
    )
    assert metrics["seen_accuracy"] == 0.5
    assert metrics["unseen_accuracy"] == 0.5
    assert metrics["harmonic_mean"] == 0.5


if __name__ == "__main__":
    test_variable_band_collation()
    test_seen_unseen_metrics()
    print("OVDG data/metric tests passed")

