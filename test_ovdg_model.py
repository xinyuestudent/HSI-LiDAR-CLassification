"""Smoke tests for the first OVDG-Mamba implementation."""

import torch

from ovdg_losses import OVDGLoss
from ovdg_model import OVDGMamba, split_conflict


def test_conflict_decomposition():
    conflict = torch.tensor([[[[0.8]], [[0.8]]]])
    agreement = torch.tensor([[[[0.9]], [[0.1]]]])
    complementary, harmful = split_conflict(conflict, agreement)

    assert complementary[0, 0].item() > harmful[0, 0].item()
    assert harmful[0, 1].item() > complementary[0, 1].item()
    assert torch.allclose(complementary + harmful, conflict)


def test_ovdg_forward_and_backward():
    torch.manual_seed(7)
    batch_size, height, width, bands = 2, 5, 5, 12
    num_classes, num_seen, text_dim = 6, 4, 32
    text_embeddings = torch.randn(num_classes, text_dim)

    model = OVDGMamba(
        text_embeddings=text_embeddings,
        num_seen_classes=num_seen,
        hsi_embed_dim=16,
        feature_dim=16,
        aux_channels=1,
        d_state=4,
        expand=1,
        max_offset_pixels=1.0,
    )
    hsi = torch.randn(batch_size, height, width, bands)
    auxiliary = torch.randn(batch_size, 1, height, width)
    wavelengths = torch.linspace(450.0, 900.0, bands)
    labels = torch.tensor([0, 3])

    outputs = model(hsi, auxiliary, wavelengths)
    assert outputs["logits"].shape == (batch_size, num_classes)
    assert outputs["offsets"].shape == (batch_size, height, width, 2)
    assert outputs["complementary_conflict"].shape == (
        batch_size, height, width, 1
    )
    assert outputs["harmful_conflict"].shape == (
        batch_size, height, width, 1
    )

    losses = OVDGLoss()(outputs, labels)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert model.semantic_head.visual_projection.weight.grad is not None


if __name__ == "__main__":
    test_conflict_decomposition()
    test_ovdg_forward_and_backward()
    print("OVDG-Mamba smoke tests passed")

