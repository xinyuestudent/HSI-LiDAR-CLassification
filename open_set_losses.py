"""Losses for the unified physics-conditioned open-set energy space."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prototype_separation(prototypes: torch.Tensor, margin: float) -> torch.Tensor:
    similarity = prototypes @ prototypes.t()
    mask = ~torch.eye(len(prototypes), dtype=torch.bool, device=prototypes.device)
    return F.relu(similarity[mask] - margin).mean()


class OpenSetEnergyLoss(nn.Module):
    def __init__(
        self,
        energy_margin: float = 0.25,
        prototype_similarity_margin: float = 0.3,
        lambda_compact: float = 0.2,
        lambda_margin: float = 0.5,
        lambda_separation: float = 0.05,
        lambda_offset: float = 0.01,
    ):
        super().__init__()
        self.energy_margin = energy_margin
        self.prototype_similarity_margin = prototype_similarity_margin
        self.lambda_compact = lambda_compact
        self.lambda_margin = lambda_margin
        self.lambda_separation = lambda_separation
        self.lambda_offset = lambda_offset

    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        energies = outputs["energies"]
        classification = F.cross_entropy(outputs["logits"], labels)
        true_energy = energies.gather(1, labels[:, None]).squeeze(1)
        wrong = energies.masked_fill(
            F.one_hot(labels, energies.shape[1]).bool(), float("inf")
        ).min(dim=1).values
        compactness = true_energy.mean()
        ranking = F.relu(self.energy_margin + true_energy - wrong).mean()
        separation = sum(
            _prototype_separation(outputs[name], self.prototype_similarity_margin)
            for name in ("hsi_prototypes", "aux_prototypes", "relation_prototypes")
        ) / 3.0

        offsets = outputs["offsets"]
        smooth_x = (offsets[:, :, 1:] - offsets[:, :, :-1]).abs().mean()
        smooth_y = (offsets[:, 1:] - offsets[:, :-1]).abs().mean()
        offset_smoothness = smooth_x + smooth_y
        total = (
            classification
            + self.lambda_compact * compactness
            + self.lambda_margin * ranking
            + self.lambda_separation * separation
            + self.lambda_offset * offset_smoothness
        )
        return {
            "total": total,
            "classification": classification,
            "compactness": compactness,
            "energy_margin": ranking,
            "prototype_separation": separation,
            "offset_smoothness": offset_smoothness,
        }

