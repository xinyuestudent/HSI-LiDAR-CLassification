"""Physics-conditioned multimodal open-set model.

This path deliberately has no CLIP or Prompt Bank dependency.  Known-class
classification and unknown rejection are both derived from the same physical
energy function, so open-set detection is not a classifier plus a detached
post-hoc module.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LiDARFeatureExtractor
from ovdg_model import (
    LocalOffsetAlignment,
    PhysicsAwareConflictFusion,
    WavelengthAwareSpectralEncoder,
    warp_with_offsets,
)


def _cosine_distance(query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    return 1.0 - query @ prototypes.t()


class PhysicsConditionedEnergyHead(nn.Module):
    """Class-conditioned energy over spectral, structural and relational cues."""

    def __init__(
        self,
        feature_dim: int,
        num_known_classes: int,
        temperature: float = 0.1,
    ):
        super().__init__()
        if num_known_classes < 2:
            raise ValueError("num_known_classes must be at least two")
        self.num_known_classes = int(num_known_classes)
        self.temperature = float(temperature)

        self.hsi_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim))
        self.aux_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim))
        self.relation_projection = nn.Sequential(
            nn.LayerNorm(feature_dim * 4),
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        self.hsi_prototypes = nn.Parameter(torch.empty(num_known_classes, feature_dim))
        self.aux_prototypes = nn.Parameter(torch.empty(num_known_classes, feature_dim))
        self.relation_prototypes = nn.Parameter(torch.empty(num_known_classes, feature_dim))
        nn.init.normal_(self.hsi_prototypes, std=0.02)
        nn.init.normal_(self.aux_prototypes, std=0.02)
        nn.init.normal_(self.relation_prototypes, std=0.02)

        # softplus keeps every physical contribution non-negative.
        self.raw_energy_weights = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        hsi_feature: torch.Tensor,
        aligned_aux_feature: torch.Tensor,
        fused_feature: torch.Tensor,
        diagnostics: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        hsi_pooled = hsi_feature.mean(dim=(1, 2))
        aux_pooled = aligned_aux_feature.mean(dim=(1, 2))
        fused_pooled = fused_feature.mean(dim=(1, 2))
        relation = self.relation_projection(
            torch.cat(
                [hsi_pooled, aux_pooled, torch.abs(hsi_pooled - aux_pooled), fused_pooled],
                dim=-1,
            )
        )

        hsi_embedding = self.hsi_projection(hsi_pooled)
        aux_embedding = self.aux_projection(aux_pooled)
        hsi_distance = _cosine_distance(hsi_embedding, self.hsi_prototypes)
        aux_distance = _cosine_distance(aux_embedding, self.aux_prototypes)
        relation_distance = _cosine_distance(relation, self.relation_prototypes)

        agreement = diagnostics["boundary_agreement"].mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        complementary = diagnostics["complementary_conflict"].mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        harmful = diagnostics["harmful_conflict"].mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        relation_support = (agreement + complementary).clamp(0.0, 1.0)

        weights = F.softplus(self.raw_energy_weights) + 1e-4
        energies = (
            weights[0] * hsi_distance
            + weights[1] * aux_distance
            + weights[2] * relation_support * relation_distance
            + weights[3] * harmful
        )
        unknown_score, closed_set_prediction = energies.min(dim=-1)
        return {
            "energies": energies,
            "logits": -energies / self.temperature,
            "unknown_score": unknown_score,
            "closed_set_prediction": closed_set_prediction,
            "hsi_embedding": F.normalize(hsi_embedding, dim=-1),
            "aux_embedding": F.normalize(aux_embedding, dim=-1),
            "relation_embedding": F.normalize(relation, dim=-1),
            "hsi_prototypes": F.normalize(self.hsi_prototypes, dim=-1),
            "aux_prototypes": F.normalize(self.aux_prototypes, dim=-1),
            "relation_prototypes": F.normalize(self.relation_prototypes, dim=-1),
            "energy_weights": weights,
            "relation_support": relation_support,
        }


class PhysicsConditionedOpenSetMamba(nn.Module):
    """HSI--LiDAR model with one energy space for recognition and rejection."""

    def __init__(
        self,
        num_known_classes: int,
        hsi_embed_dim: int = 64,
        feature_dim: int = 64,
        aux_channels: int = 1,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        max_offset_pixels: float = 2.0,
        dropout: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_known_classes = int(num_known_classes)
        self.hsi_encoder = WavelengthAwareSpectralEncoder(
            embed_dim=hsi_embed_dim,
            output_dim=feature_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.aux_encoder = LiDARFeatureExtractor(aux_channels, feature_dim, feature_dim)
        self.alignment = LocalOffsetAlignment(feature_dim, max_offset_pixels)
        self.fusion = PhysicsAwareConflictFusion(feature_dim, dropout)
        self.energy_head = PhysicsConditionedEnergyHead(
            feature_dim, num_known_classes, temperature
        )

    def forward(
        self,
        hsi: torch.Tensor,
        auxiliary: torch.Tensor,
        wavelengths: torch.Tensor,
        band_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hsi_feature = self.hsi_encoder(hsi, wavelengths, band_mask)
        aux_feature = self.aux_encoder(auxiliary)
        aligned_aux, offsets = self.alignment(hsi_feature, aux_feature)
        aligned_raw_aux = warp_with_offsets(auxiliary, offsets)
        fused, diagnostics = self.fusion(hsi_feature, aligned_aux, aligned_raw_aux)
        outputs = self.energy_head(hsi_feature, aligned_aux, fused, diagnostics)
        outputs.update(diagnostics)
        outputs.update(
            {
                "hsi_feature": hsi_feature,
                "aligned_aux_feature": aligned_aux,
                "fused_feature": fused,
                "offsets": offsets,
            }
        )
        return outputs

    @torch.no_grad()
    def predict_open_set(self, outputs: Dict[str, torch.Tensor], threshold: float) -> torch.Tensor:
        prediction = outputs["closed_set_prediction"].clone()
        prediction[outputs["unknown_score"] > threshold] = self.num_known_classes
        return prediction


def create_open_set_model(num_known_classes: int, **kwargs) -> PhysicsConditionedOpenSetMamba:
    return PhysicsConditionedOpenSetMamba(num_known_classes=num_known_classes, **kwargs)

