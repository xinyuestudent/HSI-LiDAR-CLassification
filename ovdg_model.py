"""OVDG-Mamba core model.

This module implements the first executable milestone of the cross-domain,
open-vocabulary HSI--auxiliary-modality architecture:

1. wavelength-aware spectral tokens and a true band-axis Mamba scan;
2. learned local geometric alignment before cross-modal fusion;
3. physics-inspired conflict decomposition that distinguishes complementary
   disagreement from harmful misalignment;
4. a visual--text relational semantic head that accepts pre-computed CLIP text
   embeddings and supports text-only unseen classes.

CLIP is deliberately not imported here. Text embeddings should be generated
once by a separate prompt pipeline and passed to :class:`OVDGMamba`. This keeps
the scientific model independent of a particular CLIP implementation/version.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CrossScan2D, LiDARFeatureExtractor, SelectiveScanMamba


def _normalise_wavelengths(wavelengths: torch.Tensor) -> torch.Tensor:
    """Convert common nm inputs to a stable approximately [0, 1] range."""
    wavelengths = wavelengths.float()
    # HSI metadata are commonly stored in nm (roughly 400--2500) or micrometres.
    scale = torch.where(
        wavelengths.detach().amax(dim=-1, keepdim=True) > 10.0,
        wavelengths.new_tensor(2500.0),
        wavelengths.new_tensor(2.5),
    )
    return wavelengths / scale


class WavelengthEmbedding(nn.Module):
    """Continuous wavelength position encoding with Fourier features."""

    def __init__(self, embed_dim: int, num_frequencies: int = 8):
        super().__init__()
        self.num_frequencies = num_frequencies
        fourier_dim = num_frequencies * 2
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, wavelengths: torch.Tensor) -> torch.Tensor:
        wavelengths = _normalise_wavelengths(wavelengths)
        frequencies = 2.0 ** torch.arange(
            self.num_frequencies,
            device=wavelengths.device,
            dtype=wavelengths.dtype,
        )
        phase = torch.pi * wavelengths.unsqueeze(-1) * frequencies
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.mlp(features)


class WavelengthAwareSpectralEncoder(nn.Module):
    """Encode variable-band HSI by scanning the actual spectral axis.

    Unlike the legacy ``SpectralSpatialMamba``, the sequence length passed to
    the spectral SSM is ``C`` (the number of bands), not ``H * W``. Variable
    sensors are handled by padding bands and supplying ``band_mask``.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        output_dim: int = 64,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_projection = nn.Linear(1, embed_dim)
        self.wavelength_embedding = WavelengthEmbedding(embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.spectral_mamba = SelectiveScanMamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.band_pool_score = nn.Linear(embed_dim, 1)
        self.spatial_mamba = CrossScan2D(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _prepare_metadata(
        wavelengths: torch.Tensor,
        band_mask: Optional[torch.Tensor],
        batch_size: int,
        num_bands: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if wavelengths.dim() == 1:
            wavelengths = wavelengths.unsqueeze(0).expand(batch_size, -1)
        if wavelengths.shape != (batch_size, num_bands):
            raise ValueError(
                "wavelengths must have shape [C] or [B, C], got {}".format(
                    tuple(wavelengths.shape)
                )
            )
        if band_mask is None:
            band_mask = torch.ones_like(wavelengths, dtype=torch.bool)
        elif band_mask.dim() == 1:
            band_mask = band_mask.unsqueeze(0).expand(batch_size, -1)
        if band_mask.shape != wavelengths.shape:
            raise ValueError("band_mask and wavelengths must have the same shape")
        return wavelengths, band_mask.bool()

    def forward(
        self,
        hsi: torch.Tensor,
        wavelengths: torch.Tensor,
        band_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hsi.dim() != 4:
            raise ValueError("hsi must have shape [B, H, W, C]")
        batch_size, height, width, num_bands = hsi.shape
        wavelengths, band_mask = self._prepare_metadata(
            wavelengths, band_mask, batch_size, num_bands
        )

        value_tokens = self.value_projection(hsi.unsqueeze(-1))
        wave_tokens = self.wavelength_embedding(wavelengths)
        wave_tokens = wave_tokens[:, None, None, :, :]
        tokens = self.token_norm(value_tokens + wave_tokens)

        tokens = tokens.reshape(
            batch_size * height * width, num_bands, self.embed_dim
        )
        tokens = self.spectral_mamba(tokens)

        mask = band_mask[:, None, None, :].expand(
            batch_size, height, width, num_bands
        )
        mask = mask.reshape(batch_size * height * width, num_bands)
        scores = self.band_pool_score(tokens).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        spectral = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        spectral = spectral.reshape(batch_size, height, width, self.embed_dim)

        spatial = self.spatial_mamba(spectral)
        return self.output_projection(torch.cat([spectral, spatial], dim=-1))


class LocalOffsetAlignment(nn.Module):
    """Align an auxiliary feature map to HSI using bounded local offsets."""

    def __init__(self, feature_dim: int, max_offset_pixels: float = 2.0):
        super().__init__()
        self.max_offset_pixels = float(max_offset_pixels)
        hidden_dim = max(feature_dim // 2, 16)
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(feature_dim * 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 3, padding=1),
        )
        # Start from identity alignment and learn corrections only when useful.
        nn.init.zeros_(self.offset_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].bias)

    def forward(
        self, hsi_feature: torch.Tensor, aux_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hsi_feature.shape != aux_feature.shape:
            raise ValueError("HSI and auxiliary features must have identical shapes")
        batch_size, height, width, _ = hsi_feature.shape
        hsi_cf = hsi_feature.permute(0, 3, 1, 2)
        aux_cf = aux_feature.permute(0, 3, 1, 2)
        offsets = torch.tanh(
            self.offset_predictor(torch.cat([hsi_cf, aux_cf], dim=1))
        ) * self.max_offset_pixels

        offsets_bhwc = offsets.permute(0, 2, 3, 1)
        aligned = warp_with_offsets(aux_cf, offsets_bhwc)
        return aligned.permute(0, 2, 3, 1), offsets_bhwc


def warp_with_offsets(tensor: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Warp a BCHW tensor with pixel-unit BHWC offsets."""
    batch_size, _, height, width = tensor.shape
    if offsets.shape != (batch_size, height, width, 2):
        raise ValueError("offsets must have shape [B, H, W, 2]")
    y = torch.linspace(-1.0, 1.0, height, device=tensor.device, dtype=tensor.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=tensor.device, dtype=tensor.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base_grid = torch.stack([grid_x, grid_y], dim=-1)
    base_grid = base_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    normaliser_x = max((width - 1) / 2.0, 1.0)
    normaliser_y = max((height - 1) / 2.0, 1.0)
    offset_grid = torch.stack(
        [offsets[..., 0] / normaliser_x, offsets[..., 1] / normaliser_y],
        dim=-1,
    )
    return F.grid_sample(
        tensor,
        base_grid + offset_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Finite-difference gradient magnitude for [B, 1, H, W] maps."""
    dx = F.pad(image[:, :, :, 1:] - image[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(image[:, :, 1:, :] - image[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


def split_conflict(
    feature_conflict: torch.Tensor, boundary_agreement: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split disagreement into complementary and harmful components.

    Feature disagreement with aligned physical boundaries is treated as
    potentially complementary: spectra and elevation describe different but
    co-located properties. Disagreement without boundary support is treated as
    likely misregistration/noise and therefore harmful.
    """
    boundary_agreement = boundary_agreement.clamp(0.0, 1.0)
    feature_conflict = feature_conflict.clamp(0.0, 1.0)
    complementary = feature_conflict * boundary_agreement
    harmful = feature_conflict * (1.0 - boundary_agreement)
    return complementary, harmful


class PhysicsAwareConflictFusion(nn.Module):
    """Fuse material and structural evidence while preserving useful conflict."""

    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hsi_shared = nn.Linear(feature_dim, feature_dim)
        self.aux_shared = nn.Linear(feature_dim, feature_dim)
        self.hsi_reliability = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())
        self.aux_reliability = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())
        self.complementary_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )

    @staticmethod
    def _normalise_boundary(boundary: torch.Tensor) -> torch.Tensor:
        scale = boundary.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return boundary / scale

    def forward(
        self,
        hsi_feature: torch.Tensor,
        aligned_aux_feature: torch.Tensor,
        raw_auxiliary: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hsi_shared = self.hsi_shared(hsi_feature)
        aux_shared = self.aux_shared(aligned_aux_feature)
        cosine = F.cosine_similarity(hsi_shared, aux_shared, dim=-1).unsqueeze(-1)
        feature_conflict = (1.0 - cosine) * 0.5

        hsi_energy = hsi_feature.norm(dim=-1).unsqueeze(1)
        if raw_auxiliary is None:
            aux_energy = aligned_aux_feature.norm(dim=-1).unsqueeze(1)
        else:
            aux_energy = raw_auxiliary.mean(dim=1, keepdim=True)
            if aux_energy.shape[-2:] != hsi_energy.shape[-2:]:
                aux_energy = F.interpolate(
                    aux_energy,
                    size=hsi_energy.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        hsi_boundary = self._normalise_boundary(_gradient_magnitude(hsi_energy))
        aux_boundary = self._normalise_boundary(_gradient_magnitude(aux_energy))
        boundary_agreement = torch.exp(-torch.abs(hsi_boundary - aux_boundary))
        boundary_agreement = boundary_agreement.permute(0, 2, 3, 1)
        complementary, harmful = split_conflict(
            feature_conflict, boundary_agreement
        )

        hsi_reliability = self.hsi_reliability(hsi_feature)
        aux_reliability = self.aux_reliability(aligned_aux_feature)
        # Harmful conflict suppresses the auxiliary contribution. Complementary
        # conflict is not suppressed; it opens a residual evidence pathway.
        effective_aux = aux_reliability * (1.0 - harmful)
        denominator = (hsi_reliability + effective_aux).clamp_min(1e-6)
        hsi_weight = hsi_reliability / denominator
        aux_weight = effective_aux / denominator
        shared_fusion = hsi_weight * hsi_shared + aux_weight * aux_shared

        complementary_evidence = self.complementary_projection(
            torch.cat([hsi_feature, aligned_aux_feature], dim=-1)
        )
        fused = shared_fusion + complementary * complementary_evidence
        fused = self.output(fused)
        diagnostics = {
            "feature_conflict": feature_conflict,
            "boundary_agreement": boundary_agreement,
            "complementary_conflict": complementary,
            "harmful_conflict": harmful,
            "hsi_weight": hsi_weight,
            "aux_weight": aux_weight,
        }
        return fused, diagnostics


class RelationalOpenVocabularyHead(nn.Module):
    """Reason over visual prototypes and fixed language semantic topology."""

    def __init__(
        self,
        visual_dim: int,
        text_embeddings: torch.Tensor,
        num_seen_classes: int,
        temperature: float = 0.07,
    ):
        super().__init__()
        if text_embeddings.dim() != 2:
            raise ValueError("text_embeddings must have shape [classes, text_dim]")
        num_classes, text_dim = text_embeddings.shape
        if not 0 < num_seen_classes <= num_classes:
            raise ValueError("num_seen_classes must be in [1, num_classes]")
        self.num_classes = num_classes
        self.num_seen_classes = num_seen_classes
        self.temperature = temperature
        self.visual_projection = nn.Linear(visual_dim, text_dim)
        self.visual_prototypes = nn.Parameter(torch.empty(num_seen_classes, text_dim))
        nn.init.normal_(self.visual_prototypes, std=0.02)
        self.node_projection = nn.Linear(text_dim, text_dim, bias=False)
        self.graph_scale = nn.Parameter(torch.tensor(0.1))
        self.score_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.register_buffer(
            "text_embeddings", F.normalize(text_embeddings.float(), dim=-1)
        )

    def forward(self, visual_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        query = F.normalize(self.visual_projection(visual_feature), dim=-1)
        text = self.text_embeddings
        visual_prototypes = F.normalize(self.visual_prototypes, dim=-1)

        text_scores = query @ text.t()
        visual_scores = query.new_zeros(query.shape[0], self.num_classes)
        visual_scores[:, : self.num_seen_classes] = query @ visual_prototypes.t()

        class_nodes = text.clone()
        class_nodes[: self.num_seen_classes] = F.normalize(
            text[: self.num_seen_classes] + visual_prototypes, dim=-1
        )
        text_topology = F.softmax(text @ text.t() / self.temperature, dim=-1)
        graph_message = text_topology @ self.node_projection(class_nodes)
        graph_nodes = F.normalize(
            class_nodes + self.graph_scale * graph_message, dim=-1
        )
        graph_scores = query @ graph_nodes.t()

        weights = F.softmax(self.score_weights, dim=0)
        seen_mask = query.new_zeros(self.num_classes)
        seen_mask[: self.num_seen_classes] = 1.0
        combined = (
            weights[0] * text_scores
            + weights[1] * visual_scores * seen_mask
            + weights[2] * graph_scores
        )
        logits = combined / self.temperature
        return {
            "logits": logits,
            "query_embedding": query,
            "text_scores": text_scores,
            "visual_scores": visual_scores,
            "graph_scores": graph_scores,
            "text_embeddings": text,
            "visual_prototypes": visual_prototypes,
        }


class OVDGMamba(nn.Module):
    """Open-vocabulary, domain-generalised multimodal Mamba model."""

    def __init__(
        self,
        text_embeddings: torch.Tensor,
        num_seen_classes: int,
        hsi_embed_dim: int = 64,
        feature_dim: int = 64,
        aux_channels: int = 1,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        max_offset_pixels: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hsi_encoder = WavelengthAwareSpectralEncoder(
            embed_dim=hsi_embed_dim,
            output_dim=feature_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.aux_encoder = LiDARFeatureExtractor(
            input_dim=aux_channels,
            hidden_dim=feature_dim,
            output_dim=feature_dim,
        )
        self.alignment = LocalOffsetAlignment(
            feature_dim=feature_dim,
            max_offset_pixels=max_offset_pixels,
        )
        self.fusion = PhysicsAwareConflictFusion(feature_dim, dropout=dropout)
        self.semantic_head = RelationalOpenVocabularyHead(
            visual_dim=feature_dim,
            text_embeddings=text_embeddings,
            num_seen_classes=num_seen_classes,
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
        aligned_raw_auxiliary = warp_with_offsets(auxiliary, offsets)
        fused_feature, diagnostics = self.fusion(
            hsi_feature, aligned_aux, raw_auxiliary=aligned_raw_auxiliary
        )
        pooled_feature = fused_feature.mean(dim=(1, 2))
        outputs = self.semantic_head(pooled_feature)
        outputs.update(diagnostics)
        outputs.update(
            {
                "hsi_feature": hsi_feature,
                "aligned_aux_feature": aligned_aux,
                "fused_feature": fused_feature,
                "offsets": offsets,
            }
        )
        return outputs


def create_ovdg_model(
    text_embeddings: torch.Tensor,
    num_seen_classes: int,
    **kwargs
) -> OVDGMamba:
    """Factory kept small so experiment scripts can load text banks explicitly."""
    return OVDGMamba(
        text_embeddings=text_embeddings,
        num_seen_classes=num_seen_classes,
        **kwargs
    )

