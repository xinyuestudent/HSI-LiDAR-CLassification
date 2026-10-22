"""Losses for the first OVDG-Mamba milestone."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def semantic_topology(embeddings: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity topology of a class embedding bank."""
    embeddings = F.normalize(embeddings, dim=-1)
    return embeddings @ embeddings.transpose(-1, -2)


class OVDGLoss(nn.Module):
    """Seen-class supervision plus visual--language relational alignment.

    ``domain_visual_prototypes`` is optional during the first integration. Once
    multi-source batches are available it should contain [D, C_seen, E], which
    activates the cross-domain semantic-topology consistency term.
    """

    def __init__(
        self,
        lambda_vt: float = 1.0,
        lambda_rel: float = 0.2,
        lambda_dg: float = 0.1,
        lambda_offset: float = 0.01,
    ):
        super().__init__()
        self.lambda_vt = lambda_vt
        self.lambda_rel = lambda_rel
        self.lambda_dg = lambda_dg
        self.lambda_offset = lambda_offset

    @staticmethod
    def _offset_smoothness(offsets: torch.Tensor) -> torch.Tensor:
        dx = offsets[:, :, 1:, :] - offsets[:, :, :-1, :]
        dy = offsets[:, 1:, :, :] - offsets[:, :-1, :, :]
        return dx.abs().mean() + dy.abs().mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        domain_visual_prototypes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        classification = F.cross_entropy(outputs["logits"], labels)

        query = outputs["query_embedding"]
        text = outputs["text_embeddings"]
        visual_text_logits = query @ text.t() / 0.07
        visual_text = F.cross_entropy(visual_text_logits, labels)

        visual_prototypes = outputs["visual_prototypes"]
        num_seen = visual_prototypes.shape[0]
        text_seen = text[:num_seen]
        relational = F.mse_loss(
            semantic_topology(visual_prototypes),
            semantic_topology(text_seen),
        )

        domain_consistency = classification.new_zeros(())
        if domain_visual_prototypes is not None:
            if domain_visual_prototypes.dim() != 3:
                raise ValueError(
                    "domain_visual_prototypes must have shape [D, C_seen, E]"
                )
            domain_topologies = semantic_topology(domain_visual_prototypes)
            mean_topology = domain_topologies.mean(dim=0, keepdim=True)
            domain_consistency = F.mse_loss(
                domain_topologies,
                mean_topology.expand_as(domain_topologies),
            )

        offset_smoothness = self._offset_smoothness(outputs["offsets"])
        total = (
            classification
            + self.lambda_vt * visual_text
            + self.lambda_rel * relational
            + self.lambda_dg * domain_consistency
            + self.lambda_offset * offset_smoothness
        )
        return {
            "total": total,
            "classification": classification,
            "visual_text": visual_text,
            "relational": relational,
            "domain_consistency": domain_consistency,
            "offset_smoothness": offset_smoothness,
        }

