# OVDG-Mamba implementation status

This branch starts the architecture agreed for cross-domain open-vocabulary
multimodal classification. The original HSLiNets path remains intact for
baseline reproduction.

## Implemented in milestone 1

- Wavelength-aware per-band tokenisation.
- A true spectral Mamba scan whose sequence axis is the HSI band axis.
- Padding masks for heterogeneous sensors.
- Learned bounded local offset alignment before multimodal fusion.
- Physics-inspired conflict decomposition:
  - feature disagreement with aligned HSI/DSM boundaries is retained as
    complementary conflict;
  - disagreement without boundary support is treated as harmful conflict and
    suppresses the auxiliary contribution.
- A relational open-vocabulary head using pre-computed CLIP text embeddings.
- Seen classes use visual and text prototypes; unseen classes remain text-only.
- Visual/text topology alignment and optional cross-domain topology loss.
- Paired HSI--LiDAR augmentation and the legacy band-slicing bug fix.

## Conflict interpretation

The physical assumption is intentionally narrow and testable. HSI primarily
describes material/spectral properties, while DSM/LiDAR describes geometry.
Their feature values may disagree even for a valid sample. If the spatial
boundaries still coincide after local alignment, that disagreement can carry
complementary evidence and is preserved. If feature disagreement is accompanied
by boundary mismatch, it is more likely caused by registration error or noise
and the auxiliary contribution is reduced.

The model exposes `boundary_agreement`, `complementary_conflict`,
`harmful_conflict`, `hsi_weight`, `aux_weight`, and learned `offsets` for
visualisation and ablation rather than hiding the decision inside one gate.

## Text embeddings

`OVDGMamba` accepts a tensor `[num_vocabulary_classes, text_dim]`. Generate this
tensor offline with a frozen CLIP text encoder and an auditable prompt list.
The model does not transform HSI into pseudo-RGB or use the CLIP image encoder.

Generate the first local bank with:

```powershell
pip install open_clip_torch
python clip_prompt_bank.py `
  --config configs/prompt_bank_landcover.json `
  --output artifacts/landcover_vit_b32_openai.pt `
  --model ViT-B-32 `
  --pretrained openai

python demo_prompt_bank.py --bank artifacts/landcover_vit_b32_openai.pt
```

The generator ensembles multiple prompts per class, normalises every prompt
embedding, averages the ensemble, and normalises the final class vector. It
saves the exact prompts, class order, seen mask, model name, and pretrained tag
beside the tensor so experiments remain auditable.

## Next milestone

1. Add Trento, Houston2013, and Augsburg multi-domain adapters with real centre
   wavelengths and common semantic mappings.
2. Integrate `OVDGLoss` into a multi-source trainer.
3. Add open-vocabulary seen/unseen metrics and leave-one-domain-out protocols.
4. Visualise offsets and beneficial/harmful conflict maps.

