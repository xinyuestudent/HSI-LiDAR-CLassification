# Physics-Conditioned Multimodal Open-Set v1

This branch adds a fixed-taxonomy Open-Set path while leaving the earlier
Open-Vocabulary/Prompt-Bank experiment unchanged.

## Core decision

The model does not attach a separate unknown detector to a closed-set network.
It learns one class-conditioned physical energy for every known class:

`E_k = w_h d(h,p_hk) + w_l d(l,p_lk) + w_r A d(r,p_rk) + w_c C_harmful`

Classification uses the class with minimum energy. Rejection uses that same
minimum energy: a sample is unknown when `min_k E_k > threshold`.

Boundary-supported conflict is retained as complementary relational evidence.
Conflict without aligned physical boundaries raises every known-class energy.

## Trento protocol

- Known: building, road, low_vegetation.
- Held-out unknown: ground and wood, collapsed to one `unknown` label.
- Unknown pixels never enter training or validation.
- Train, validation and test patch supports are spatially disjoint.
- The threshold is the selected quantile of known validation energies. Test
  unknown labels are never used to tune it.

## Run

For a first smoke experiment without verified wavelengths:

```bash
python train_open_set.py \
  --data-dir /path/to/TrentoDataset \
  --allow-linear-debug-wavelengths \
  --epochs 50 \
  --output-dir experiments/open_set_trento_pcme_seed42
```

When authentic centre wavelengths are available, replace the debug flag with:

```bash
--wavelengths /path/to/trento_wavelengths.json
```

Run at least seeds 42, 1021 and 456. Primary results are H, unknown AUROC,
known accuracy and unknown recall. `results.json` also stores macro-F1, AUPR,
FPR95, split counts and conflict diagnostics.

## Files

- `open_set_model.py`: unified physical energy model.
- `open_set_losses.py`: compactness, class margin and prototype separation.
- `open_set_data.py`: seen/unseen-aware spatial split.
- `open_set_metrics.py`: open-set metrics.
- `train_open_set.py`: training, known-only threshold calibration and testing.
- `configs/open_set_trento.json`: the fixed Trento taxonomy.

