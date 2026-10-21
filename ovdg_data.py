"""Data protocol for seen-to-seen+unseen OVDG experiments."""

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import TRENTO_CONFIG, get_class_dict
from dataset import (
    augment_paired_data,
    extract_patches_from_coords,
    load_trento_data,
)
from spatial_split import spatial_train_test_split


def load_wavelengths(
    path: Optional[str],
    expected_bands: int,
    allow_linear_debug: bool = False,
    debug_range_nm: Tuple[float, float] = (400.0, 1000.0),
) -> np.ndarray:
    """Load real sensor centre wavelengths, refusing silent fabrication."""
    if path is None:
        if not allow_linear_debug:
            raise ValueError(
                "A real centre-wavelength file is required. Pass --wavelengths, "
                "or explicitly use --allow-linear-debug-wavelengths for a smoke test."
            )
        return np.linspace(
            debug_range_nm[0], debug_range_nm[1], expected_bands,
            dtype=np.float32,
        )

    extension = os.path.splitext(path)[1].lower()
    if extension == ".npy":
        wavelengths = np.load(path)
    elif extension == ".json":
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        wavelengths = payload.get("wavelengths", payload) if isinstance(payload, dict) else payload
        wavelengths = np.asarray(wavelengths)
    elif extension in {".txt", ".csv"}:
        wavelengths = np.loadtxt(path, delimiter="," if extension == ".csv" else None)
    else:
        raise ValueError("Wavelength file must be .npy, .json, .txt, or .csv")

    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    if len(wavelengths) != expected_bands:
        raise ValueError(
            "Expected {} wavelengths, found {}".format(expected_bands, len(wavelengths))
        )
    if not np.isfinite(wavelengths).all() or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Wavelengths must be finite and strictly increasing")
    return wavelengths


def load_label_mapping(path: str, class_to_index: Dict[str, int]) -> Dict[int, int]:
    """Map original integer labels to the prompt bank's canonical order."""
    with open(path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    mapping = {}
    for raw_label, canonical_name in config["class_mapping"].items():
        if canonical_name is None:
            continue
        if canonical_name not in class_to_index:
            raise ValueError(
                "Canonical class '{}' is absent from the prompt bank".format(canonical_name)
            )
        mapping[int(raw_label)] = int(class_to_index[canonical_name])
    return mapping


def _filter_and_remap_coords(
    coords: np.ndarray,
    label_mapping: Dict[int, int],
    seen_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    kept_coords = []
    mapped_labels = []
    for row, col, original_label in coords:
        original_label = int(original_label)
        if original_label not in label_mapping:
            continue
        mapped = label_mapping[original_label]
        if seen_mask is not None and not bool(seen_mask[mapped]):
            continue
        kept_coords.append([int(row), int(col), original_label])
        mapped_labels.append(mapped)
    if not kept_coords:
        raise ValueError("No coordinates remain after class mapping/filtering")
    return np.asarray(kept_coords), np.asarray(mapped_labels, dtype=np.int64)


def _fit_patch_standardisation(hsi: np.ndarray, auxiliary: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "hsi_mean": hsi.mean(axis=(0, 1, 2), keepdims=True),
        "hsi_std": hsi.std(axis=(0, 1, 2), keepdims=True).clip(1e-6),
        "aux_mean": auxiliary.mean(axis=(0, 1, 2), keepdims=True),
        "aux_std": auxiliary.std(axis=(0, 1, 2), keepdims=True).clip(1e-6),
    }


def _apply_patch_standardisation(
    hsi: np.ndarray, auxiliary: np.ndarray, stats: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    hsi = (hsi - stats["hsi_mean"]) / stats["hsi_std"]
    auxiliary = (auxiliary - stats["aux_mean"]) / stats["aux_std"]
    return hsi.astype(np.float32), auxiliary.astype(np.float32)


class OVDGPatchDataset(Dataset):
    """Patch dataset carrying domain and sensor metadata with every sample."""

    def __init__(
        self,
        hsi: np.ndarray,
        auxiliary: np.ndarray,
        labels: np.ndarray,
        wavelengths: np.ndarray,
        domain_id: int,
    ):
        if not (len(hsi) == len(auxiliary) == len(labels)):
            raise ValueError("HSI, auxiliary data, and labels must have equal length")
        self.hsi = torch.from_numpy(np.ascontiguousarray(hsi, dtype=np.float32))
        self.auxiliary = torch.from_numpy(
            np.ascontiguousarray(auxiliary, dtype=np.float32)
        )
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.wavelengths = torch.from_numpy(np.asarray(wavelengths, dtype=np.float32))
        self.domain_id = int(domain_id)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        auxiliary = self.auxiliary[index].permute(2, 0, 1)
        return {
            "hsi": self.hsi[index],
            "auxiliary": auxiliary,
            "label": self.labels[index],
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "wavelengths": self.wavelengths,
        }


def pad_ovdg_batch(samples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad heterogeneous sensors to the largest band count in this batch."""
    max_bands = max(sample["hsi"].shape[-1] for sample in samples)
    hsi_batch, wavelength_batch, mask_batch = [], [], []
    for sample in samples:
        bands = sample["hsi"].shape[-1]
        padding = max_bands - bands
        hsi_batch.append(F_pad_last(sample["hsi"], padding))
        wavelength_batch.append(F_pad_last(sample["wavelengths"], padding))
        mask = torch.zeros(max_bands, dtype=torch.bool)
        mask[:bands] = True
        mask_batch.append(mask)
    return {
        "hsi": torch.stack(hsi_batch),
        "auxiliary": torch.stack([sample["auxiliary"] for sample in samples]),
        "label": torch.stack([sample["label"] for sample in samples]),
        "domain_id": torch.stack([sample["domain_id"] for sample in samples]),
        "wavelengths": torch.stack(wavelength_batch),
        "band_mask": torch.stack(mask_batch),
    }


def F_pad_last(tensor: torch.Tensor, padding: int) -> torch.Tensor:
    if padding <= 0:
        return tensor
    shape = list(tensor.shape)
    shape[-1] = padding
    return torch.cat([tensor, tensor.new_zeros(shape)], dim=-1)


def prepare_trento_ovdg_dataloaders(
    data_dir: str,
    prompt_bank: Dict,
    mapping_path: str,
    wavelength_path: Optional[str],
    patch_size: int = 9,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    domain_id: int = 0,
    seed: int = 42,
    augment: bool = True,
    allow_linear_debug_wavelengths: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build a leakage-aware Trento seen-only train / full-vocabulary test split."""
    hsi, auxiliary, gt = load_trento_data(data_dir)
    wavelengths = load_wavelengths(
        wavelength_path,
        expected_bands=hsi.shape[-1],
        allow_linear_debug=allow_linear_debug_wavelengths,
    )
    label_mapping = load_label_mapping(mapping_path, prompt_bank["class_to_index"])
    seen_mask = prompt_bank["seen_mask"].cpu().numpy().astype(bool)

    train_coords, test_coords = spatial_train_test_split(
        gt=gt,
        class_dict=get_class_dict(TRENTO_CONFIG["class_info"]),
        method="disjoint",
        patch_size=patch_size,
        seed=seed,
        min_test_ratio=0.2,
        min_test_samples=50,
    )
    source_coords, source_labels = _filter_and_remap_coords(
        train_coords, label_mapping, seen_mask=seen_mask
    )
    target_coords, target_labels = _filter_and_remap_coords(
        test_coords, label_mapping, seen_mask=None
    )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(source_coords))
    num_val = max(1, int(round(len(order) * val_ratio)))
    val_indices, train_indices = order[:num_val], order[num_val:]

    train_hsi, train_aux, _ = extract_patches_from_coords(
        hsi, auxiliary, source_coords[train_indices], patch_size
    )
    val_hsi, val_aux, _ = extract_patches_from_coords(
        hsi, auxiliary, source_coords[val_indices], patch_size
    )
    test_hsi, test_aux, _ = extract_patches_from_coords(
        hsi, auxiliary, target_coords, patch_size
    )
    train_labels = source_labels[train_indices]
    val_labels = source_labels[val_indices]

    stats = _fit_patch_standardisation(train_hsi, train_aux)
    train_hsi, train_aux = _apply_patch_standardisation(train_hsi, train_aux, stats)
    val_hsi, val_aux = _apply_patch_standardisation(val_hsi, val_aux, stats)
    test_hsi, test_aux = _apply_patch_standardisation(test_hsi, test_aux, stats)

    if augment:
        train_hsi, train_aux, train_labels = augment_paired_data(
            train_hsi,
            train_aux,
            train_labels,
            rotations=[90, 180, 270],
            flip_up_down=True,
            flip_left_right=True,
        )

    datasets = [
        OVDGPatchDataset(train_hsi, train_aux, train_labels, wavelengths, domain_id),
        OVDGPatchDataset(val_hsi, val_aux, val_labels, wavelengths, domain_id),
        OVDGPatchDataset(test_hsi, test_aux, target_labels, wavelengths, domain_id),
    ]
    return tuple(
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(index == 0),
            collate_fn=pad_ovdg_batch,
        )
        for index, dataset in enumerate(datasets)
    )

