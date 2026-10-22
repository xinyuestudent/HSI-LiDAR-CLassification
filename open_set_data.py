"""Seen/unseen-aware spatial protocol for Trento open-set experiments."""

import json
from typing import Dict, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from dataset import augment_paired_data, extract_patches_from_coords, load_trento_data
from ovdg_data import (
    OVDGPatchDataset,
    _apply_patch_standardisation,
    _fit_patch_standardisation,
    load_wavelengths,
    pad_ovdg_batch,
)


def load_open_set_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    known = {int(raw): int(mapped) for raw, mapped in config["known_raw_to_index"].items()}
    unknown = [int(value) for value in config["unknown_raw_labels"]]
    num_known = len(config["known_class_names"])
    if set(known.values()) != set(range(num_known)):
        raise ValueError("known_raw_to_index must cover contiguous indices [0, K)")
    if set(known).intersection(unknown):
        raise ValueError("A raw class cannot be both known and unknown")
    config["known_raw_to_index"] = known
    config["unknown_raw_labels"] = unknown
    config["num_known_classes"] = num_known
    return config


def _valid_coords(gt: np.ndarray, raw_label: int, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    coords = np.argwhere(gt == raw_label)
    if not len(coords):
        return coords
    mask = (
        (coords[:, 0] >= half) & (coords[:, 0] < gt.shape[0] - half)
        & (coords[:, 1] >= half) & (coords[:, 1] < gt.shape[1] - half)
    )
    return coords[mask]


def _compact_edge_subset(coords: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Choose a compact spatial tail, avoiding random pixel leakage."""
    count = min(max(int(count), 0), len(coords))
    if count == 0:
        return coords[:0]
    candidates = []
    jitter = rng.random(len(coords)) * 1e-6
    for axis in (0, 1):
        for sign in (-1.0, 1.0):
            order = np.argsort(sign * coords[:, axis] + jitter)
            subset = coords[order[:count]]
            extent = np.ptp(subset, axis=0) + 1
            candidates.append((int(extent[0] * extent[1]), subset))
    return min(candidates, key=lambda item: item[0])[1]


def _blocked(mask: np.ndarray, coords: np.ndarray, radius: int):
    for row, col in coords:
        mask[
            max(0, row - radius): min(mask.shape[0], row + radius + 1),
            max(0, col - radius): min(mask.shape[1], col + radius + 1),
        ] = True


def _with_label(coords: np.ndarray, raw_label: int) -> np.ndarray:
    if not len(coords):
        return np.empty((0, 3), dtype=np.int64)
    return np.column_stack([coords, np.full(len(coords), raw_label, dtype=np.int64)])


def seen_unseen_spatial_split(
    gt: np.ndarray,
    known_raw_labels,
    unknown_raw_labels,
    patch_size: int = 9,
    train_per_raw_class: int = 125,
    val_per_raw_class: int = 50,
    test_per_raw_class: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Known-only train/val and known+unknown test with disjoint patch support."""
    rng = np.random.default_rng(seed)
    known_raw_labels = [int(x) for x in known_raw_labels]
    unknown_raw_labels = [int(x) for x in unknown_raw_labels]
    all_labels = known_raw_labels + unknown_raw_labels
    radius = patch_size - 1  # centre distance required for two patches not to overlap

    test_parts = []
    for raw_label in all_labels:
        coords = _valid_coords(gt, raw_label, patch_size)
        selected = _compact_edge_subset(coords, test_per_raw_class, rng)
        test_parts.append(_with_label(selected, raw_label))
    test = np.concatenate(test_parts) if test_parts else np.empty((0, 3), dtype=np.int64)

    test_blocked = np.zeros(gt.shape, dtype=bool)
    _blocked(test_blocked, test[:, :2], radius)
    val_parts = []
    for raw_label in known_raw_labels:
        coords = _valid_coords(gt, raw_label, patch_size)
        coords = coords[~test_blocked[coords[:, 0], coords[:, 1]]]
        selected = _compact_edge_subset(coords, val_per_raw_class, rng)
        val_parts.append(_with_label(selected, raw_label))
    val = np.concatenate(val_parts) if val_parts else np.empty((0, 3), dtype=np.int64)

    train_blocked = test_blocked.copy()
    _blocked(train_blocked, val[:, :2], radius)
    train_parts = []
    for raw_label in known_raw_labels:
        coords = _valid_coords(gt, raw_label, patch_size)
        coords = coords[~train_blocked[coords[:, 0], coords[:, 1]]]
        if len(coords) > train_per_raw_class:
            coords = coords[rng.choice(len(coords), train_per_raw_class, replace=False)]
        train_parts.append(_with_label(coords, raw_label))
    train = np.concatenate(train_parts) if train_parts else np.empty((0, 3), dtype=np.int64)

    for name, split in (("train", train), ("validation", val), ("test", test)):
        if not len(split):
            raise ValueError("{} split is empty; reduce per-class counts".format(name))
    return train, val, test


def _map_labels(coords: np.ndarray, config: Dict, allow_unknown: bool) -> np.ndarray:
    unknown_index = config["num_known_classes"]
    mapped = []
    for raw in coords[:, 2].astype(int):
        if raw in config["known_raw_to_index"]:
            mapped.append(config["known_raw_to_index"][raw])
        elif allow_unknown and raw in config["unknown_raw_labels"]:
            mapped.append(unknown_index)
        else:
            raise ValueError("Raw label {} is invalid in this split".format(raw))
    return np.asarray(mapped, dtype=np.int64)


def prepare_trento_open_set_dataloaders(
    data_dir: str,
    config_path: str,
    wavelength_path: Optional[str],
    patch_size: int = 9,
    batch_size: int = 32,
    seed: int = 42,
    augment: bool = True,
    train_per_raw_class: int = 125,
    val_per_raw_class: int = 50,
    test_per_raw_class: int = 500,
    allow_linear_debug_wavelengths: bool = False,
):
    hsi, auxiliary, gt = load_trento_data(data_dir)
    config = load_open_set_config(config_path)
    wavelengths = load_wavelengths(
        wavelength_path, hsi.shape[-1], allow_linear_debug_wavelengths
    )
    train_coords, val_coords, test_coords = seen_unseen_spatial_split(
        gt,
        config["known_raw_to_index"].keys(),
        config["unknown_raw_labels"],
        patch_size,
        train_per_raw_class,
        val_per_raw_class,
        test_per_raw_class,
        seed,
    )

    train_hsi, train_aux, _ = extract_patches_from_coords(hsi, auxiliary, train_coords, patch_size)
    val_hsi, val_aux, _ = extract_patches_from_coords(hsi, auxiliary, val_coords, patch_size)
    test_hsi, test_aux, _ = extract_patches_from_coords(hsi, auxiliary, test_coords, patch_size)
    train_labels = _map_labels(train_coords, config, False)
    val_labels = _map_labels(val_coords, config, False)
    test_labels = _map_labels(test_coords, config, True)

    stats = _fit_patch_standardisation(train_hsi, train_aux)
    train_hsi, train_aux = _apply_patch_standardisation(train_hsi, train_aux, stats)
    val_hsi, val_aux = _apply_patch_standardisation(val_hsi, val_aux, stats)
    test_hsi, test_aux = _apply_patch_standardisation(test_hsi, test_aux, stats)
    if augment:
        train_hsi, train_aux, train_labels = augment_paired_data(
            train_hsi, train_aux, train_labels,
            rotations=[90, 180, 270], flip_up_down=True, flip_left_right=True,
        )

    datasets = (
        OVDGPatchDataset(train_hsi, train_aux, train_labels, wavelengths, 0),
        OVDGPatchDataset(val_hsi, val_aux, val_labels, wavelengths, 0),
        OVDGPatchDataset(test_hsi, test_aux, test_labels, wavelengths, 0),
    )
    loaders = tuple(
        DataLoader(dataset, batch_size=batch_size, shuffle=index == 0, collate_fn=pad_ovdg_batch)
        for index, dataset in enumerate(datasets)
    )
    split_summary = {
        name: {str(int(label)): int(np.sum(coords[:, 2] == label)) for label in np.unique(coords[:, 2])}
        for name, coords in (("train", train_coords), ("validation", val_coords), ("test", test_coords))
    }
    return loaders[0], loaders[1], loaders[2], config, split_summary

