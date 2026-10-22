"""Synthetic checks for seen/unseen-aware spatial splitting."""

import numpy as np

from open_set_data import seen_unseen_spatial_split


def _minimum_chebyshev(left, right):
    distances = np.maximum(
        np.abs(left[:, None, 0] - right[None, :, 0]),
        np.abs(left[:, None, 1] - right[None, :, 1]),
    )
    return int(distances.min())


def test_split_excludes_unknown_from_train_and_prevents_patch_overlap():
    gt = np.zeros((100, 120), dtype=np.int64)
    gt[10:40, 10:50] = 1
    gt[55:90, 10:50] = 2
    gt[10:40, 70:110] = 3
    train, val, test = seen_unseen_spatial_split(
        gt, [1, 2], [3], patch_size=5,
        train_per_raw_class=20, val_per_raw_class=10, test_per_raw_class=20,
    )
    assert set(np.unique(train[:, 2])) == {1, 2}
    assert set(np.unique(val[:, 2])) == {1, 2}
    assert set(np.unique(test[:, 2])) == {1, 2, 3}
    assert _minimum_chebyshev(train, val) >= 5
    assert _minimum_chebyshev(train, test) >= 5
    assert _minimum_chebyshev(val, test) >= 5


if __name__ == "__main__":
    test_split_excludes_unknown_from_train_and_prevents_patch_overlap()
    print("Open-Set data split tests passed")

