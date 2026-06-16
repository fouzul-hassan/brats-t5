"""
Unit tests for split assertion and manifest integrity.

Critical test: deliberately construct an overlapping split and assert the assertion fires.
Run: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.splits import assert_no_group_overlap, make_splits


def _fake_manifest(n_samples=100, n_patients=10, n_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    patients = rng.integers(0, n_patients, size=n_samples)
    labels = rng.integers(0, n_classes, size=n_samples)
    return pd.DataFrame({
        "key": [f"s_{i:06d}" for i in range(n_samples)],
        "label_idx": labels,
        "label_abbr": [str(l) for l in labels],
        "patient": patients,
        "slide": patients * 2,
        "shard_path": "/fake/shard.tar",
        "offset_jpg": rng.integers(0, 1_000_000, size=n_samples),
        "size_jpg": rng.integers(50_000, 150_000, size=n_samples),
        "split": "train",
    })


# ── core assertion test ────────────────────────────────────────────────────────

def test_overlap_assertion_fires_on_random_split():
    """
    A random (leaky) split WILL have patient overlap.
    assert_no_group_overlap must fire — this is the most important test.
    """
    df = _fake_manifest(n_samples=200, n_patients=5, n_classes=10, seed=42)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(df.index, df["label_idx"]))

    # At least one fold should have overlap when groups << samples
    overlap_found = False
    for fold_idx, (tr, va) in enumerate(splits):
        train_g = set(df.iloc[tr]["patient"])
        val_g = set(df.iloc[va]["patient"])
        if train_g & val_g:
            overlap_found = True
            with pytest.raises(AssertionError, match="GROUP OVERLAP DETECTED"):
                assert_no_group_overlap(df, tr, va, fold_idx)
    assert overlap_found, (
        "Expected at least one fold to have patient overlap with random split, "
        "but none was found — test setup is wrong."
    )


def test_grouped_split_has_no_overlap():
    """Grouped split must produce zero overlap on all folds."""
    df = _fake_manifest(n_samples=500, n_patients=20, n_classes=10, seed=7)
    splits = make_splits(df, n_folds=5, seed=42, regime="grouped")
    for fold_idx, (tr, va) in enumerate(splits):
        train_g = set(df.iloc[tr]["patient"])
        val_g = set(df.iloc[va]["patient"])
        assert train_g & val_g == set(), (
            f"Fold {fold_idx}: unexpected group overlap in grouped split!"
        )


def test_assert_no_group_overlap_clean():
    """assert_no_group_overlap should not raise when there is no overlap."""
    df = _fake_manifest(n_samples=100, n_patients=10)
    # construct clean split: patients 0-7 in train, 8-9 in val
    train_idx = df[df["patient"] < 8].index.values
    val_idx = df[df["patient"] >= 8].index.values
    assert_no_group_overlap(df, train_idx, val_idx, fold=0)  # should not raise


def test_assert_no_group_overlap_fires():
    """Manually construct an overlapping split and verify assertion fires."""
    df = _fake_manifest(n_samples=50, n_patients=5)
    # put patient 0 in both train and val
    train_idx = df[df["patient"].isin([0, 1, 2, 3])].index.values
    val_idx = df[df["patient"].isin([0, 4])].index.values  # patient 0 overlaps!
    with pytest.raises(AssertionError, match="GROUP OVERLAP DETECTED"):
        assert_no_group_overlap(df, train_idx, val_idx, fold=0)


# ── manifest integrity tests ───────────────────────────────────────────────────

def test_manifest_columns():
    df = _fake_manifest()
    required = ["key", "label_idx", "patient", "slide", "shard_path", "offset_jpg", "size_jpg"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_manifest_label_range():
    df = _fake_manifest(n_samples=500, n_classes=10)
    assert df["label_idx"].min() >= 0
    assert df["label_idx"].max() <= 9


def test_manifest_no_null_patient():
    df = _fake_manifest()
    assert df["patient"].isna().sum() == 0


def test_random_split_n_folds():
    df = _fake_manifest(n_samples=300, n_patients=30, n_classes=10)
    splits = make_splits(df, n_folds=5, seed=42, regime="random")
    assert len(splits) == 5
    for tr, va in splits:
        assert len(tr) + len(va) == len(df)
        assert len(set(tr) & set(va)) == 0  # index-level no overlap (not group-level)


def test_grouped_split_n_folds():
    df = _fake_manifest(n_samples=300, n_patients=30, n_classes=10)
    splits = make_splits(df, n_folds=5, seed=42, regime="grouped")
    assert len(splits) == 5


def test_grouped_split_covers_all_samples():
    df = _fake_manifest(n_samples=300, n_patients=30, n_classes=10)
    splits = make_splits(df, n_folds=5, seed=42, regime="grouped")
    all_val_idx = np.concatenate([va for _, va in splits])
    assert len(np.unique(all_val_idx)) == len(df), (
        "Grouped split val indices don't cover all samples exactly once."
    )


def test_invalid_regime_raises():
    df = _fake_manifest()
    with pytest.raises(AssertionError, match="Unknown regime"):
        make_splits(df, n_folds=5, seed=42, regime="bad_regime")
