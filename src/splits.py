"""
Split generation for BraTS-Path experiments.

Two regimes:
  A) random  — StratifiedKFold (leaky, reproduces reported 0.98 CV)
  B) grouped — StratifiedGroupKFold on Patient (honest, no slide/patient bleed)

The grouped regime asserts zero group overlap on every fold. Any violation aborts.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

log = logging.getLogger(__name__)

FoldList = List[Tuple[np.ndarray, np.ndarray]]


def make_splits(
    manifest: pd.DataFrame,
    n_folds: int,
    seed: int,
    regime: str,
) -> FoldList:
    """
    Returns list of (train_indices, val_indices) into manifest.index.

    regime: 'random' or 'grouped'
    """
    assert regime in ("random", "grouped"), f"Unknown regime: {regime!r}"

    y = manifest["label_idx"].values
    groups = manifest["patient"].values
    idx = np.arange(len(manifest))

    if regime == "random":
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = [(tr, va) for tr, va in skf.split(idx, y)]
        log.info("Random splits: %d folds (leaky — ignores patient group)", n_folds)

    else:  # grouped
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = []
        for fold, (tr, va) in enumerate(sgkf.split(idx, y, groups)):
            assert_no_group_overlap(manifest, tr, va, fold)
            splits.append((tr, va))
        log.info("Grouped splits: %d folds, zero patient overlap verified on all folds", n_folds)

    _log_split_stats(manifest, splits, regime)
    return splits


def assert_no_group_overlap(
    manifest: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold: int,
) -> None:
    """
    Hard assertion: no patient appears in both train and val.
    Aborts the run if violated — a silently wrong split is the worst outcome.
    """
    train_groups = set(manifest.iloc[train_idx]["patient"].unique())
    val_groups = set(manifest.iloc[val_idx]["patient"].unique())
    overlap = train_groups & val_groups
    if overlap:
        raise AssertionError(
            f"Fold {fold}: GROUP OVERLAP DETECTED — {len(overlap)} patient(s) "
            f"appear in both train and val: {sorted(overlap)[:10]}… "
            "This split is invalid. Aborting."
        )


def _log_split_stats(manifest: pd.DataFrame, splits: FoldList, regime: str) -> None:
    print(f"\n{'='*60}")
    print(f"SPLIT STATS — regime={regime!r}")
    print(f"{'='*60}")
    for fold, (tr, va) in enumerate(splits):
        tr_df = manifest.iloc[tr]
        va_df = manifest.iloc[va]
        print(
            f"  Fold {fold}: train={len(tr_df):,} patches / "
            f"{tr_df['patient'].nunique()} patients  |  "
            f"val={len(va_df):,} patches / {va_df['patient'].nunique()} patients"
        )
    print(f"{'='*60}\n")


def fold_site_distribution(manifest: pd.DataFrame, splits: FoldList) -> None:
    """
    Print institution distribution per fold if site data exists.
    Site data is NOT present in this dataset (metadata TSV = Synapse file records only).
    """
    if "site" not in manifest.columns:
        print(
            "\n[Site diagnostics] No site/institution column in manifest. "
            "The Synapse metadata only contains file-level records (shard filenames, sizes, md5). "
            "Site-per-fold analysis cannot be performed from available data.\n"
            "MANUAL TODO: Confirm on the Synapse portal whether the hidden test set "
            "comes from held-out institutions. This cannot be verified from the downloaded files.\n"
        )
        return

    print("\nSite distribution per fold (grouped split):")
    for fold, (tr, va) in enumerate(splits):
        va_df = manifest.iloc[va]
        dist = va_df["site"].value_counts().to_dict()
        print(f"  Fold {fold} val sites: {dist}")
