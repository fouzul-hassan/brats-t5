"""
Shard scanning, manifest building, and PyTorch Dataset for BraTS-Path 2026.

Key facts (verified by inspection):
- Each shard: {key}.jpg (512x512 RGB) + {key}.cls (2-byte ASCII int 0-9)
- Labels live in .cls sidecars — the mapping CSV has NO label column
- Mapping CSV columns: Name, Patient (126 unique), Slide (255 unique)
- Group key for honest CV = Patient
"""

import io
import json
import logging
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

log = logging.getLogger(__name__)

# Abbreviation → index from class_map.json (verified)
CLASS_ABBR_TO_IDX = {
    "CT": 0, "DM": 1, "IC": 2, "LI": 3, "MP": 4,
    "NC": 5, "PL": 6, "PN": 7, "WM": 8, "NOTA": 9,
}
IDX_TO_ABBR = {v: k for k, v in CLASS_ABBR_TO_IDX.items()}
NUM_CLASSES = 10


def scan_shard(shard_path: str) -> List[dict]:
    """
    Read one tar shard and return per-sample records.
    Only reads .cls content (2 bytes) and .jpg header metadata (offset + size).
    Does NOT decode image pixel data — fast even for 4 GB shards.
    """
    records: Dict[str, dict] = {}
    with tarfile.open(shard_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            basename = os.path.basename(member.name)
            if "." not in basename:
                continue
            key, ext = basename.split(".", 1)
            if ext == "jpg":
                records.setdefault(key, {})["offset_jpg"] = member.offset_data
                records[key]["size_jpg"] = member.size
                records[key]["shard_path"] = shard_path
            elif ext == "cls":
                f = tf.extractfile(member)
                label = int(f.read().decode().strip())
                records.setdefault(key, {})["label_idx"] = label

    # only return complete pairs
    complete = []
    for key, info in records.items():
        if "label_idx" in info and "offset_jpg" in info:
            complete.append({"key": key, **info})
        else:
            log.warning("Incomplete record for key %s in %s", key, shard_path)
    return complete


def build_manifest(cfg: dict, overwrite: bool = False) -> pd.DataFrame:
    """
    Scan all training shards, join with mapping CSV, write preprocessed/manifest.parquet.

    Manifest columns:
        key, label_idx, label_abbr, patient, slide, shard_path, offset_jpg, size_jpg, split
    """
    preprocessed_dir = Path(cfg["data"]["preprocessed_dir"])
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    out_path = preprocessed_dir / "manifest.parquet"

    if out_path.exists() and not overwrite:
        log.info("Manifest already exists at %s — loading.", out_path)
        return pd.read_parquet(out_path)

    labeled_dir = Path(cfg["data"]["labeled_dir"])
    shard_paths = sorted(glob(str(labeled_dir / "shard-*.tar")))
    assert len(shard_paths) > 0, f"No shards found in {labeled_dir}"
    log.info("Found %d shards in %s", len(shard_paths), labeled_dir)

    # parallel shard scan
    all_records = []
    n_workers = min(len(shard_paths), max(1, (os.cpu_count() or 4)))
    log.info("Scanning shards with %d workers …", n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(scan_shard, p): p for p in shard_paths}
        for i, f in enumerate(as_completed(futures), 1):
            recs = f.result()
            all_records.extend(recs)
            log.info("  [%d/%d] %s → %d patches", i, len(shard_paths),
                     os.path.basename(futures[f]), len(recs))

    df = pd.DataFrame(all_records)
    log.info("Total patches from shards: %d", len(df))

    # join with mapping CSV for Patient / Slide
    mapping_path = labeled_dir / "BraTS-Path-2026-Train-Patch-Patient-Slide-Mapping.csv"
    assert mapping_path.exists(), f"Mapping CSV not found: {mapping_path}"
    mapping = pd.read_csv(mapping_path)
    # columns: Name, Patient, Slide
    mapping = mapping.rename(columns={"Name": "key", "Patient": "patient", "Slide": "slide"})

    df = df.merge(mapping, on="key", how="left")

    # add label abbreviation
    df["label_abbr"] = df["label_idx"].map(IDX_TO_ABBR)
    df["split"] = "train"

    # --- hard assertions ---
    n_null_patient = df["patient"].isna().sum()
    assert n_null_patient == 0, (
        f"{n_null_patient} rows have null patient — mapping CSV incomplete. "
        "Cannot run grouped CV."
    )

    for cls_idx in range(NUM_CLASSES):
        cls_df = df[df["label_idx"] == cls_idx]
        n_groups = cls_df["patient"].nunique()
        assert n_groups > 1, (
            f"Class {IDX_TO_ABBR[cls_idx]} (idx={cls_idx}) has only {n_groups} patient group(s). "
            "Grouped CV is meaningless for this class."
        )

    # reorder columns
    df = df[["key", "label_idx", "label_abbr", "patient", "slide",
             "shard_path", "offset_jpg", "size_jpg", "split"]]
    df.to_parquet(out_path, index=False)
    log.info("Manifest written to %s (%d rows)", out_path, len(df))
    return df


def print_manifest_stats(df: pd.DataFrame) -> None:
    """Print dataset statistics after building manifest."""
    print("\n" + "=" * 60)
    print("MANIFEST STATISTICS")
    print("=" * 60)
    print(f"Total patches : {len(df):,}")
    print(f"Patients      : {df['patient'].nunique()}")
    print(f"Slides        : {df['slide'].nunique()}")
    print(f"Shards        : {df['shard_path'].nunique()}")
    print("\nClass distribution:")
    vc = df.groupby(["label_idx", "label_abbr"]).size().reset_index(name="count")
    vc["pct"] = vc["count"] / len(df) * 100
    for _, row in vc.sort_values("label_idx").iterrows():
        print(f"  [{int(row.label_idx)}] {row.label_abbr:6s}  {int(row['count']):>8,}  ({row.pct:.1f}%)")

    print("\nPatches per patient (group stats):")
    grp = df.groupby("patient").size()
    print(f"  min={grp.min()}  median={int(grp.median())}  "
          f"mean={grp.mean():.0f}  max={grp.max()}")

    print("\nNOTE: No site/institution data found in the Synapse metadata. "
          "Site-per-fold diagnostics cannot be performed.")
    print("=" * 60 + "\n")


def subsample_manifest(df: pd.DataFrame, max_per_class: int,
                       seed: int = 42) -> pd.DataFrame:
    """
    Subsample to max_per_class patches per class.
    Subsampling is group-aware: we drop whole patches, not whole patients,
    so the same patients still appear (class balance within patients may differ).
    Both split regimes get the SAME subset so Δ is valid.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for cls_idx in range(NUM_CLASSES):
        cls_df = df[df["label_idx"] == cls_idx]
        if len(cls_df) > max_per_class:
            idx = rng.choice(cls_df.index, size=max_per_class, replace=False)
            parts.append(cls_df.loc[idx])
        else:
            parts.append(cls_df)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset

    class BraTSDataset(Dataset):
        """
        Random-access dataset using direct byte-seek into tar shards.
        Avoids holding open file handles; safe with DataLoader multiprocessing.
        """

        def __init__(self, manifest_df: pd.DataFrame, transform=None):
            self.df = manifest_df.reset_index(drop=True)
            self.transform = transform

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int):
            row = self.df.iloc[idx]
            with open(row["shard_path"], "rb") as f:
                f.seek(int(row["offset_jpg"]))
                data = f.read(int(row["size_jpg"]))
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, int(row["label_idx"])

        def get_labels(self) -> np.ndarray:
            return self.df["label_idx"].values.astype(np.int64)

        def get_groups(self) -> np.ndarray:
            return self.df["patient"].values

    class ValDataset(Dataset):
        """
        Validation set — jpg only, no labels.
        Returns (image_tensor, key_str) for feature extraction / submission.
        """

        def __init__(self, val_dir: str, transform=None):
            val_dir = Path(val_dir)
            self.shard = val_dir / "val-shard-000000.tar"
            assert self.shard.exists(), f"Val shard not found: {self.shard}"
            self.transform = transform
            self._records = self._scan()

        def _scan(self) -> List[dict]:
            records = []
            with tarfile.open(self.shard, "r") as tf:
                for m in tf:
                    if m.isfile() and m.name.endswith(".jpg"):
                        key = os.path.basename(m.name).split(".", 1)[0]
                        records.append({
                            "key": key,
                            "offset": m.offset_data,
                            "size": m.size,
                        })
            return records

        def __len__(self) -> int:
            return len(self._records)

        def __getitem__(self, idx: int):
            r = self._records[idx]
            with open(self.shard, "rb") as f:
                f.seek(r["offset"])
                data = f.read(r["size"])
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, r["key"]

except ImportError:
    log.warning("torch not available — Dataset classes not registered.")


# ---------------------------------------------------------------------------
# Smoke-test helpers (no timm, no GPU required)
# ---------------------------------------------------------------------------

def make_smoke_dataset(out_dir: str, n_patches: int = 300, n_classes: int = 10,
                       n_patients: int = 12, seed: int = 42) -> Tuple[str, str]:
    """
    Generate tiny synthetic shards + mapping CSV for smoke testing.
    Returns (labeled_dir, preprocessed_dir).
    """
    import struct
    rng = np.random.default_rng(seed)
    labeled_dir = Path(out_dir) / "smoke_labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    # assign patients and slides
    patient_ids = (rng.integers(0, n_patients, size=n_patches)).tolist()
    slide_ids = [pid * 3 + rng.integers(0, 3) for pid in patient_ids]
    labels = (rng.integers(0, n_classes, size=n_patches)).tolist()
    keys = [f"smoke_{i:06d}" for i in range(n_patches)]

    # pack into 2 shards
    n_shards = 2
    mapping_rows = []
    for shard_idx in range(n_shards):
        start = shard_idx * (n_patches // n_shards)
        end = (shard_idx + 1) * (n_patches // n_shards) if shard_idx < n_shards - 1 else n_patches
        shard_path = labeled_dir / f"shard-{shard_idx:06d}.tar"
        with tarfile.open(shard_path, "w") as tf:
            for i in range(start, end):
                # tiny synthetic JPEG via PIL
                arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(arr, mode="RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                jpg_bytes = buf.getvalue()

                cls_bytes = str(labels[i]).encode()
                key = keys[i]

                for ext, data in [("jpg", jpg_bytes), ("cls", cls_bytes)]:
                    info = tarfile.TarInfo(name=f"{key}.{ext}")
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))

                mapping_rows.append({
                    "Name": key,
                    "Patient": patient_ids[i],
                    "Slide": slide_ids[i],
                })

    # mapping CSV
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(
        labeled_dir / "BraTS-Path-2026-Train-Patch-Patient-Slide-Mapping.csv",
        index=False
    )

    # class_map.json
    class_map = {k: v for k, v in CLASS_ABBR_TO_IDX.items()}
    with open(labeled_dir / "class_map.json", "w") as fp:
        json.dump(class_map, fp)

    log.info("Smoke dataset: %d patches, %d shards, %d patients → %s",
             n_patches, n_shards, n_patients, labeled_dir)
    return str(labeled_dir)
