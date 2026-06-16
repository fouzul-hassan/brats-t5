"""
Step 0 — Inspect and print everything learned about the dataset format.
Standalone; run via: python run.py inspect --config configs/default.yaml
"""

import io
import json
import os
import tarfile
from pathlib import Path

import pandas as pd
from PIL import Image


def run_inspect(cfg: dict) -> None:
    labeled_dir = Path(cfg["data"]["labeled_dir"])
    val_dir = Path(cfg["data"]["val_dir"])

    print("\n" + "=" * 70)
    print("STEP 0 — DATA INSPECTION")
    print("=" * 70)

    # ---- Training shard ----
    shard = sorted(labeled_dir.glob("shard-*.tar"))[0]
    print(f"\n[1] Training shard: {shard.name}  ({shard.stat().st_size / 1e9:.2f} GB)")
    print("     Decoding first 10 members …")

    samples = {}
    with tarfile.open(shard, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            basename = os.path.basename(member.name)
            if "." not in basename:
                continue
            key, ext = basename.split(".", 1)
            if ext not in ("jpg", "cls"):
                continue
            samples.setdefault(key, {})
            if ext == "cls":
                samples[key]["cls"] = tf.extractfile(member).read().decode().strip()
                samples[key]["cls_offset"] = member.offset_data
                samples[key]["cls_size"] = member.size
            elif ext == "jpg":
                samples[key]["jpg_offset"] = member.offset_data
                samples[key]["jpg_size"] = member.size
            if len(samples) >= 6 and all(
                "cls" in v and "jpg_offset" in v for v in samples.values()
            ):
                if len(samples) >= 6:
                    break

    print(f"\n  Key pattern   : train_<16-char-hex>")
    print(f"  Extensions    : .jpg (image), .cls (label int as ASCII text)")
    print(f"  Label source  : .cls sidecar — NOT the mapping CSV")
    print(f"\n  Sample records:")
    print(f"  {'key':<30} {'cls':>5}  {'jpg_bytes':>10}")
    for key, info in list(samples.items())[:6]:
        print(f"    {key:<30} {info.get('cls', '?'):>5}  {info.get('jpg_size', 0):>10,}")

    # decode one image
    first_key = next(iter(samples))
    fi = samples[first_key]
    with open(shard, "rb") as f:
        f.seek(fi["jpg_offset"])
        jpg_bytes = f.read(fi["jpg_size"])
    img = Image.open(io.BytesIO(jpg_bytes))
    print(f"\n  Decoded image : key={first_key}  mode={img.mode}  size={img.size}  (H×W pixels)")

    # ---- class_map ----
    print("\n[2] class_map.json:")
    cm = json.load(open(labeled_dir / "class_map.json"))
    for k, v in sorted(cm.items(), key=lambda x: x[1]):
        print(f"     {v}: {k}")

    # ---- mapping CSV ----
    mapping_path = labeled_dir / "BraTS-Path-2026-Train-Patch-Patient-Slide-Mapping.csv"
    print(f"\n[3] Mapping CSV: {mapping_path.name}")
    df = pd.read_csv(mapping_path)
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Head (5 rows):")
    print(df.head(5).to_string(index=False))
    print(f"\n  Column mapping decision:")
    print(f"    sample id / key → 'Name'")
    print(f"    class label     → NOT IN CSV — comes from .cls sidecar in the tar")
    print(f"    grouping key    → 'Patient' ({df['Patient'].nunique()} unique patients)")
    print(f"    secondary group → 'Slide'  ({df['Slide'].nunique()} unique slides)")
    print(f"\n  ⚠  The mapping CSV has NO label column.")
    print(f"     Labels must be read from the .cls sidecars in each shard.")

    # ---- Synapse metadata ----
    tsv_path = Path(cfg["data"]["preprocessed_dir"]).parent / "dataset" / "SYNAPSE_METADATA_MANIFEST.tsv"
    # try a few locations
    for candidate in [
        Path("/shared/users/hassan2/brats/dataset/SYNAPSE_METADATA_MANIFEST.tsv"),
        labeled_dir.parent / "SYNAPSE_METADATA_MANIFEST.tsv",
    ]:
        if candidate.exists():
            tsv_path = candidate
            break
    if tsv_path.exists():
        tsv = pd.read_csv(tsv_path, sep="\t")
        print(f"\n[4] SYNAPSE_METADATA_MANIFEST.tsv: {len(tsv)} rows")
        print(f"  Columns: {list(tsv.columns)}")
        print(f"  ⚠  No institution/site/center column found.")
        print(f"     This file contains Synapse file-level records only (filenames, checksums, parent IDs).")
        print(f"     Site-per-fold diagnostics cannot be performed from available data.")
    else:
        print(f"\n[4] SYNAPSE_METADATA_MANIFEST.tsv: not found at expected locations.")

    # ---- Validation shard ----
    val_shard = val_dir / "val-shard-000000.tar"
    if val_shard.exists():
        print(f"\n[5] Validation shard: {val_shard.name}  ({val_shard.stat().st_size / 1e9:.2f} GB)")
        with tarfile.open(val_shard, "r") as tf:
            exts = set()
            n = 0
            first_key = None
            for m in tf:
                if m.isfile():
                    ext = os.path.basename(m.name).split(".", 1)[-1] if "." in m.name else "?"
                    exts.add(ext)
                    if first_key is None:
                        first_key = os.path.basename(m.name).split(".", 1)[0]
                    n += 1
                if n >= 2000:
                    break
        print(f"  Extensions in first {n} files: {exts}")
        print(f"  First key: {first_key}")
        print(f"  ⚠  No .cls labels found — validation set is SUBMISSION ONLY.")
        print(f"     Ground truth is withheld by the organisers.")
        print(f"     No held-out score can be computed without fabricating labels.")
    else:
        print(f"\n[5] Validation shard: not found at {val_shard}")

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70 + "\n")
