#!/usr/bin/env python3
"""
BraTS-Path 2026 — Leakage & Foundation Model De-risking Pipeline

Subcommands:
  inspect     Print data format details (run first, always)
  preprocess  Build preprocessed/manifest.parquet from all shards
  exp1        Experiment 1: leakage quantification (random vs grouped CV)
  exp2        Experiment 2: frozen foundation model linear probe
  valset      Extract features from official val set (no labels — submission only)
  smoke       End-to-end test on synthetic data, CPU, <2 min
  all         Run: inspect → preprocess → exp1 → exp2 → valset → report

Usage:
  python run.py smoke
  python run.py inspect --config configs/default.yaml
  python run.py preprocess --config configs/default.yaml
  python run.py exp1 --config configs/default.yaml
  python run.py exp2 --config configs/default.yaml
  python run.py all --config configs/default.yaml
"""

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# ── logging setup ─────────────────────────────────────────────────────────────

def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )
    logging.info("Log: %s", log_path)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_cfg(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── subcommands ────────────────────────────────────────────────────────────────

def cmd_inspect(cfg: dict, run_dir: Path) -> None:
    from src.inspect_data import run_inspect
    run_inspect(cfg)


def cmd_preprocess(cfg: dict, run_dir: Path, overwrite: bool = False) -> None:
    from src.data import build_manifest, print_manifest_stats, subsample_manifest
    df = build_manifest(cfg, overwrite=overwrite)
    print_manifest_stats(df)

    max_per_class = cfg["data"].get("max_patches_per_class")
    if max_per_class:
        df_sub = subsample_manifest(df, max_per_class, seed=cfg["seed"])
        logging.info(
            "Subsampled to %d patches (%d per class max)", len(df_sub), max_per_class
        )
    return df


def cmd_exp1(cfg: dict, run_dir: Path, smoke: bool = False) -> dict:
    import pandas as pd
    from src.data import build_manifest, subsample_manifest
    from src.splits import make_splits, fold_site_distribution
    from src.baseline import run_experiment1, summarise_regime
    from src.probe import print_leakage_table

    if smoke:
        df = pd.read_parquet(Path(cfg["data"]["preprocessed_dir"]) / "manifest.parquet")
    else:
        df = build_manifest(cfg)
        max_per_class = cfg["data"].get("max_patches_per_class")
        if max_per_class:
            df = subsample_manifest(df, max_per_class, seed=cfg["seed"])

    logging.info("Manifest: %d patches, %d patients", len(df), df["patient"].nunique())

    splits_A = make_splits(df, cfg["n_folds"], cfg["seed"], regime="random")
    splits_B = make_splits(df, cfg["n_folds"], cfg["seed"], regime="grouped")
    fold_site_distribution(df, splits_B)

    results_A, results_B = run_experiment1(df, splits_A, splits_B, cfg, run_dir, smoke=smoke)

    sum_A = summarise_regime(results_A, "Regime A — Random / Leaky")
    sum_B = summarise_regime(results_B, "Regime B — Grouped / Honest")
    print_leakage_table(sum_A, sum_B)

    return {"sum_A": sum_A, "sum_B": sum_B, "results_A": results_A, "results_B": results_B}


def cmd_exp2(cfg: dict, run_dir: Path, smoke: bool = False) -> dict:
    import pandas as pd
    from src.data import build_manifest, subsample_manifest
    from src.splits import make_splits
    from src.features import extract_features, load_features
    from src.probe import run_probe, summarise_probe

    if smoke:
        df = pd.read_parquet(Path(cfg["data"]["preprocessed_dir"]) / "manifest.parquet")
        model_names = ["smoke"]
    else:
        df = build_manifest(cfg)
        max_per_class = cfg["data"].get("max_patches_per_class")
        if max_per_class:
            df = subsample_manifest(df, max_per_class, seed=cfg["seed"])
        model_names = cfg["features"]["models"]

    splits_B = make_splits(df, cfg["n_folds"], cfg["seed"], regime="grouped")

    all_probe_results = {}
    for model_name in model_names:
        logging.info("\n%s\nEXP2 — model=%s\n%s", "=" * 60, model_name, "=" * 60)
        ok = extract_features(df, model_name, cfg, run_dir=run_dir, smoke=smoke)
        if not ok:
            logging.warning("Skipping probe for %s — feature extraction failed (access denied?)", model_name)
            continue

        X, y, groups = load_features(model_name, df, cfg)
        results = run_probe(model_name, X, y, groups, splits_B, cfg, run_dir, smoke=smoke)
        summary = summarise_probe(results, model_name)
        all_probe_results[model_name] = summary

    return all_probe_results


def cmd_valset(cfg: dict, run_dir: Path, smoke: bool = False) -> None:
    """
    Extract features from the official validation set.
    NO labels → no score is computed. Features are cached for later submission.
    """
    if smoke:
        logging.info("[valset] Skipped in smoke mode (no synthetic val set).")
        return

    from src.data import ValDataset
    from src.features import load_model_for_extraction, HF_URLS
    import torch
    from torch.utils.data import DataLoader
    import numpy as np

    val_dir = cfg["data"]["val_dir"]
    preprocessed_dir = Path(cfg["data"]["preprocessed_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Official val set: %s (LABELS NOT AVAILABLE — submission only)", val_dir)

    for model_name in cfg["features"]["models"]:
        feat_dir = preprocessed_dir / "features" / f"val_{model_name}"
        feat_dir.mkdir(parents=True, exist_ok=True)
        out_npz = feat_dir / "val_features.npz"
        if out_npz.exists():
            logging.info("Val features for %s already cached at %s", model_name, out_npz)
            continue

        model, transform, embed_fn = load_model_for_extraction(model_name, device, cfg)
        if model is None:
            continue

        ds = ValDataset(val_dir, transform=transform)
        loader = DataLoader(ds, batch_size=cfg["features"]["batch_size"],
                            shuffle=False, num_workers=cfg["features"]["num_workers"])
        all_feats, all_keys = [], []
        use_fp16 = cfg["features"].get("fp16", True)
        with torch.inference_mode():
            ctx = (torch.autocast("cuda", dtype=torch.float16)
                   if use_fp16 and device.type == "cuda" else _null_ctx())
            with ctx:
                for xb, keys in loader:
                    feats = embed_fn(model, xb.to(device))
                    all_feats.append(feats.float().cpu().numpy())
                    all_keys.extend(keys)

        feats_arr = np.concatenate(all_feats, axis=0).astype(np.float32)
        np.savez_compressed(out_npz, keys=np.array(all_keys), feats=feats_arr)
        logging.info("Val features cached: %s  shape=%s", out_npz, feats_arr.shape)

    logging.info(
        "\nValidation set result: NO LABELS — cannot evaluate models on val set. "
        "Features are cached at brats/preprocessed/features/val_*/ for submission use."
    )


def cmd_smoke(base_dir: str) -> None:
    """
    End-to-end smoke test on synthetic data.
    CPU-only, <2 min, no HF access, no timm pretrained weights required.
    Emits REPORT.md with clearly-marked synthetic numbers.
    """
    import tempfile
    import shutil
    import pandas as pd
    from src.data import make_smoke_dataset, build_manifest, print_manifest_stats
    from src.splits import make_splits, fold_site_distribution
    from src.baseline import run_experiment1, summarise_regime
    from src.probe import print_leakage_table, run_probe, summarise_probe
    from src.features import extract_features, load_features
    from src.report import generate_report

    t_start = time.time()
    print("\n" + "=" * 70)
    print("SMOKE TEST — synthetic data, CPU, no real model weights")
    print("=" * 70 + "\n")

    smoke_root = Path(base_dir) / "smoke_run"
    smoke_root.mkdir(parents=True, exist_ok=True)
    smoke_prep = smoke_root / "preprocessed"
    smoke_prep.mkdir(exist_ok=True)

    # build synthetic config
    labeled_dir = make_smoke_dataset(str(smoke_root), n_patches=300, n_patients=12, seed=42)
    cfg = {
        "seed": 42,
        "n_folds": 2,  # fewer folds for speed
        "group_col": "patient",
        "data": {
            "labeled_dir": labeled_dir,
            "val_dir": "/dev/null",  # no val set in smoke
            "preprocessed_dir": str(smoke_prep),
            "max_patches_per_class": None,
        },
        "output_dir": str(smoke_root / "runs"),
        "image_size": 32,  # tiny for speed
        "train": {
            "model": "efficientnet_b0",
            "pretrained": False,
            "epochs": 2,
            "batch_size": 16,
            "lr": 1e-3,
            "weight_decay": 0.01,
            "early_stop_patience": 1,
            "num_workers": 0,
            "label_smoothing": 0.0,
        },
        "features": {
            "models": ["smoke"],
            "batch_size": 16,
            "num_workers": 0,
            "fp16": False,
        },
        "probe": {"C": 1.0, "max_iter": 50, "tol": 1e-3, "solver": "lbfgs"},
    }

    run_dir = Path(cfg["output_dir"]) / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)
    set_seeds(42)

    # preprocess
    df = build_manifest(cfg)
    print_manifest_stats(df)

    # splits
    splits_A = make_splits(df, cfg["n_folds"], cfg["seed"], regime="random")
    splits_B = make_splits(df, cfg["n_folds"], cfg["seed"], regime="grouped")
    fold_site_distribution(df, splits_B)

    # exp1
    results_A, results_B = run_experiment1(df, splits_A, splits_B, cfg, run_dir, smoke=True)
    sum_A = summarise_regime(results_A, "Regime A — Random / Leaky [SYNTHETIC]")
    sum_B = summarise_regime(results_B, "Regime B — Grouped / Honest [SYNTHETIC]")
    print_leakage_table(sum_A, sum_B)

    # exp2
    ok = extract_features(df, "smoke", cfg, run_dir=run_dir, smoke=True)
    assert ok, "Smoke feature extraction failed"
    X, y, groups = load_features("smoke", df, cfg)
    probe_results = run_probe("smoke", X, y, groups, splits_B, cfg, run_dir, smoke=True)
    summarise_probe(probe_results, "smoke")

    # report
    manifest_stats = {
        "n_patches": len(df),
        "n_patients": df["patient"].nunique(),
        "n_slides": df["slide"].nunique(),
        "n_shards": df["shard_path"].nunique(),
        "class_counts": df["label_idx"].value_counts().to_dict(),
    }
    report_path = generate_report(run_dir, cfg, smoke=True, manifest_stats=manifest_stats)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"SMOKE TEST PASSED in {elapsed:.1f}s")
    print(f"REPORT.md → {report_path}")
    print(f"{'='*70}\n")


def cmd_all(cfg: dict, run_dir: Path) -> None:
    from src.report import generate_report
    from src.data import build_manifest
    import pandas as pd

    cmd_inspect(cfg, run_dir)
    df = cmd_preprocess(cfg, run_dir)
    if not isinstance(df, pd.DataFrame):
        df = build_manifest(cfg)

    exp1_results = cmd_exp1(cfg, run_dir)
    exp2_results = cmd_exp2(cfg, run_dir)
    cmd_valset(cfg, run_dir)

    manifest_stats = {
        "n_patches": len(df),
        "n_patients": df["patient"].nunique(),
        "n_slides": df["slide"].nunique(),
        "n_shards": df["shard_path"].nunique(),
        "class_counts": df["label_idx"].value_counts().to_dict(),
    }
    generate_report(run_dir, cfg, smoke=False, manifest_stats=manifest_stats)
    logging.info("All done. See REPORT.md")


# ── main ───────────────────────────────────────────────────────────────────────

class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


def main():
    parser = argparse.ArgumentParser(description="BraTS-Path 2026 pipeline")
    parser.add_argument("command", choices=["inspect", "preprocess", "exp1", "exp2",
                                             "valset", "smoke", "all"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing manifest/features")
    parser.add_argument("--base-dir", default="/shared/users/hassan2/brats-t5",
                        help="Base dir for smoke test outputs")
    args = parser.parse_args()

    if args.command == "smoke":
        cmd_smoke(args.base_dir)
        return

    cfg = load_cfg(args.config)
    seed = cfg.get("seed", 42)
    set_seeds(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["output_dir"]) / f"{args.command}_{timestamp}"
    setup_logging(run_dir)

    logging.info("Command: %s | Config: %s | Seed: %d", args.command, args.config, seed)
    logging.info("Run dir: %s", run_dir)

    dispatch = {
        "inspect": lambda: cmd_inspect(cfg, run_dir),
        "preprocess": lambda: cmd_preprocess(cfg, run_dir, overwrite=args.overwrite),
        "exp1": lambda: cmd_exp1(cfg, run_dir),
        "exp2": lambda: cmd_exp2(cfg, run_dir),
        "valset": lambda: cmd_valset(cfg, run_dir),
        "all": lambda: cmd_all(cfg, run_dir),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
