"""
Frozen foundation-model feature extraction.

Models:
  uni2h    — MahmoodLab/UNI2-h  (ViT-H/14, 1536-d)
  virchow2 — paige-ai/Virchow2  (ViT-H/14, 2560-d = CLS + mean(patch tokens))

Features cached per shard as NPZ:
  preprocessed/features/{model_name}/shard_{i:06d}.npz
  Each NPZ: keys=np.array([str,...]), feats=np.array([N, D], float32)

On 401/403 from HuggingFace: prints the access URL and exits — never prompts.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HF_URLS = {
    "uni2h": "https://huggingface.co/MahmoodLab/UNI2-h",
    "virchow2": "https://huggingface.co/paige-ai/Virchow2",
}

UNI2H_DIM = 1536
VIRCHOW2_DIM = 2560
SMOKE_DIM = 64


def check_hf_access(model_name: str) -> bool:
    """Returns True if model is accessible. On 401/403, prints URL and returns False."""
    repo_map = {
        "uni2h": "MahmoodLab/UNI2-h",
        "virchow2": "paige-ai/Virchow2",
    }
    repo = repo_map.get(model_name)
    if repo is None:
        return True  # smoke / unknown — no HF check needed

    try:
        from huggingface_hub import model_info
        model_info(repo)
        log.info("HF access OK for %s", repo)
        return True
    except Exception as e:
        err = str(e)
        if any(code in err for code in ("401", "403", "GatedRepo", "gated", "access")):
            print(f"\n{'='*60}")
            print(f"ACCESS DENIED — {model_name.upper()} model is gated.")
            print(f"Request access at: {HF_URLS[model_name]}")
            print("Then re-run with: python run.py exp2 (after huggingface-cli login)")
            print(f"{'='*60}\n")
            return False
        raise


def load_uni2h(device) -> Tuple:
    """Load UNI2-h exactly as specified in the brief."""
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    timm_kwargs = {
        "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
        "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667 * 2,
        "num_classes": 0, "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU,
        "reg_tokens": 8, "dynamic_img_size": True,
    }
    log.info("Loading UNI2-h from HF hub …")
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
    ).eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    log.info("UNI2-h loaded. Embedding dim: %d", UNI2H_DIM)
    return model, transform


def load_uni_v1(device) -> Tuple:
    """Fallback UNI v1 (ViT-L/16, 1024-d) if UNI2-h is not accessible."""
    import timm
    import torch
    from huggingface_hub import hf_hub_download
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    log.info("Loading UNI v1 (fallback) …")
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16,
        init_values=1e-5, num_classes=0, dynamic_img_size=True,
    ).eval().to(device)
    weights_path = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin")
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    log.info("UNI v1 loaded. Embedding dim: 1024")
    return model, transform


def load_virchow2(device) -> Tuple:
    """Load Virchow2 exactly as specified in the brief."""
    import timm
    import torch
    from timm.layers import SwiGLUPacked
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    log.info("Loading Virchow2 from HF hub …")
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2", pretrained=True,
        mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
    ).eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    log.info("Virchow2 loaded. Embedding dim: %d", VIRCHOW2_DIM)
    return model, transform


def _embed_virchow2(model, image_batch) -> "torch.Tensor":
    """Virchow2 embedding = concat(CLS, mean(patch_tokens))."""
    import torch
    output = model(image_batch)          # [B, 261, 1280]
    cls = output[:, 0]                   # [B, 1280]  — token 0
    patch = output[:, 5:]               # [B, 256, 1280] — drop CLS + 4 register tokens
    return torch.cat([cls, patch.mean(1)], dim=-1)  # [B, 2560]


def load_model_for_extraction(model_name: str, device, cfg: dict) -> Tuple:
    """
    Load the backbone and its transform.
    Returns (model, transform, embed_fn) where embed_fn(model, batch) → tensor.
    """
    if model_name == "smoke":
        return _smoke_model()

    if model_name == "uni2h":
        if not check_hf_access("uni2h"):
            return None, None, None
        try:
            model, transform = load_uni2h(device)
        except Exception as e:
            if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                print(f"\nACCESS DENIED. Request access at: {HF_URLS['uni2h']}\n")
                return None, None, None
            # try fallback v1
            log.warning("UNI2-h failed (%s) — trying fallback UNI v1", e)
            if not check_hf_access("uni2h"):
                return None, None, None
            model, transform = load_uni_v1(device)
        embed_fn = lambda m, x: m(x)
        return model, transform, embed_fn

    if model_name == "virchow2":
        if not check_hf_access("virchow2"):
            return None, None, None
        try:
            model, transform = load_virchow2(device)
        except Exception as e:
            if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                print(f"\nACCESS DENIED. Request access at: {HF_URLS['virchow2']}\n")
                return None, None, None
            raise
        embed_fn = _embed_virchow2
        return model, transform, embed_fn

    raise ValueError(f"Unknown model: {model_name!r}")


def _smoke_model():
    """Random 64-d embedding stub for smoke test — no GPU, no HF, no torchvision."""
    import torch
    import torch.nn as nn
    import numpy as np
    from PIL import Image as PILImage

    class _RandomEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(3, SMOKE_DIM, bias=False)
        def forward(self, x):
            pooled = self.pool(x).squeeze(-1).squeeze(-1)
            return self.proj(pooled)

    _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _transform(img):
        img = img.resize((224, 224), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _mean) / _std
        return torch.from_numpy(arr.transpose(2, 0, 1))

    model = _RandomEmbed().eval()
    embed_fn = lambda m, x: m(x)
    return model, _transform, embed_fn


def shard_npz_path(feat_dir: Path, shard_path: str) -> Path:
    stem = Path(shard_path).stem  # e.g. shard-000000
    return feat_dir / f"{stem}.npz"


def extract_features(
    manifest: pd.DataFrame,
    model_name: str,
    cfg: dict,
    run_dir: Optional[Path] = None,
    smoke: bool = False,
) -> bool:
    """
    Extract and cache embeddings per shard. Resumable — skips complete shards.
    Returns True if extraction succeeded, False if access was denied.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.data import BraTSDataset

    preprocessed_dir = Path(cfg["data"]["preprocessed_dir"])
    feat_dir = preprocessed_dir / "features" / model_name
    feat_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = cfg.get("features", {}).get("fp16", True) and not smoke
    batch_size = cfg["features"]["batch_size"] if not smoke else 8
    num_workers = cfg["features"]["num_workers"] if not smoke else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if smoke:
        device = torch.device("cpu")

    # load model
    model, transform, embed_fn = load_model_for_extraction(model_name, device, cfg)
    if model is None:
        return False  # access denied — already printed URL

    model.to(device)

    # group manifest by shard
    shards = manifest.groupby("shard_path")
    n_shards = manifest["shard_path"].nunique()
    log.info("Extracting %s features for %d patches across %d shards …",
             model_name, len(manifest), n_shards)

    total_t0 = time.time()
    total_patches = 0

    for shard_idx, (shard_path, shard_df) in enumerate(shards):
        npz_path = shard_npz_path(feat_dir, shard_path)
        if npz_path.exists():
            log.info("  [%d/%d] %s — already cached, skipping",
                     shard_idx + 1, n_shards, Path(shard_path).name)
            continue

        shard_df = shard_df.reset_index(drop=True)
        ds = BraTSDataset(shard_df, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=device.type == "cuda")

        all_feats = []
        all_keys = shard_df["key"].tolist()
        t0 = time.time()

        with torch.inference_mode():
            ctx = (torch.autocast("cuda", dtype=torch.float16)
                   if use_fp16 and device.type == "cuda" else _null_ctx())
            with ctx:
                for xb, _ in loader:
                    xb = xb.to(device)
                    feats = embed_fn(model, xb)
                    all_feats.append(feats.float().cpu().numpy())

        feats_arr = np.concatenate(all_feats, axis=0).astype(np.float32)
        np.savez_compressed(npz_path, keys=np.array(all_keys), feats=feats_arr)

        elapsed = time.time() - t0
        total_patches += len(shard_df)
        patches_per_sec = len(shard_df) / elapsed
        log.info(
            "  [%d/%d] %s — %d patches, %.0f patches/s, dim=%d, saved %s",
            shard_idx + 1, n_shards, Path(shard_path).name,
            len(shard_df), patches_per_sec, feats_arr.shape[1], npz_path.name,
        )

    total_elapsed = time.time() - total_t0
    log.info(
        "Feature extraction done: %d patches in %.0fs (%.0f patches/s)",
        total_patches, total_elapsed,
        total_patches / total_elapsed if total_elapsed > 0 else 0,
    )
    return True


def load_features(
    model_name: str,
    manifest: pd.DataFrame,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all cached embeddings and align to manifest row order.
    Returns (X, y, groups) aligned to manifest.
    """
    preprocessed_dir = Path(cfg["data"]["preprocessed_dir"])
    feat_dir = preprocessed_dir / "features" / model_name

    # build key → feature mapping
    key_to_feat: Dict[str, np.ndarray] = {}
    npz_files = sorted(feat_dir.glob("*.npz"))
    assert len(npz_files) > 0, f"No cached features at {feat_dir} — run exp2 first"

    for npz_path in npz_files:
        d = np.load(npz_path, allow_pickle=False)
        for k, f in zip(d["keys"], d["feats"]):
            key_to_feat[str(k)] = f

    # align to manifest
    missing = [k for k in manifest["key"] if k not in key_to_feat]
    assert len(missing) == 0, (
        f"{len(missing)} manifest keys have no cached features — "
        "extraction incomplete. Re-run feature extraction."
    )

    X = np.stack([key_to_feat[k] for k in manifest["key"]], axis=0)
    y = manifest["label_idx"].values.astype(np.int64)
    groups = manifest["patient"].values
    log.info("Loaded features: X=%s, %d classes, %d patients",
             X.shape, len(np.unique(y)), len(np.unique(groups)))
    return X, y, groups


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass
