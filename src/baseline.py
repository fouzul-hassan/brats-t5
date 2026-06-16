"""
EfficientNet-B0 supervised baseline (Experiment 1).

Identical training recipe across both split regimes — the split is the only variable.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def get_transforms(image_size: int = 224, train: bool = True):
    """
    Light augmentation.
    Uses torchvision when available (GPU jobs); falls back to a pure PIL+torch
    pipeline when torchvision is incompatible (CPU login node / smoke).
    """
    try:
        import torchvision.transforms as T
        if train:
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    except Exception:
        # fallback: PIL-only transform returning a torch tensor
        import torch
        import numpy as np
        from PIL import Image as PILImage

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def _transform(img):
            img = img.resize((image_size, image_size), PILImage.BILINEAR)
            if train:
                import random
                if random.random() > 0.5:
                    img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
                if random.random() > 0.5:
                    img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            return torch.from_numpy(arr.transpose(2, 0, 1))  # HWC → CHW

        return _transform


def build_model(cfg: dict, smoke: bool = False):
    """
    Return EfficientNet-B0 (timm) for real runs, or a tiny CPU stub for smoke.
    """
    import torch.nn as nn
    if smoke:
        return _SmokeNet(num_classes=10)

    import timm
    model = timm.create_model(
        cfg["train"]["model"],
        pretrained=cfg["train"].get("pretrained", True),
        num_classes=10,
    )
    return model


class _SmokeNet(object):  # not nn.Module so no torch import needed at definition time
    """Tiny model for CPU smoke test — no timm required."""
    def __new__(cls, num_classes=10):
        import torch.nn as nn
        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d(4)
                self.fc = nn.Linear(3 * 4 * 4, num_classes)
            def forward(self, x):
                return self.fc(self.pool(x).flatten(1))
        return _Net()


def make_class_weighted_sampler(labels: np.ndarray):
    import torch
    from torch.utils.data import WeightedRandomSampler
    classes, counts = np.unique(labels, return_counts=True)
    weights_per_class = 1.0 / counts
    sample_weights = np.array([weights_per_class[np.where(classes == l)[0][0]] for l in labels])
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                                 f1_score, roc_auc_score)
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics["macro_auroc"] = roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovr"
            )
        except Exception:
            metrics["macro_auroc"] = float("nan")
    return metrics


def train_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    run_dir: Path,
    smoke: bool = False,
) -> Dict[str, float]:
    """
    Train one fold and return val metrics.
    All metric values come from actual model predictions — nothing hardcoded.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from src.data import BraTSDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg.get("image_size", 224)
    tcfg = cfg["train"]
    epochs = tcfg["epochs"] if not smoke else 2
    batch_size = tcfg["batch_size"] if not smoke else 16
    num_workers = tcfg["num_workers"] if not smoke else 0
    patience = tcfg["early_stop_patience"] if not smoke else 1

    train_ds = BraTSDataset(train_df, transform=get_transforms(image_size, train=True))
    val_ds = BraTSDataset(val_df, transform=get_transforms(image_size, train=False))

    sampler = make_class_weighted_sampler(train_ds.get_labels())
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=device.type == "cuda")

    model = build_model(cfg, smoke=smoke).to(device)
    optimizer = AdamW(model.parameters(), lr=tcfg["lr"],
                      weight_decay=tcfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.1))

    best_f1 = -1.0
    best_state = None
    no_improve = 0
    fold_log = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        scheduler.step()
        train_loss /= len(train_ds)

        # --- val ---
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)
                all_preds.append(preds)
                all_labels.append(yb.numpy())
                all_probs.append(probs)

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        elapsed = time.time() - t0

        log.info(
            "Fold %d | Epoch %d/%d | loss=%.4f | F1=%.4f | MCC=%.4f | elapsed=%.0fs",
            fold_idx, epoch, epochs, train_loss, metrics["macro_f1"], metrics["mcc"], elapsed
        )
        fold_log.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Fold %d: early stop at epoch %d (patience=%d)", fold_idx, epoch, patience)
                break

    # reload best and compute final val metrics
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_preds.append(np.argmax(probs, axis=-1))
            all_labels.append(yb.numpy())
            all_probs.append(probs)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    final_metrics = compute_metrics(y_true, y_pred, y_prob)
    final_metrics["fold"] = fold_idx
    final_metrics["best_epoch"] = epoch - no_improve
    final_metrics["elapsed_s"] = time.time() - t0

    # save per-fold log
    fold_dir = run_dir / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_log).to_csv(fold_dir / "epoch_log.csv", index=False)
    pd.DataFrame([final_metrics]).to_csv(fold_dir / "fold_metrics.csv", index=False)

    return final_metrics


def run_experiment1(manifest: pd.DataFrame, splits_A, splits_B, cfg: dict,
                    run_dir: Path, smoke: bool = False) -> Tuple[List[dict], List[dict]]:
    """
    Run both split regimes. Returns (results_A, results_B).
    """
    from src.data import BraTSDataset  # noqa: F401 — ensure import works

    results_A, results_B = [], []

    for regime, splits, results in [
        ("random", splits_A, results_A),
        ("grouped", splits_B, results_B),
    ]:
        log.info("\n%s\nEXP1 — regime=%s\n%s", "=" * 60, regime, "=" * 60)
        regime_dir = run_dir / f"exp1_{regime}"
        regime_dir.mkdir(parents=True, exist_ok=True)
        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            train_df = manifest.iloc[tr_idx].reset_index(drop=True)
            val_df = manifest.iloc[va_idx].reset_index(drop=True)
            m = train_fold(fold_idx, train_df, val_df, cfg,
                           regime_dir, smoke=smoke)
            m["regime"] = regime
            results.append(m)
            log.info("Fold %d/%d done — F1=%.4f MCC=%.4f",
                     fold_idx + 1, len(splits), m["macro_f1"], m["mcc"])

    return results_A, results_B


def summarise_regime(results: List[dict], label: str) -> Dict[str, Tuple[float, float]]:
    """Return {metric: (mean, std)} and print a summary."""
    df = pd.DataFrame(results)
    metrics = ["macro_f1", "mcc", "balanced_acc"]
    out = {}
    print(f"\n{label}")
    for m in metrics:
        vals = df[m].values
        mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1))
        out[m] = (mean, std)
        print(f"  {m:16s}: {mean:.4f} ± {std:.4f}")
    return out
