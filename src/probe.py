"""
Logistic regression linear probe on frozen foundation-model embeddings (Experiment 2).

Runs on the same grouped folds as Exp1(B) — never random split.
Reports weighted (balanced) and unweighted probes to show imbalance effect.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LEADERBOARD_BAND = (0.43, 0.63)  # public F1 range for comparison


def run_probe(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: list,
    cfg: dict,
    run_dir: Path,
    smoke: bool = False,
) -> Dict[str, List[dict]]:
    """
    Run linear probe on grouped folds.
    Returns dict with 'balanced' and 'unweighted' lists of per-fold metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (f1_score, matthews_corrcoef,
                                 balanced_accuracy_score, roc_auc_score)

    pcfg = cfg.get("probe", {})
    C = pcfg.get("C", 1.0)
    max_iter = pcfg.get("max_iter", 1000) if not smoke else 50
    tol = pcfg.get("tol", 1e-4)
    solver = pcfg.get("solver", "saga")

    results = {"balanced": [], "unweighted": []}
    probe_dir = run_dir / f"exp2_{model_name}"
    probe_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # standardise: fit on train only
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        for variant, cw in [("balanced", "balanced"), ("unweighted", None)]:
            clf = LogisticRegression(
                C=C, class_weight=cw, multi_class="multinomial",
                solver=solver, max_iter=max_iter, tol=tol,
                random_state=cfg.get("seed", 42),
            )
            clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_va_s)
            y_prob = clf.predict_proba(X_va_s)

            m = {
                "fold": fold_idx,
                "model": model_name,
                "variant": variant,
                "macro_f1": f1_score(y_va, y_pred, average="macro", zero_division=0),
                "mcc": matthews_corrcoef(y_va, y_pred),
                "balanced_acc": balanced_accuracy_score(y_va, y_pred),
            }
            try:
                m["macro_auroc"] = roc_auc_score(
                    y_va, y_prob, average="macro", multi_class="ovr"
                )
            except Exception:
                m["macro_auroc"] = float("nan")

            results[variant].append(m)
            log.info(
                "Probe %s fold=%d variant=%s | F1=%.4f MCC=%.4f BalAcc=%.4f AUROC=%.4f",
                model_name, fold_idx, variant,
                m["macro_f1"], m["mcc"], m["balanced_acc"], m.get("macro_auroc", float("nan")),
            )

        # save fold predictions for audit trail
        fold_df = pd.DataFrame({
            "key": pd.Series(range(len(y_va))),  # index into va_idx
            "y_true": y_va,
            "y_pred_balanced": results["balanced"][-1]["macro_f1"],  # just summary for now
        })
        fold_df.to_csv(probe_dir / f"fold_{fold_idx:02d}_summary.csv", index=False)

    # save per-fold metrics
    for variant in ("balanced", "unweighted"):
        df = pd.DataFrame(results[variant])
        df.to_csv(probe_dir / f"{variant}_fold_metrics.csv", index=False)

    return results


def summarise_probe(
    results: Dict[str, List[dict]],
    model_name: str,
    baseline_grouped: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Print summary table and return {variant: {metric: (mean, std)}}."""
    metrics = ["macro_f1", "mcc", "balanced_acc", "macro_auroc"]
    summary = {}

    print(f"\n{'='*70}")
    print(f"PROBE RESULTS — {model_name.upper()}")
    print(f"{'='*70}")
    print(f"{'Variant':<14} {'Metric':<18} {'Mean':>8} {'Std':>8}")
    print("-" * 52)

    for variant, fold_results in results.items():
        df = pd.DataFrame(fold_results)
        variant_summary = {}
        for m in metrics:
            vals = df[m].values
            mean, std = float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1) if len(vals) > 1 else 0)
            variant_summary[m] = (mean, std)
            print(f"  {variant:<12} {m:<18} {mean:>8.4f} {std:>8.4f}")
        summary[variant] = variant_summary
        print()

    # leaderboard comparison
    best_f1 = summary["balanced"]["macro_f1"][0]
    lo, hi = LEADERBOARD_BAND
    if lo <= best_f1 <= hi:
        verdict = f"INSIDE leaderboard band [{lo:.2f}–{hi:.2f}]"
    elif best_f1 > hi:
        verdict = f"ABOVE leaderboard band (>{hi:.2f})"
    else:
        verdict = f"BELOW leaderboard band (<{lo:.2f})"
    print(f"\n  Frozen {model_name} macro-F1 = {best_f1:.4f} → {verdict}")

    if baseline_grouped:
        base_f1 = baseline_grouped.get("macro_f1", (float("nan"),))[0]
        delta = best_f1 - base_f1
        print(f"  vs. honest EfficientNet baseline: {base_f1:.4f}  Δ={delta:+.4f}")

    print(f"{'='*70}\n")
    return summary


def print_leakage_table(
    summary_A: Dict,
    summary_B: Dict,
) -> float:
    """Print Table A from the report. Returns Δ(F1)."""
    f1_a = summary_A["macro_f1"][0]
    f1_b = summary_B["macro_f1"][0]
    delta = f1_a - f1_b

    print(f"\n{'='*70}")
    print("TABLE A — LEAKAGE QUANTIFICATION (EfficientNet-B0, 5-fold CV)")
    print(f"{'='*70}")
    print(f"{'Regime':<30} {'macro-F1':>10} {'MCC':>10} {'BalAcc':>10}")
    print("-" * 62)
    for label, s in [("(A) Random / leaky", summary_A), ("(B) Grouped / honest", summary_B)]:
        f1 = f"{s['macro_f1'][0]:.4f} ± {s['macro_f1'][1]:.4f}"
        mcc = f"{s['mcc'][0]:.4f} ± {s['mcc'][1]:.4f}"
        ba = f"{s['balanced_acc'][0]:.4f} ± {s['balanced_acc'][1]:.4f}"
        print(f"  {label:<28} {f1:>10}  {mcc:>10}  {ba:>10}")
    print(f"\n  Δ = F1(A) − F1(B) = {delta:+.4f}")
    organiser_gap = 0.98 - 0.52
    leakage_pct = delta / organiser_gap * 100 if organiser_gap > 0 else float("nan")
    print(f"  Organiser gap (0.98→0.52) ≈ {organiser_gap:.2f}")
    print(f"  Fraction of gap attributable to leakage: {leakage_pct:.0f}%")
    if abs(f1_b - 0.52) < 0.08:
        print(
            "\n  ✓ FINDING: Grouped CV F1 ≈ leaderboard (~0.52) → honest local proxy confirmed."
        )
    print(f"{'='*70}\n")
    return delta
