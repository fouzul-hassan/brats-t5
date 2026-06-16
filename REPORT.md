# BraTS-Path 2026 — Leakage & Foundation Model De-risking Report

Generated: 2026-06-16 08:27:23  |  Run dir: `/shared/users/hassan2/brats-t5/smoke_run/runs/smoke`

> **⚠ SYNTHETIC DATA — ALL NUMBERS BELOW ARE FROM RANDOM DATA, NOT REAL BRATS IMAGES**


---

## Data Summary

- Total patches: **300**
- Patients (groups): **12**
- Slides: **36**
- Shards: **2**
- Group key for CV: **Patient** (strongest available grouping)
- Site data: **Not available** (Synapse metadata = file records only; see Manual TODOs)

### Class distribution

| Idx | Abbr | Full name (confirmed from class_map.json) | Count | % |
|-----|------|-------------------------------------------|-------|---|
| 0 | CT | Cellular Tumour | 34 | 11.3% |
| 1 | DM | Dense Macrophage | 33 | 11.0% |
| 2 | IC | Cortical Infiltration | 23 | 7.7% |
| 3 | LI | Leptomeningeal Infiltration | 18 | 6.0% |
| 4 | MP | Microvascular Proliferation | 27 | 9.0% |
| 5 | NC | Geographic Necrosis | 30 | 10.0% |
| 6 | PL | Lymphocyte Presence | 35 | 11.7% |
| 7 | PN | Pseudopalisading Necrosis | 38 | 12.7% |
| 8 | WM | White Matter Penetration | 30 | 10.0% |
| 9 | NOTA | None Of The Above | 32 | 10.7% |

## Table A — Leakage Quantification (Experiment 1)

**Model:** EfficientNet-B0 (timm, pretrained ImageNet)  |  **5-fold CV**  |  AdamW + cosine, class-balanced sampler, early-stop on val macro-F1

| Regime | macro-F1 | MCC | Balanced Accuracy |
|--------|----------|-----|-------------------|
| (A) Random / leaky (`StratifiedKFold`) | 0.0256 ± 0.0074 | 0.0250 ± 0.0511 | 0.1004 ± 0.0089 |
| (B) Grouped / honest (`StratifiedGroupKFold` on Patient) | 0.0395 ± 0.0016 | 0.0206 ± 0.0638 | 0.0965 ± 0.0144 |

**Δ = F1(A) − F1(B) = -0.0140**
Organiser-reported gap: 0.98 (CV) → ~0.52 (leaderboard) ≈ 0.46
**Fraction attributable to patch-level leakage: -3%**

## Table B — Frozen Foundation Model Probe (Experiment 2)

Same grouped folds as Exp1(B). Standardised features + `LogisticRegression(class_weight='balanced', multinomial)`

| Model | Dim | macro-F1 | MCC | Balanced Acc | macro-AUROC |
|-------|-----|----------|-----|--------------|-------------|
| Smoke stub (random) | 64 | 0.1064 ± 0.0061 | 0.0434 ± 0.0085 | 0.1374 ± 0.0209 | 0.5384 ± 0.0125 |

### 3.5 Verdict — Foundation model vs. leaderboard


## Validation Set (Official Held-Out)

The official validation set (`val-shard-000000.tar`) contains **jpg patches only — no labels**. It is a submission-only set; ground truth is withheld by the organisers.

→ Features have been extracted and cached for later submission pipeline use. No held-out score is reported here (doing so would require fabricating labels).

## Site / Leakage Diagnostics

**Site data is not available** in the downloaded files. The `SYNAPSE_METADATA_MANIFEST.tsv` contains only Synapse file-level metadata (shard filenames, checksums, parent IDs) — no institution/center/site column. The mapping CSV (`Name`, `Patient`, `Slide`) also contains no site information.

Site-per-fold distribution cannot be computed from available data. See **Manual TODOs** below.

## Falsifiable Hypothesis for SAP-LeJEPA

> **Hypothesis:** SAP-LeJEPA linear-probe macro-F1 will exceed frozen TBD's **X** by ≥ 0.05 under the grouped split (StratifiedGroupKFold on Patient, 5-fold). If it does not, in-domain SSL offers no measurable advantage over a frozen pathology foundation model on this task and dataset size, and further SSL investment is not justified.

## Compute / Timeline Estimate

*(Throughput not yet measured — run preprocess + exp1 first to get timing.)*

## Manual TODOs (Only You Can Do)

1. **Site confirmation**: Log into [Synapse](https://www.synapse.org/) and confirm whether the BraTS-Path 2026 hidden test set is drawn from held-out institutions not present in the training set. This determines whether site-shift is an active confounder beyond leakage.

2. **HuggingFace gated access**: Ensure you have been granted access to both [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) and [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2), and that `huggingface-cli login` has been run on the GPU node before launching Exp2.
