# BraTS-Path 2026 — Pipeline Explanation

## Table of Contents
1. [What This Pipeline Does and Why](#1-what-this-pipeline-does-and-why)
2. [Data Layout and Format](#2-data-layout-and-format)
3. [Step 0 — Data Inspection](#3-step-0--data-inspection)
4. [Step 1 — Preprocessing](#4-step-1--preprocessing)
5. [Experiment 1 — Leakage Quantification](#5-experiment-1--leakage-quantification)
6. [Experiment 2 — Foundation Model Probe](#6-experiment-2--foundation-model-probe)
7. [Validation Set Handling](#7-validation-set-handling)
8. [Report Generation](#8-report-generation)
9. [Code Architecture](#9-code-architecture)
10. [How to Run](#10-how-to-run)
11. [Expected Outputs](#11-expected-outputs)
12. [Key Design Decisions and Why](#12-key-design-decisions-and-why)

---

## 1. What This Pipeline Does and Why

### The Problem

BraTS 2026, Task 5 (BraTS-Path) is a 10-class patch-level classification task on H&E-stained glioblastoma tissue. Patches come from whole-slide images (WSIs) across 11 different institutions.

The organiser's EfficientNet baseline reports a **5-fold cross-validation (CV) macro-F1 of ~0.98**, but the same model scores only **~0.52–0.55 on the held-out validation/test sets**. That is a 0.46-point collapse — almost half the apparent performance vanishes the moment the model faces data it hasn't seen before.

Before committing time and compute to a complex in-domain SSL approach (SAP-LeJEPA), this pipeline answers two foundational questions cheaply (~1 GPU-day):

| Question | Experiment | Why it matters |
|----------|-----------|----------------|
| How much of the 0.98→0.52 collapse is just bad CV splitting? | Exp 1 | If leakage explains most of it, fixing the split gives a trustworthy local proxy — no need to burn submission credits to iterate |
| What score does a frozen pathology foundation model already achieve? | Exp 2 | Sets the real bar SAP-LeJEPA must beat; if frozen models already match the leaderboard, SSL may not be worth pursuing |

---

## 2. Data Layout and Format

### Directory Structure

```
brats/
└── dataset/
    ├── BraTS-PATH-2026-Training-Labeled-Set/
    │   ├── shard-000000.tar … shard-000039.tar   ← 40 shards, ~4 GB each
    │   ├── class_map.json                         ← label abbreviation → index
    │   └── BraTS-Path-2026-Train-Patch-Patient-Slide-Mapping.csv
    ├── BraTS-Path-2026-Unlabeled-...-Scripts/     ← NOT used in these experiments
    └── BraTS-Path-2026-Validation-Set/
        ├── val-shard-000000.tar                   ← submission-only, no labels
        └── class_map.json
```

### Shard Format (WebDataset)

Each `.tar` shard contains **pairs of files per patch**:

```
train_00001d0cffd1d2a6.jpg    ← 512×512 RGB image (~100 KB compressed JPEG)
train_00001d0cffd1d2a6.cls    ← 2-byte ASCII integer, e.g. "5"
train_0000cd35c76c0e74.jpg
train_0000cd35c76c0e74.cls
...
```

**Critical fact:** Labels live in `.cls` sidecars **inside the tar**, NOT in the mapping CSV. The mapping CSV only provides `Name → Patient → Slide` grouping information. This was verified by inspection before writing a single line of training code.

### Class Map

```json
{"CT": 0, "DM": 1, "IC": 2, "LI": 3, "MP": 4,
 "NC": 5, "PL": 6, "PN": 7, "WM": 8, "NOTA": 9}
```

| Index | Abbr | Full Name |
|-------|------|-----------|
| 0 | CT | Cellular Tumour |
| 1 | DM | Dense Macrophage |
| 2 | IC | Cortical Infiltration |
| 3 | LI | Leptomeningeal Infiltration |
| 4 | MP | Microvascular Proliferation |
| 5 | NC | Geographic Necrosis |
| 6 | PL | Lymphocyte Presence |
| 7 | PN | Pseudopalisading Necrosis |
| 8 | WM | White Matter Penetration |
| 9 | NOTA | None Of The Above |

### Scale

- **1,631,432 total patches** across 40 shards
- **126 unique patients**, 255 unique slides
- Each patient contributes ~13,000 patches on average (highly variable)

---

## 3. Step 0 — Data Inspection

**File:** `src/inspect_data.py` | **Command:** `python run.py inspect`

Before writing the pipeline, this step decodes one shard and prints everything learned about the data format:

- Key pattern (`train_<16-char-hex>`)
- File extensions present (`.jpg` + `.cls`)
- Whether labels are in the tar or the CSV (→ tar)
- Decoded image shape and mode
- Mapping CSV headers and sample rows
- Which column provides the grouping key
- Whether the metadata TSV contains site/institution information (→ it does not)
- Whether the validation set has labels (→ it does not)

This step exists because **wrong assumptions about where labels live will silently waste a GPU-day**. The baseline was written only after this step confirmed the data format.

---

## 4. Step 1 — Preprocessing

**File:** `src/data.py` | **Command:** `python run.py preprocess`  
**Output:** `brats/preprocessed/manifest.parquet`

### What It Does

1. **Scans all 40 shards in parallel** using Python's `ProcessPoolExecutor`. For each shard it reads:
   - The `.jpg` member's `offset_data` and `size` (byte position inside the tar — enables O(1) random access later without re-scanning)
   - The `.cls` member's 2-byte content (the integer label)
   
   This reads only the tar headers and tiny `.cls` files — the actual image pixels are never loaded during preprocessing.

2. **Joins with the mapping CSV** to attach `patient` and `slide` to each patch key.

3. **Hard-asserts data integrity:**
   - Every patch has a non-null patient ID (grouped CV is meaningless otherwise)
   - Every class spans more than one patient (otherwise a class would always appear in both train and val — grouped CV cannot handle it)

4. **Writes `manifest.parquet`** with these columns:

   | Column | Description |
   |--------|-------------|
   | `key` | Unique patch identifier (e.g. `train_00001d0cffd1d2a6`) |
   | `label_idx` | Integer class label 0–9 |
   | `label_abbr` | Abbreviation (e.g. `CT`, `PN`) |
   | `patient` | Patient ID — the grouping key for honest CV |
   | `slide` | Slide ID (nested under patient) |
   | `shard_path` | Absolute path to the `.tar` file containing this patch |
   | `offset_jpg` | Byte offset of the JPEG data inside that tar |
   | `size_jpg` | Byte size of the JPEG data |
   | `split` | Always `"train"` for labeled set |

### Why Store Byte Offsets?

With 1.63M patches across 40 tars (~160 GB total), you cannot load everything into RAM. The offset trick lets the `BraTSDataset` class do this in `__getitem__`:

```python
with open(shard_path, 'rb') as f:
    f.seek(offset_jpg)        # jump directly to this patch
    data = f.read(size_jpg)   # read only this patch's bytes
img = Image.open(io.BytesIO(data))
```

This is O(1) per patch, works with PyTorch's multi-process DataLoader, and requires no intermediate extraction to disk.

---

## 5. Experiment 1 — Leakage Quantification

**File:** `src/baseline.py`, `src/splits.py` | **Command:** `python run.py exp1`

### The Idea

The same EfficientNet-B0 is trained twice under two different splitting strategies. Everything else (model architecture, optimizer, augmentation, batch size, seeds) is **identical**. The split is the only variable.

### Regime A — Random / Leaky (`StratifiedKFold`)

`sklearn.model_selection.StratifiedKFold` splits patches randomly, stratified by class label. It completely ignores which patient a patch came from.

**The problem:** A single patient contributes hundreds to thousands of patches. When the same patient appears in both the training fold and the validation fold, the model has already "seen" tissue from that patient during training. Validation performance is inflated because the model is being tested on a near-duplicate of something it trained on.

This is exactly how the organiser's baseline achieved 0.98 CV — all those patches from the same patient in train and val look nearly identical, so the model trivially memorises them.

### Regime B — Grouped / Honest (`StratifiedGroupKFold`)

`sklearn.model_selection.StratifiedGroupKFold` splits by patient group. A patient's patches either go entirely into the training fold or entirely into the validation fold — never both.

**After every fold is created, the pipeline hard-asserts:**
```python
overlap = set(train_patients) & set(val_patients)
assert len(overlap) == 0, f"GROUP OVERLAP DETECTED: {overlap}"
```
If this assertion fires for any fold, the run aborts immediately. A silently invalid split is the worst possible outcome.

### Training Recipe (same for both regimes)

| Setting | Value |
|---------|-------|
| Model | EfficientNet-B0 (timm, ImageNet pretrained) |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01) |
| Scheduler | Cosine annealing over all epochs |
| Loss | Cross-entropy with label smoothing 0.1 |
| Sampling | `WeightedRandomSampler` (class-balanced — each class gets equal expected frequency per batch) |
| Early stopping | Stop if val macro-F1 doesn't improve for 3 epochs |
| Max epochs | 15 |
| Image size | 224×224 (resized from 512×512) |

### What We Measure

After 5 folds for each regime:

- **macro-F1** — primary metric (unweighted average F1 across all 10 classes)
- **MCC** — Matthews Correlation Coefficient (robust to class imbalance)
- **Balanced Accuracy** — mean recall across classes

The key output is:

```
Δ = F1(A) − F1(B)
```

And the fraction of the organiser's gap (0.98 − 0.52 = 0.46) that this Δ explains.

**Interpretation:**
- If Δ ≈ 0.46 → leakage explains nearly all of the collapse. Fix the split and you have a reliable local proxy.
- If Δ ≈ 0 → the gap is due to something else (stain shift, site shift, domain gap) — leakage is not the primary issue.
- If grouped CV F1 (B) ≈ 0.52 → we have a faithful local proxy for the leaderboard and can iterate without burning submission credits.

---

## 6. Experiment 2 — Foundation Model Probe

**Files:** `src/features.py`, `src/probe.py` | **Command:** `python run.py exp2`

### The Idea

Instead of fine-tuning anything, we extract frozen embeddings from two state-of-the-art pathology foundation models and train only a simple logistic regression head. This measures what these models already know about glioblastoma tissue without any task-specific training.

This establishes the real bar: if a frozen model already matches or beats the leaderboard, training SAP-LeJEPA from scratch may not be worth the effort.

### Models

#### UNI2-h (`MahmoodLab/UNI2-h`)
- Architecture: ViT-H/14 (Vision Transformer, huge, 14×14 pixel patches)
- Pretrained on: 100K+ H&E pathology slides (in-domain!)
- Embedding dimension: **1536-d** (the pooled [CLS] token output)
- Access: Gated on HuggingFace — requires requesting access

```python
# exact code from the brief — not improvised
timm_kwargs = {
    'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
    'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
    'num_classes': 0, 'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
    'reg_tokens': 8, 'dynamic_img_size': True,
}
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
embedding = model(image)  # → [B, 1536]
```

#### Virchow2 (`paige-ai/Virchow2`)
- Architecture: ViT-H/14 with 4 register tokens
- Pretrained on: 3.1M pathology slides
- Embedding dimension: **2560-d** = CLS token (1280) + mean of patch tokens (1280)
- Token handling is non-trivial — register tokens must be excluded:

```python
output = model(image)      # → [B, 261, 1280]  (1 CLS + 4 register + 256 patch)
cls    = output[:, 0]      # → [B, 1280]
patch  = output[:, 5:]     # → [B, 256, 1280]  (skip CLS + 4 register tokens)
embedding = torch.cat([cls, patch.mean(1)], dim=-1)  # → [B, 2560]
```

Using only the CLS token (1280-d) would discard spatial information. The concat with mean-pooled patch tokens is the correct embedding for Virchow2 as specified by the authors.

### Gated Model Access Guard

Before loading any model, the pipeline makes a lightweight HuggingFace API call to check access:

```python
from huggingface_hub import model_info
model_info("MahmoodLab/UNI2-h")  # raises 401/403 if not granted
```

On failure, it prints the exact access URL and exits — it never retries, prompts for tokens, or attempts to bypass gating.

### Feature Caching (Resumable)

Extracting 1.63M embeddings takes time. Features are cached as NPZ files per shard:

```
brats/preprocessed/features/
├── uni2h/
│   ├── shard-000000.npz   ← keys=["train_abc...", ...], feats=float32[N, 1536]
│   ├── shard-000001.npz
│   └── ...
└── virchow2/
    ├── shard-000000.npz   ← feats=float32[N, 2560]
    └── ...
```

If a job is interrupted (SLURM time limit, etc.), re-running will skip already-extracted shards and continue from where it left off. The probe only runs after all shards are complete.

### Linear Probe

After extraction, two probes are fit for each model:

1. **Balanced probe** — `LogisticRegression(class_weight='balanced')`: corrects for class imbalance by upweighting minority classes
2. **Unweighted probe** — `LogisticRegression(class_weight=None)`: shows what happens without imbalance correction (should be worse — demonstrates the imbalance effect)

The probe always uses the **same grouped folds as Exp 1(B)** — never the random split. This ensures all numbers are comparable.

**Standardisation:** `StandardScaler` is fit on the training split only, then applied to val. This prevents information leakage from the val set into the feature normalisation.

### Metrics

| Metric | What it measures |
|--------|-----------------|
| macro-F1 | Average F1 across all 10 classes, unweighted — the challenge's primary metric |
| MCC | Matthews Correlation Coefficient — robust to imbalance, −1 to +1 |
| Balanced Accuracy | Mean recall per class — directly measures per-class performance |
| macro-AUROC (OvR) | Area under ROC for each class vs. rest, averaged — measures discrimination regardless of threshold |

### Leaderboard Verdict

The pipeline automatically compares the probe's macro-F1 against the public leaderboard band (0.43–0.63) and prints a verdict line — whether the frozen model is inside, above, or below the band. This is the direct answer to "should we bother with SAP-LeJEPA?"

---

## 7. Validation Set Handling

**Command:** `python run.py valset`

The official validation set (`val-shard-000000.tar`, ~10 GB) contains **JPEG patches only — no `.cls` label sidecars**. Ground truth is withheld by the organisers for the leaderboard.

The pipeline:
1. Detects that no labels are present (verified by inspection — only `.jpg` files in the tar)
2. Extracts UNI2-h and Virchow2 embeddings and caches them at `brats/preprocessed/features/val_uni2h/` and `val_virchow2/`
3. **Does not compute any score** — doing so would require fabricating labels
4. Reports this honestly in REPORT.md

The cached val features are ready for use in a submission pipeline once the models are finalised.

---

## 8. Report Generation

**File:** `src/report.py` | **Output:** `brats-t5/REPORT.md`

The report is assembled entirely from logged CSV files under `runs/`. **No number is hardcoded.** Every metric traces back to a real prediction made by a real model on real data.

The report contains:
- **Table A** — leakage quantification: random vs grouped F1/MCC/BalAcc, Δ, and fraction of the organiser gap explained
- **Table B** — frozen probe results for UNI2-h and Virchow2 vs. the honest baseline and the leaderboard band
- **Verdict** — explicit statement of whether a frozen model already clears the leaderboard
- **Validation set** — honest note that labels are unavailable
- **Site diagnostics** — explains why site-per-fold analysis is impossible (Synapse metadata has no institution column)
- **Falsifiable hypothesis** — a specific, numerical prediction for SAP-LeJEPA to beat (filled from real probe numbers once Exp 2 runs)
- **Compute estimate** — measured throughput (patches/s from actual extraction timing) scaled to full-dataset GPU-hours
- **Manual TODOs** — two things only the researcher can do: confirm site info on Synapse, and ensure HF model access

---

## 9. Code Architecture

```
brats-t5/
├── run.py                  ← main entrypoint (subcommands: smoke, inspect, preprocess, exp1, exp2, valset, all)
├── configs/
│   └── default.yaml        ← single config controls all paths, seeds, hyperparameters
├── src/
│   ├── data.py             ← shard scanning, manifest building, BraTSDataset, smoke generator
│   ├── splits.py           ← StratifiedKFold + StratifiedGroupKFold, overlap assertion
│   ├── baseline.py         ← EfficientNet-B0 training loop, metrics, early stopping
│   ├── features.py         ← UNI2-h + Virchow2 loading, embedding extraction, NPZ cache
│   ├── probe.py            ← LogisticRegression probe, leaderboard comparison, Table A/B
│   ├── report.py           ← REPORT.md assembly from logged CSVs
│   └── inspect_data.py     ← Step 0 data format inspection
├── slurm/
│   ├── install_deps.sh     ← installs torch+torchvision+timm+huggingface_hub on GPU node
│   ├── run_preprocess.sh   ← SLURM job: build manifest.parquet
│   ├── run_exp1.sh         ← SLURM job: experiment 1
│   ├── run_exp2.sh         ← SLURM job: experiment 2
│   └── run_all.sh          ← submits all three as a chained dependency graph
├── tests/
│   └── test_split_assertion.py  ← 11 unit tests including the mandatory overlap-fires test
└── REPORT.md               ← generated output (⚠ currently contains synthetic smoke-test numbers)
```

### Data Flow

```
40 × shard-*.tar
       │
       ▼ scan_shard() [parallel, CPU]
manifest.parquet  ←── mapping CSV (Patient/Slide)
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
  StratifiedKFold                   StratifiedGroupKFold
  (Regime A, leaky)                 (Regime B, honest + asserted)
       │                                      │
       ▼                                      ▼
EfficientNet-B0               EfficientNet-B0 + UNI2-h probe + Virchow2 probe
       │                                      │
       └──────────────────┬───────────────────┘
                          ▼
                      REPORT.md
```

---

## 10. How to Run

### Prerequisites

1. You are in `/shared/users/hassan2/brats-t5/`
2. SLURM partition `gpu` is available with node `iit-MS-7E06`
3. HuggingFace access granted for `MahmoodLab/UNI2-h` and `paige-ai/Virchow2`

### Step-by-step

```bash
# 1. Verify the pipeline works end-to-end on synthetic data (CPU, <3s)
PYTHONPATH=. /shared/users/hassan2/envs/physionet2026/bin/python run.py smoke

# 2. Install GPU dependencies on the compute node
sbatch slurm/install_deps.sh
# Wait for completion → cat runs/install_<jobid>.out → must end with "ALL IMPORTS OK"

# 3. Log in to HuggingFace on the GPU node (one-time, interactive)
srun --partition=gpu --gres=gpu:1 --pty bash
conda activate /shared/users/hassan2/envs/physionet2026
huggingface-cli login    # paste your token
exit

# 4. Submit the full pipeline
bash slurm/run_all.sh
# This submits: preprocess → exp1 + exp2 (in parallel, both after preprocess)

# 5. Monitor
squeue -u $USER
tail -f runs/preprocess_<jobid>.out
tail -f runs/exp1_<jobid>.out
tail -f runs/exp2_<jobid>.out

# 6. Read results
cat REPORT.md
```

### Individual Commands

```bash
# Inspect data format only (fast, no GPU)
PYTHONPATH=. python run.py inspect --config configs/default.yaml

# Build manifest only
PYTHONPATH=. python run.py preprocess --config configs/default.yaml

# Run exp1 only (needs GPU)
sbatch slurm/run_exp1.sh

# Run exp2 only (needs GPU + HF access)
sbatch slurm/run_exp2.sh
```

### Smoke Test (no GPU, no HF, synthetic data)

```bash
PYTHONPATH=. /shared/users/hassan2/envs/physionet2026/bin/python run.py smoke
```

Runs the entire pipeline end-to-end on 300 randomly generated patches in ~3 seconds. Produces a `REPORT.md` watermarked with `⚠ SYNTHETIC DATA`. Use this to verify the pipeline is wired correctly before submitting SLURM jobs.

### Unit Tests

```bash
PYTHONPATH=. /shared/users/hassan2/envs/physionet2026/bin/python -m pytest tests/ -v
```

11 tests, including one that **deliberately constructs an overlapping split and verifies the assertion fires** — the most important correctness guarantee in the whole pipeline.

---

## 11. Expected Outputs

### After `preprocess`
```
brats/preprocessed/
└── manifest.parquet    (1,631,432 rows × 9 columns, ~150 MB)
```

### After `exp1`
```
brats-t5/runs/exp1_<timestamp>/
├── exp1_random/
│   ├── fold_00/fold_metrics.csv    ← macro_f1, mcc, balanced_acc for this fold
│   ├── fold_00/epoch_log.csv       ← per-epoch train loss + val metrics
│   └── fold_01/ … fold_04/
└── exp1_grouped/
    └── fold_00/ … fold_04/
```

### After `exp2`
```
brats/preprocessed/features/
├── uni2h/
│   └── shard-000000.npz … shard-000039.npz    (~9.5 GB total, float32[N, 1536])
└── virchow2/
    └── shard-000000.npz … shard-000039.npz    (~16.7 GB total, float32[N, 2560])

brats-t5/runs/exp2_<timestamp>/
├── exp2_uni2h/
│   ├── balanced_fold_metrics.csv
│   └── unweighted_fold_metrics.csv
└── exp2_virchow2/
    └── …
```

### Final Output
```
brats-t5/REPORT.md    ← all numbers from real logged CSVs, nothing hardcoded
```

---

## 12. Key Design Decisions and Why

### Why read labels from `.cls` sidecars, not the mapping CSV?
The mapping CSV has no label column (`Name`, `Patient`, `Slide` only). This was confirmed by inspecting the actual CSV. Any code that tried to get labels from the CSV would silently produce wrong data.

### Why store byte offsets in the manifest?
1.63M patches × ~100 KB each = ~160 GB. Extracting everything to disk wastes space and takes hours. Seeking directly into the tar at a known byte offset is O(1) and reads only the bytes for the requested patch.

### Why `Patient` as the group key, not `Slide`?
A patient has multiple slides. Grouping by slide would still allow patches from the same patient to appear in both train and val (via different slides). Patient is the strongest available grouping — it guarantees that the model has seen zero tissue from each validation patient during training.

### Why 5-fold CV instead of a fixed split?
Five folds give mean ± std across folds, making it possible to see whether the Δ between regimes is consistent or just fold-specific noise. A single split gives one number with no uncertainty estimate.

### Why LogisticRegression for the probe, not a small MLP?
Linear probes are the standard evaluation for frozen representations because they measure only the linear separability of the embedding space. A non-linear head could partially compensate for poor representations and muddy the comparison between models.

### Why cache features per-shard rather than one big file?
Resumability. Feature extraction for 1.63M patches takes hours. If the SLURM job hits the time limit at shard 30, re-running picks up from shard 31. One big file would mean starting over.

### Why does the validation set get features extracted but no score reported?
The val set has no labels — they are withheld by the organisers. Computing a score would require either guessing labels (fabrication) or comparing against a metric that doesn't exist locally. The features are extracted so they're ready for submission once the final model is chosen.

### Why is the `multi_class` parameter warning in sklearn acceptable?
The `FutureWarning: 'multi_class' was deprecated` is from sklearn ≥ 1.5. The parameter will be removed in 1.7 but is still functional now. The probe explicitly sets `multi_class='multinomial'` because this is the correct setting for 10-class logistic regression (vs. the default one-vs-rest). The warning is expected and harmless.
