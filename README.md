# 🌿 Campus Vegetation Recognition

A traditional machine-learning pipeline for classifying **8 campus plant species** from self-collected smartphone photos. No deep learning — features are hand-crafted (HOG + HSV histogram + LBP), classifiers are scikit-learn (SVM, Random Forest, Gradient Boosting).

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [8 Plant Classes](#8-plant-classes)
3. [Pipeline Overview](#pipeline-overview)
4. [Setup & Installation](#setup--installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Running the Pipeline](#running-the-pipeline)
7. [Web app — launch & use](#step-3--launch-interactive-app-task-5)
8. [Output Files](#output-files)
9. [Design Decisions & Justifications](#design-decisions--justifications)
10. [Known Limitations](#known-limitations)
11. [Assignment Task Checklist](#assignment-task-checklist)

---

## Project Structure

```
COMP4423_Assignment2/
│
├── data/                           # Self-collected images (not included in full in some submissions)
│   ├── train/
│   │   ├── Camellia/
│   │   ├── Ficus_microcarpa/
│   │   ├── Ficus_microcarpa_golden_leaves/
│   │   ├── Goeppertia_makoyana/
│   │   ├── Juniperus_chinensis/
│   │   ├── Podocarpus_macrophyllus/
│   │   ├── Rhododendron/
│   │   └── Xanthostemon_chrysanthus/
│   ├── val/
│   │   └── <same 8 sub-folders>
│   └── test/
│       └── <same 8 sub-folders>
│
├── outputs/                        # Training + evaluation artefacts (created automatically)
│   ├── best_model.pkl
│   ├── SVM_RBF.pkl
│   ├── RandomForest.pkl
│   ├── GradientBoosting.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── label_encoder.pkl
│   ├── val_metrics.json
│   ├── train_log.txt
│   ├── test_metrics.txt            # from evaluate.py
│   ├── test_evaluation.json
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   ├── results.csv
│   ├── error_analysis.txt
│   ├── examples_correct_sheet.png
│   ├── examples_wrong_sheet.png
│   └── examples/                   # copied qualitative correct/wrong images
│
├── classifier.py                   # Task 3 — feature extraction + classifier training (main entry)
├── train.py                        # Thin wrapper: `python train.py` ≡ `python classifier.py`
├── evaluate.py                     # Task 4 — held-out test metrics + error analysis
├── app.py                          # Task 5 — local Flask web UI for predictions
├── templates/
│   └── index.html                  # Upload form + instructions (used by app.py)
├── requirements.txt
└── README.md
```

---

## 8 Plant Classes

All species were collected on the **PolyU Hung Hom campus**. Ground truth was confirmed via on-site plant nameplates for 7 of 8 classes; Rhododendron was verified by field guide.

| # | Class (folder name) | Species | Plant Type | Ground Truth |
|---|---|---|---|---|
| 1 | `Camellia` | *Camellia japonica* | Flowering shrub | Nameplate ✅ |
| 2 | `Ficus_microcarpa` | *Ficus microcarpa* | Evergreen tree | Nameplate ✅ |
| 3 | `Ficus_microcarpa_golden_leaves` | *Ficus microcarpa* 'Golden Leaves' | Ornamental shrub | Nameplate ✅ |
| 4 | `Goeppertia_makoyana` | *Goeppertia makoyana* | Indoor herb | Nameplate ✅ |
| 5 | `Juniperus_chinensis` | *Juniperus chinensis* | Coniferous shrub | Nameplate ✅ |
| 6 | `Podocarpus_macrophyllus` | *Podocarpus macrophyllus* | Coniferous tree | Nameplate ✅ |
| 7 | `Rhododendron` | *Rhododendron* spp. | Flowering shrub | Field guide ✅ |
| 8 | `Xanthostemon_chrysanthus` | *Xanthostemon chrysanthus* | Flowering tree | Nameplate ✅ |

---

## Pipeline Overview

```
Raw Images (128×128 RGB)
        │
        ├──► HOG   (shape & edges)          ~1764-D
        ├──► HSV Histogram (colour)            96-D
        └──► LBP   (micro-texture)             26-D
                        │
               Concatenate → ~1886-D
                        │
               StandardScaler (zero mean, unit variance)
                        │
               PCA (keep 95% variance) → ~150–250-D
                        │
          ┌─────────────┼─────────────────┐
          ▼             ▼                 ▼
      SVM (RBF)   Random Forest    Gradient Boosting
      fixed cfg   GridSearchCV     fixed cfg
                  (light tuning)
          │             │                 │
          └─────────────┴─────────────────┘
                        │
              Best model by val accuracy
                        │
                   best_model.pkl
```

---

## Setup & Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone / download this repository
cd COMP4423_Assignment2

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` installs:
- `scikit-learn` — classifiers, GridSearchCV, metrics
- `scikit-image` — HOG and LBP feature extraction
- `numpy`, `Pillow` — image loading and array operations
- `joblib` — model serialisation
- `tqdm` — progress bars
- `matplotlib` — contact sheets and confusion matrix plots

---

## Dataset Preparation

Images must be organised into the three-level folder structure shown above **before** running any script. Each class folder should contain only JPEG/PNG image files.

**Minimum image counts (per class):**

| Split | Target count |
|---|---|
| train | ≥ 42 images per class (70%) |
| val   | ≥ 9 images per class  (15%) |
| test  | ≥ 9 images per class  (15%) |

**File naming convention used during collection:**
```
<CLASS_PREFIX>_L<LOCATION_ID>_<INDEX>.jpg
e.g.  CAM_L01_0023.jpg   →  Camellia, Location 1, image 23
      FIC_L02_0007.jpg   →  Ficus microcarpa, Location 2, image 7
```

> ⚠️ The `data/` folder may be **omitted** from some submissions due to file size. The full dataset is available separately as `Assignment2_Code_<ID>_<Name>.zip`.

---

## Running the Pipeline

### Step 1 — Train classifiers (Task 3)

```bash
python train.py
# or: python classifier.py
```

With optional arguments:
```bash
python train.py \
    --data_dir  data     \      # path to root data folder (train/ + val/ required)
    --out_dir   outputs  \    # where to save models and logs
    --img_size  128      \    # resize all images to 128×128
    --seed      42       \    # global random seed
    --n_jobs    -1       \    # parallel jobs (-1 = all CPU cores)
    --eval_test               # optional: also evaluate best model on data/test/
```

**Expected runtime** (on a modern laptop, ~600 images):
- Feature extraction: ~2–4 min
- RF GridSearchCV (18 combinations × 3-fold): ~3–6 min
- SVM + GBDT training: ~1–2 min
- **Total: ~6–12 min**

### Step 2 — Evaluate on test set (Task 4)

Requires `outputs/` from training (`best_model.pkl`, `scaler.pkl`, `pca.pkl`, `label_encoder.pkl`) and a held-out `data/test/<class>/` tree.

```bash
python evaluate.py
# optional:
python evaluate.py --data_dir data --out_dir outputs --n_examples 5 --max_wrong_sheet 36
```

Writes **test accuracy**, **confusion matrix** (PNG + JSON), **per-class metrics**, **per-image CSV**, **qualitative example folders** and **contact sheets**, and **`error_analysis.txt`** (failure modes vs. HOG/HSV/LBP limitations + empirical confusion pairs).

### Step 3 — Launch interactive app (Task 5)

Small **Flask** web UI (`app.py` + `templates/index.html`) for uploading a photo and viewing the predicted plant class. It runs **locally** (default: only this computer).

**Prerequisite:** `outputs/` must contain the trained pipeline files (`best_model.pkl`, `scaler.pkl`, `pca.pkl`, `label_encoder.pkl`). If they are missing, run **Step 1 — Train classifiers** (above) first.

#### How to launch

1. Open a terminal in the project root (the folder that contains `app.py`).
2. Install dependencies (once): `pip install -r requirements.txt`
3. Start the server: `python app.py`
4. When the terminal prints the URL, open **http://127.0.0.1:5000/** in your web browser (Chrome, Firefox, Safari, Edge).

**Useful options** (same command line as `python app.py`):

| Flag | Meaning |
|------|---------|
| `--port 8080` | Listen on port `8080` instead of `5000` |
| `--host 0.0.0.0` | Allow other devices on your Wi‑Fi/LAN to open the page (less private; use only on trusted networks) |
| `--out_dir path/to/outputs` | Load models from another folder |
| `--debug` | Auto-reload on code changes (development only) |

Stop the server with **Ctrl+C** in the terminal.

#### How to use

1. On the page, click **Choose File** / **Browse** and pick a plant photo from your disk (**JPEG**, **PNG**, **WebP**, or **GIF**; max **16 MB**).
2. Click **Predict**.
3. Read the **predicted class**, **confidence**, and the **score bars** for all eight species.

The image is resized to **128×128** (same as training) before feature extraction. Results are for demonstration only; misclassifications are possible (see [Known Limitations](#known-limitations)).

**Health check (optional):** open **http://127.0.0.1:5000/health** — JSON `{"status":"ok"}` means the model loaded.

---

## Output Files

All files are written to `outputs/` (created automatically).

| File | Description |
|---|---|
| `best_model.pkl` | Best classifier selected by validation accuracy |
| `SVM_RBF.pkl` | Trained SVM model |
| `RandomForest.pkl` | Tuned Random Forest model |
| `GradientBoosting.pkl` | Trained Gradient Boosting model |
| `scaler.pkl` | Fitted `StandardScaler` — **must** be applied before any prediction |
| `pca.pkl` | Fitted PCA transform — applied after scaling |
| `label_encoder.pkl` | Maps integer predictions back to class name strings |
| `val_metrics.json` | Validation metrics per model, RF grid search, **Python/sklearn/skimage versions** (reproducibility), optional **test_eval** if `--eval_test` |
| `train_log.txt` | Full human-readable training log with feature config, hyperparameters, per-class metrics, and environment snapshot |

**After `evaluate.py`:** `test_metrics.txt`, `test_evaluation.json`, `confusion_matrix.png`, `per_class_metrics.png`, `results.csv`, `error_analysis.txt`, `examples/{correct,wrong}/`, `examples_*_sheet.png`.

### Sample `val_metrics.json` structure
```json
{
  "seed": 42,
  "img_size": 128,
  "data_dir": "/path/to/data",
  "feature_dim_raw": 1886,
  "pca_components": 235,
  "environment": {
    "python": "3.11.x",
    "sklearn": "1.x.x",
    "skimage": "0.x.x",
    "numpy": "1.x.x",
    "joblib": "1.x.x",
    "PYTHONHASHSEED": "42",
    "git_commit_short": "abc1234"
  },
  "rf_grid_search": {
    "best_params": { "n_estimators": 300, "max_depth": null, "min_samples_leaf": 1 },
    "best_cv_score": 0.8187
  },
  "classifiers": {
    "SVM_RBF": { "val_accuracy": 0.9375, "macro_f1": 0.9199, "weighted_f1": 0.94, "train_time_s": 0.07 },
    "RandomForest": { "val_accuracy": 0.875, "macro_f1": 0.8616, "weighted_f1": 0.87, "train_time_s": 0.2 },
    "GradientBoosting": { "val_accuracy": 0.8875, "macro_f1": 0.8655, "weighted_f1": 0.88, "train_time_s": 19.0 }
  },
  "best_model": "SVM_RBF",
  "test_eval": {
    "best_model": "SVM_RBF",
    "n_samples": 93,
    "accuracy": 0.9247,
    "macro_f1": 0.9043
  }
}
```
*Field `test_eval` is `null` unless you pass `--eval_test`.*

---

## Design Decisions & Justifications

### Why these three features?

Each descriptor captures a different visual aspect of plant appearance. Using all three in combination gives a richer, more robust representation than any single descriptor alone.

**HOG — Histogram of Oriented Gradients**
- Encodes the distribution of local edge directions across the image
- Plant-specific relevance: leaf margin shape, petal outlines, and bark ridges all produce distinctive orientation histograms
- Robust to moderate illumination changes through block normalisation (L2-Hys)
- Limitation: sensitive to large viewpoint changes (a leaf seen edge-on looks very different from above)

**HSV Colour Histogram**
- Captures dominant colour distribution with less sensitivity to brightness than RGB histograms
- Plant-specific relevance: Hue cleanly separates red Camellia blooms, yellow Xanthostemon flowers, and green foliage; Saturation distinguishes vivid blooms from dried/pale leaves
- Limitation: highly sensitive to major illumination shifts (indoor fluorescent vs. outdoor sun can shift the perceived hue of the same leaf)

**LBP — Local Binary Pattern (uniform)**
- Encodes micro-texture by thresholding each pixel against its circular neighbourhood
- Plant-specific relevance: distinguishes glossy smooth Ficus leaves, coarse-textured Rhododendron leaves, and scale-like Juniperus foliage
- The `uniform` variant keeps only patterns with ≤ 2 binary transitions, giving a compact 26-D descriptor
- Limitation: degrades with motion blur, out-of-focus shots, or extreme scale changes

### Why PCA before classification?

The raw concatenated feature vector is ~1886-D but most of that variance is concentrated in far fewer dimensions. PCA (retaining 95% variance) reduces the vector to ~150–250-D, which:
- Removes noisy low-variance dimensions that could hurt SVM generalisation
- Dramatically speeds up SVM kernel matrix computation
- Has no measurable accuracy cost at the 95% threshold

### Why these three classifiers?

- **SVM (RBF)** — Strong baseline on dense, standardised features after PCA; maximum-margin decision in the kernel-induced space often works well for medium-sized feature vectors. In our pipeline it is a fixed-configuration baseline (no grid search) to keep runtime reasonable.
- **Random Forest** — Robust to feature scale (rank-based splits); we run a **small GridSearchCV** over `n_estimators`, `max_depth`, and `min_samples_leaf` because RF benefits from light tuning and yields interpretable feature importances for error analysis.
- **Gradient Boosting** — Sequential tree ensemble that often complements bagging methods; fixed defaults serve as a third, distinct inductive bias.

The **best** of the three is chosen **only by validation accuracy** (see `best_model` in `val_metrics.json`); on our campus split, RBF SVM often wins after PCA + scaling, but this is data-dependent.

### Why light tuning only for Random Forest?

SVM and GBDT are included as comparison baselines with principled fixed defaults. Full grid search on all three classifiers would increase runtime by 3× with minimal benefit given the small dataset. The RF grid covers 18 combinations (3 × 3 × 2) with 3-fold CV — enough to find the best `n_estimators` / `max_depth` tradeoff without overfitting the validation set.

---

## Known Limitations

| Limitation | Cause | Impact |
|---|---|---|
| Background bias | HOG and colour histograms encode background pixels along with the plant | Model may learn wall colour or sky context rather than plant morphology |
| Illumination sensitivity | HSV histograms shift under different white balances | Indoor Goeppertia vs. outdoor classes may be confused if lighting differs greatly from training |
| Viewpoint sensitivity | HOG is not rotation-invariant | Same plant photographed from below vs. from the side may produce very different HOG vectors |
| Scale sensitivity | Fixed 128×128 resize does not preserve aspect ratio or scale context | A close-up leaf crop and a full-tree shot produce very different feature vectors for the same class |
| Two Ficus classes | Standard vs. golden-leaved cultivar are visually similar at the leaf level | Colour histogram is the primary discriminating feature; confusion between these two classes is expected |

These limitations are analysed quantitatively in Task 4 using the held-out test set confusion matrix and qualitative error examples.

---

## Assignment Task Checklist

| Task | Status | Script / File |
|---|---|---|
| Task 1: Class definition & collection plan | ✅ Done | `Task1_Report_Section.docx` |
| Task 2: Dataset building & labelling | ✅ Done | `Task2_Report_Section.docx` |
| Task 3: Train classifier | ✅ Done | `classifier.py` / `train.py` |
| Task 4: Evaluation & error analysis | ✅ Done | `evaluate.py` |
| Task 5: Application | ✅ Done | `app.py` + `templates/index.html` |
| Task 6: Full report | 🔲 In progress | `Assignment2_Report_<ID>_<Name>.pdf` |

---

## Reproducibility Guarantee

All random seeds are fixed via `--seed 42` (passed to `numpy`, `random`, `os.environ["PYTHONHASHSEED"]`, and scikit-learn estimators that accept `random_state`). `val_metrics.json` records **library versions** (`numpy`, `scikit-learn`, `scikit-image`, `joblib`) and optionally a **short git commit** when training is run inside a git checkout. Re-running `train.py` / `classifier.py` with the same code, data, seed, and dependency versions reproduces the same metrics and models; minor floating-point differences can occur across CPU/BLAS builds.

---