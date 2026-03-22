"""
Traditional ML Training Pipeline
================================================================
Campus Vegetation Recognition

Pipeline overview
-----------------
1. Load images from  <data_dir>/train/<class>/*.{jpg,jpeg,png}
                and  <data_dir>/val/<class>/*
2. Extract features per image:
      • HOG  — shape / edge structure (~1764-D raw)
      • HSV colour histogram — illumination-robust colour distribution (96-D)
      • LBP  — micro-texture descriptor (26-D)
   Concatenated → single feature vector (~1886-D raw → PCA → ~150–250-D)
3. Standardise with StandardScaler (zero mean, unit variance)
4. Train three classifiers:
      • SVM (RBF kernel)          — fixed sensible defaults
      • Random Forest             — light GridSearchCV on n_estimators / max_depth
      • Gradient Boosting (GBDT)  — fixed sensible defaults
5. Select best model by validation accuracy
6. Save: best_model.pkl, scaler.pkl, pca.pkl, label_encoder.pkl, train_log.txt,
   val_metrics.json (includes library versions for reproducibility)

Usage
-----
    python classifier.py
    python train.py                          # thin wrapper; same CLI
    python classifier.py --data_dir data --out_dir outputs --seed 42 --eval_test

Requirements
------------
    pip install -r requirements.txt
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from PIL import Image
import skimage
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CLI arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Task 3 — Vegetation classifier training")
    p.add_argument("--data_dir",  type=str, default="data",
                   help="Root folder with train/ and val/ sub-directories")
    p.add_argument("--out_dir",   type=str, default="outputs",
                   help="Directory for saved models and logs")
    p.add_argument("--img_size",  type=int, default=128,
                   help="Resize all images to (img_size × img_size) before feature extraction")
    p.add_argument("--seed",      type=int, default=42,
                   help="Global random seed for reproducibility")
    p.add_argument("--n_jobs",    type=int, default=-1,
                   help="Parallel jobs for GridSearchCV / RF (-1 = all cores)")
    p.add_argument(
        "--eval_test",
        action="store_true",
        help="If <data_dir>/test exists, evaluate the best model on the held-out test split",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def reproducibility_metadata() -> dict:
    """Versions and environment snapshot for assignment reproducibility reporting."""
    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "skimage": skimage.__version__,
        "joblib": joblib.__version__,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
    }
    try:
        root = Path(__file__).resolve().parent
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if sha.returncode == 0 and sha.stdout.strip():
            meta["git_commit_short"] = sha.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return meta


def metrics_dict(y_true, y_pred, le: LabelEncoder) -> dict:
    """Validation/test metrics bundle for JSON export."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_class": {
            k: v
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path, img_size: int) -> np.ndarray:
    """Load an image, convert to RGB, and resize to (img_size × img_size)."""
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    return np.array(img, dtype=np.uint8)


def extract_hog(img_rgb: np.ndarray) -> np.ndarray:
    """
    HOG on the luminance channel (YCbCr Y).

    Why HOG for plants?
    -------------------
    HOG encodes the distribution of local edge directions.  Leaf margins,
    petal shapes, and bark textures all produce characteristic orientation
    histograms that are largely invariant to moderate illumination changes
    (because HOG normalises within cells) and to small positional shifts
    (because it pools over a cell).  It is, however, sensitive to large
    viewpoint changes — a limitation discussed in Task 4.

    Parameters
    -----------
    orientations=9  — coarse enough to be stable but fine enough to
                       distinguish rounded (Camellia) from spiky (Juniperus)
                       silhouettes.
    pixels_per_cell=(16,16) — large cells reduce noise on 128×128 crops.
    cells_per_block=(2,2)   — standard block normalisation.
    """
    from skimage.color import rgb2gray
    gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def extract_hsv_hist(img_rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Concatenated H / S / V histograms (bins each) → 3*bins-D vector.

    Why HSV histograms for plants?
    ------------------------------
    Hue captures dominant leaf/flower colour (green foliage vs. red Camellia
    vs. yellow Xanthostemon) with less sensitivity to brightness than raw RGB.
    Saturation separates vivid blooms from pale/dried foliage.
    Value (brightness) encodes global exposure and helps detect shade vs.
    direct-sun captures.  We use 32 bins per channel (96-D total) — fine
    enough to separate classes but coarse enough to be robust to minor
    illumination shifts.
    """
    img_hsv = np.array(Image.fromarray(img_rgb).convert("HSV"), dtype=np.float32)
    feats = []
    for ch, max_val in zip(range(3), [180.0, 256.0, 256.0]):
        hist, _ = np.histogram(img_hsv[:, :, ch], bins=bins, range=(0, max_val))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)   # L1 normalise
        feats.append(hist)
    return np.concatenate(feats)


def extract_lbp(img_rgb: np.ndarray, radius: int = 3, n_points: int = 24) -> np.ndarray:
    """
    Uniform LBP histogram on the greyscale image.

    Why LBP for plants?
    -------------------
    LBP captures fine micro-texture — the difference between glossy Ficus
    leaves, the scale-like foliage of Juniperus, and the broad smooth blades
    of Podocarpus is well encoded by local binary patterns.  The 'uniform'
    variant retains only patterns with ≤ 2 binary transitions (which cover
    most real-world textures) and bins the rest into a single 'non-uniform'
    bucket, keeping the descriptor compact (n_points + 2 bins = 26-D here).
    LBP is largely invariant to monotonic illumination changes, though it
    degrades with blur — a limitation explored in Task 4.
    """
    from skimage.color import rgb2gray
    gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)
    lbp_img = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = n_points + 2  # uniform LBP bin count
    hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Concatenate HOG + HSV histogram + LBP into one feature vector.

    Combining complementary descriptors improves robustness:
    - HOG covers shape / structure
    - HSV histogram covers colour distribution
    - LBP covers fine surface texture
    Each descriptor is sensitive to different aspects of appearance, so
    their combination is more discriminative than any single descriptor.
    """
    hog_feat   = extract_hog(img_rgb)          # ~1764-D (before PCA)
    color_feat = extract_hsv_hist(img_rgb)     # 96-D
    lbp_feat   = extract_lbp(img_rgb)          # 26-D
    return np.concatenate([hog_feat, color_feat, lbp_feat])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_split(split_dir: Path, img_size: int, label_encoder: LabelEncoder = None):
    """
    Load all images from split_dir/<class_name>/*.jpg|png|jpeg.

    Returns
    -------
    X : np.ndarray, shape (N, feature_dim)
    y : np.ndarray, shape (N,)   — integer-encoded labels
    label_encoder : fitted LabelEncoder (returned / reused for consistency)
    """
    image_paths, labels = [], []

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class sub-directories found in {split_dir}")

    class_names = [d.name for d in class_dirs]
    print(f"\n  Classes found ({len(class_names)}): {class_names}")

    for class_dir in class_dirs:
        imgs = list(class_dir.glob("*.jpg")) + \
               list(class_dir.glob("*.jpeg")) + \
               list(class_dir.glob("*.png"))
        for p in imgs:
            image_paths.append(p)
            labels.append(class_dir.name)

    print(f"  Total images: {len(image_paths)}")

    # Fit or reuse label encoder
    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(class_names)

    # Extract features
    X, y = [], []
    for path, label in tqdm(zip(image_paths, labels),
                            total=len(image_paths),
                            desc=f"  Extracting features from {split_dir.name}/"):
        try:
            img = load_image(path, img_size)
            feat = extract_features(img)
            X.append(feat)
            y.append(label_encoder.transform([label])[0])
        except Exception as e:
            print(f"  [WARN] Skipping {path.name}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_encoder


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Classifier definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_classifiers(seed: int, n_jobs: int = -1):
    """
    Return a dict of {name: classifier} with justified default settings.

    SVM (RBF kernel)
    ----------------
    C=10, gamma='scale' are well-established defaults for normalised
    feature spaces.  SVM with RBF kernel finds a maximum-margin boundary in
    a high-dimensional implicit feature space, which works well when the
    feature vectors are normalised and the number of features is comparable
    to the number of samples.  probability=True enables Platt-style
    probability outputs; random_state fixes the calibration RNG for
    reproducibility.

    Random Forest
    -------------
    n_estimators=200 (will be tuned), max_features='sqrt' (standard for
    classification), min_samples_leaf=2 (light regularisation to reduce
    overfitting on smaller classes).  RF is robust to scale differences
    between feature groups (HOG values vs. normalised histograms) because it
    uses rank-based splits.

    Gradient Boosting
    -----------------
    n_estimators=200, learning_rate=0.1, max_depth=4, subsample=0.8.
    GBDT builds trees sequentially to correct residuals of the ensemble.
    subsample=0.8 adds stochasticity for better generalisation.  Fixed
    defaults are used here; GBDT is included primarily as a comparison point.
    """
    return {
        "SVM_RBF": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,       # starting point; tuned by GridSearchCV
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            random_state=seed,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Light hyperparameter tuning for Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def tune_random_forest(X_train, y_train, seed: int, n_jobs: int):
    """
    GridSearchCV over a small, principled grid for Random Forest.

    Rationale
    ---------
    RF is famously robust; n_estimators=100–300 with max_features='sqrt'
    covers the sweet spot.  We add max_depth to prevent individual trees from
    overfitting small classes and min_samples_leaf for regularisation.
    The 3-fold StratifiedKFold (rather than 5-fold) keeps runtime reasonable
    while still giving a stable estimate of generalisation performance.

    Grid
    ----
    n_estimators  : [100, 200, 300]  — diminishing returns beyond 300
    max_depth     : [None, 20, 30]   — None = fully grown (default)
    min_samples_leaf: [1, 2]         — 1 = standard; 2 = light regularisation
    """
    print("\n  [GridSearchCV] Tuning Random Forest ...")
    param_grid = {
        "n_estimators":   [100, 200, 300],
        "max_depth":      [None, 20, 30],
        "min_samples_leaf": [1, 2],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rf_base = RandomForestClassifier(
        max_features="sqrt", n_jobs=n_jobs, random_state=seed
    )
    gs = GridSearchCV(
        rf_base, param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"  Best RF params : {gs.best_params_}")
    print(f"  Best CV score  : {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir  = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    log_lines = []  # accumulated log entries

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("COMP4423 Assignment 2 — Task 3: Vegetation Classifier Training")
    log("=" * 70)
    log(f"Data directory : {data_dir.resolve()}")
    log(f"Output directory: {out_dir.resolve()}")
    log(f"Image size     : {args.img_size}×{args.img_size}")
    log(f"Random seed    : {args.seed}")
    log(f"Parallel jobs  : {args.n_jobs}")

    # ── 7.1  Load datasets ────────────────────────────────────────────────
    log("\n[1/5] Loading & extracting features ...")
    t0 = time.time()

    log("\n  Training set:")
    X_train, y_train, le = load_split(train_dir, args.img_size)

    log("\n  Validation set:")
    X_val, y_val, _     = load_split(val_dir, args.img_size, label_encoder=le)

    log(f"\n  Feature vector dimensionality (raw): {X_train.shape[1]}")
    log(f"  Train samples : {X_train.shape[0]}")
    log(f"  Val   samples : {X_val.shape[0]}")
    log(f"  Classes       : {[str(c) for c in le.classes_]}")
    log(f"  Feature extraction time: {time.time()-t0:.1f}s")

    # ── 7.2  Standardise ──────────────────────────────────────────────────
    log("\n[2/5] Standardising features ...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    # ── 7.3  PCA dimensionality reduction ─────────────────────────────────
    # Keep enough components to explain 95% of variance.
    # Reducing ~1886-D → ~200-D speeds up SVM training significantly and
    # can improve generalisation by removing noisy low-variance dimensions.
    log("\n[3/5] Applying PCA (95% variance threshold) ...")
    pca = PCA(n_components=0.95, random_state=args.seed)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca   = pca.transform(X_val_sc)
    log(f"  PCA kept {pca.n_components_} components "
        f"(from {X_train.shape[1]} → {X_train_pca.shape[1]})")

    # ── 7.4  Train classifiers ────────────────────────────────────────────
    log("\n[4/5] Training classifiers ...")
    classifiers = build_classifiers(args.seed, n_jobs=args.n_jobs)

    # Light GridSearchCV for Random Forest only
    rf_tuned, rf_best_params, rf_cv_score = tune_random_forest(
        X_train_pca, y_train, seed=args.seed, n_jobs=args.n_jobs
    )
    classifiers["RandomForest"] = rf_tuned   # replace default with tuned version

    results = {}
    trained_models = {}

    for name, clf in classifiers.items():
        log(f"\n  Training {name} ...")
        t_start = time.time()
        clf.fit(X_train_pca, y_train)
        t_elapsed = time.time() - t_start

        y_pred = clf.predict(X_val_pca)
        md = metrics_dict(y_val, y_pred, le)
        results[name] = {
            "val_accuracy":     md["accuracy"],
            "macro_f1":         md["macro_f1"],
            "weighted_f1":      md["weighted_f1"],
            "train_time_s":     round(t_elapsed, 2),
            "per_class":        md["per_class"],
            "confusion_matrix": md["confusion_matrix"],
        }

        log(f"    Val accuracy : {md['accuracy']:.4f}")
        log(f"    Macro F1     : {md['macro_f1']:.4f}")
        log(f"    Train time   : {t_elapsed:.1f}s")
        log(
            f"\n{classification_report(y_val, y_pred, target_names=le.classes_, zero_division=0)}"
        )

        trained_models[name] = clf

    # ── 7.5  Select best model ────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["val_accuracy"])
    best_acc  = results[best_name]["val_accuracy"]
    log(f"\n[5/5] Best model: {best_name}  (val accuracy = {best_acc:.4f})")

    test_eval = None
    if args.eval_test:
        test_dir = data_dir / "test"
        if test_dir.is_dir():
            log("\n  Held-out test evaluation (best model only) ...")
            X_test, y_test, _ = load_split(test_dir, args.img_size, label_encoder=le)
            X_test_pca = pca.transform(scaler.transform(X_test))
            y_hat = trained_models[best_name].predict(X_test_pca)
            tm = metrics_dict(y_test, y_hat, le)
            test_eval = {
                "best_model": best_name,
                "n_samples": int(len(y_test)),
                "accuracy": tm["accuracy"],
                "macro_f1": tm["macro_f1"],
                "weighted_f1": tm["weighted_f1"],
                "per_class": tm["per_class"],
                "confusion_matrix": tm["confusion_matrix"],
            }
            log(f"    Test accuracy : {tm['accuracy']:.4f}")
            log(f"    Test macro F1 : {tm['macro_f1']:.4f}")
            log(
                f"\n{classification_report(y_test, y_hat, target_names=le.classes_, zero_division=0)}"
            )
        else:
            log(f"\n  [--eval_test] No directory {test_dir} — skipping test evaluation.")

    # ── 7.6  Save artefacts ───────────────────────────────────────────────
    log("\n  Saving model artefacts ...")
    joblib.dump(trained_models[best_name], out_dir / "best_model.pkl")
    joblib.dump(scaler,                    out_dir / "scaler.pkl")
    joblib.dump(pca,                       out_dir / "pca.pkl")
    joblib.dump(le,                        out_dir / "label_encoder.pkl")

    # Save all models (so Task 4 can load any of them)
    for name, clf in trained_models.items():
        joblib.dump(clf, out_dir / f"{name}.pkl")

    env = reproducibility_metadata()
    summary = {
        "seed": args.seed,
        "img_size": args.img_size,
        "data_dir": str(data_dir.resolve()),
        "feature_dim_raw": int(X_train.shape[1]),
        "pca_components": int(pca.n_components_),
        "environment": env,
        "rf_grid_search": {
            "best_params": rf_best_params,
            "best_cv_score": round(rf_cv_score, 4),
        },
        "classifiers": results,
        "best_model": best_name,
        "test_eval": test_eval,
    }
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # train_log.txt
    log_lines.append("\n" + "=" * 70)
    log_lines.append("FEATURE EXTRACTION SUMMARY")
    log_lines.append("=" * 70)
    log_lines.append("HOG  : 9 orientations, 16×16 cells, 2×2 block normalisation (L2-Hys)")
    log_lines.append("HSV  : 32-bin histogram per channel (H/S/V), L1-normalised → 96-D")
    log_lines.append("LBP  : uniform, radius=3, 24 points → 26-D histogram")
    log_lines.append(f"Concat raw dim : {X_train.shape[1]}")
    log_lines.append(f"After PCA (95%) : {pca.n_components_} components")
    log_lines.append("\nRANDOM FOREST GRID SEARCH")
    log_lines.append(f"  Best params   : {rf_best_params}")
    log_lines.append(f"  Best CV score : {rf_cv_score:.4f}")
    log_lines.append("\nREPRODUCIBILITY")
    for k, v in env.items():
        log_lines.append(f"  {k}: {v}")
    if test_eval is not None:
        log_lines.append("\nHELD-OUT TEST (best model)")
        log_lines.append(f"  accuracy: {test_eval['accuracy']}")
        log_lines.append(f"  macro_f1: {test_eval['macro_f1']}")

    with open(out_dir / "train_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    log(f"\n  Saved: best_model.pkl, scaler.pkl, pca.pkl, label_encoder.pkl")
    log(f"         val_metrics.json, train_log.txt")
    log(f"         {', '.join(f'{n}.pkl' for n in trained_models)}")
    log("\nDone.")


if __name__ == "__main__":
    main()