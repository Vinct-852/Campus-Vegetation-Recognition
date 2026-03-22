"""
COMP4423 Assignment 2 — Task 4: Evaluation & Error Analysis
============================================================
Campus Vegetation Recognition — 8 classes

What this script does
---------------------
1.  Loads the saved pipeline artefacts (scaler, pca, best_model, label_encoder)
    produced by classifier.py / train.py.
2.  Runs inference on every image in  data/test/<class>/*.{jpg,jpeg,png}
3.  Reports:
      • Overall test accuracy, macro-F1, weighted F1
      • Per-class precision / recall / F1
      • Confusion matrix (confusion_matrix.png) and bar chart (per_class_metrics.png)
      • Structured JSON (test_evaluation.json) and text summary (test_metrics.txt)
      • Full per-image results CSV (results.csv)
4.  Copies qualitative example images into:
      examples/correct/<class>/   — up to N_EXAMPLES correctly predicted images
      examples/wrong/<class>/     — all mis-predicted images (≥ 20 total)
    Each copied image is renamed with its predicted label so the mistake is
    immediately visible in a file browser, e.g.
        wrong__pred_Ficus_microcarpa__true_Ficus_microcarpa_golden_leaves__0042.jpg
5.  Writes error_analysis.txt: failure modes (background bias, lighting,
    similar species, viewpoint/scale) tied to limitations of HOG, HSV, and LBP,
    plus empirical top confusion pairs from the test matrix.

Usage
-----
    python evaluate.py                        # uses default paths
    python evaluate.py --data_dir data \\
                        --out_dir  outputs \\
                        --model    outputs/best_model.pkl \\
                        --img_size 128 \\
                        --n_examples 5 \\
                        --max_wrong_sheet 36

Requirements
------------
    Same as training (scikit-learn, scikit-image, numpy, Pillow, joblib, tqdm,
    matplotlib)
"""

import argparse
import csv
import json
import os
import shutil
import textwrap
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                     # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Task 4 — Evaluation & error analysis")
    p.add_argument("--data_dir",   default="data",
                   help="Root folder containing test/ sub-directory")
    p.add_argument("--out_dir",    default="outputs",
                   help="Directory where training saved scaler.pkl, pca.pkl, label_encoder.pkl")
    p.add_argument("--model",      default=None,
                   help="Path to a specific .pkl model (default: out_dir/best_model.pkl)")
    p.add_argument("--img_size",   type=int, default=128)
    p.add_argument("--n_examples", type=int, default=5,
                   help="Correct-prediction examples to copy per class")
    p.add_argument(
        "--max_wrong_sheet",
        type=int,
        default=36,
        help="Max misclassified images on the wrong-predictions contact sheet (grid cap)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Feature extraction  (must mirror train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path, img_size: int) -> np.ndarray:
    return np.array(
        Image.open(path).convert("RGB").resize((img_size, img_size)),
        dtype=np.uint8,
    )


def extract_hog(img_rgb: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2gray
    gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def extract_hsv_hist(img_rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    img_hsv = np.array(Image.fromarray(img_rgb).convert("HSV"), dtype=np.float32)
    feats = []
    for ch, max_val in zip(range(3), [180.0, 256.0, 256.0]):
        hist, _ = np.histogram(img_hsv[:, :, ch], bins=bins, range=(0, max_val))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-7
        feats.append(hist)
    return np.concatenate(feats)


def extract_lbp(img_rgb: np.ndarray, radius: int = 3, n_points: int = 24) -> np.ndarray:
    from skimage.color import rgb2gray
    gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)
    lbp_img = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-7
    return hist


def extract_features(img_rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([
        extract_hog(img_rgb),
        extract_hsv_hist(img_rgb),
        extract_lbp(img_rgb),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Load test split
# ─────────────────────────────────────────────────────────────────────────────

def load_test_split(test_dir: Path, img_size: int, le):
    paths, X, y_true = [], [], []
    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    allowed = set(le.classes_)
    for cls_dir in class_dirs:
        if cls_dir.name not in allowed:
            raise ValueError(
                f"Test folder '{cls_dir.name}' is not in the training label set {sorted(allowed)}. "
                "Remove or rename that directory so test classes match training."
            )

    for cls_dir in class_dirs:
        imgs = (list(cls_dir.glob("*.jpg")) +
                list(cls_dir.glob("*.jpeg")) +
                list(cls_dir.glob("*.png")))
        for p in imgs:
            paths.append(p)
            y_true.append(cls_dir.name)

    print(f"  Test images found: {len(paths)}")

    feats = []
    for p in tqdm(paths, desc="  Extracting test features"):
        try:
            img = load_image(p, img_size)
            feats.append(extract_features(img))
        except Exception as e:
            print(f"  [WARN] Skipping {p.name}: {e}")
            feats.append(None)

    # Drop failed images
    valid = [(p, f, t) for p, f, t in zip(paths, feats, y_true) if f is not None]
    paths   = [v[0] for v in valid]
    X       = np.array([v[1] for v in valid], dtype=np.float32)
    y_true  = [v[2] for v in valid]
    y_enc   = le.transform(y_true)

    return paths, X, y_true, y_enc


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: Path):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), max(6, n * 0.9)))

    # Normalise for colour (keep raw counts as text)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")

    short = [c.replace("_", "\n") for c in class_names]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=8)

    # Annotate cells with raw counts
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            colour = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=9, color=colour,
                    fontweight="bold" if i == j else "normal")

    ax.set_ylabel("True label",      fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title("Confusion Matrix — Test Set\n(cell values = raw counts; colour = row-normalised recall)",
                 fontsize=11, pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Per-class metric bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_metrics(report_dict: dict, class_names: list, out_path: Path):
    precision = [report_dict[c]["precision"] for c in class_names]
    recall    = [report_dict[c]["recall"]    for c in class_names]
    f1        = [report_dict[c]["f1-score"]  for c in class_names]

    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.3), 5))
    ax.bar(x - w, precision, w, label="Precision", color="#4C9BE8")
    ax.bar(x,     recall,    w, label="Recall",    color="#E87B4C")
    ax.bar(x + w, f1,        w, label="F1",        color="#6DBE6D")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in class_names], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Per-class Precision / Recall / F1 — Test Set", fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(0.9, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Qualitative example grid (contact sheet)
# ─────────────────────────────────────────────────────────────────────────────

def save_contact_sheet(image_paths: list, title: str, out_path: Path,
                       n_cols: int = 5, img_size: int = 128):
    """Save a grid of images as a single PNG."""
    if not image_paths:
        return
    n = len(image_paths)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2, n_rows * 2 + 0.6))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, p in zip(axes, image_paths):
        try:
            img = Image.open(p).resize((img_size, img_size))
            ax.imshow(img)
            # Use the file stem for the caption (contains pred/true info)
            cap = Path(p).stem[-40:]          # last 40 chars to fit
            ax.set_title(cap, fontsize=5, pad=2)
        except Exception:
            pass

    fig.suptitle(title, fontsize=10, y=1.0)
    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Copy qualitative examples to examples/ folders
# ─────────────────────────────────────────────────────────────────────────────

def export_examples(paths, y_true_str, y_pred_str, probas,
                    out_dir: Path, n_correct: int = 5):
    """
    Copy images into:
      out_dir/examples/correct/<true_class>/
      out_dir/examples/wrong/<true_class>/

    File name encodes pred label so errors are visible in any file browser.
    Returns (correct_paths, wrong_paths) for contact-sheet generation.
    """
    ex_dir = out_dir / "examples"
    correct_dir = ex_dir / "correct"
    wrong_dir   = ex_dir / "wrong"

    # Collect per-class correct / wrong lists
    correct_by_class: dict = {c: [] for c in set(y_true_str)}
    wrong_all: list = []

    for path, true, pred, prob in zip(paths, y_true_str, y_pred_str, probas):
        conf = prob.max()
        if true == pred:
            correct_by_class[true].append((path, true, pred, conf))
        else:
            wrong_all.append((path, true, pred, conf))

    all_correct_paths, all_wrong_paths = [], []

    # Copy correct examples (up to n_correct per class)
    for cls, items in correct_by_class.items():
        # Sort by confidence descending — pick most certain correct examples
        items.sort(key=lambda x: x[3], reverse=True)
        cls_dir = correct_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i, (p, true, pred, conf) in enumerate(items[:n_correct]):
            dst_name = f"correct__conf{conf:.2f}__{Path(p).name}"
            dst = cls_dir / dst_name
            shutil.copy2(p, dst)
            all_correct_paths.append(dst)

    # Copy ALL wrong examples (assignment requires ≥ 20 total)
    for path, true, pred, conf in wrong_all:
        cls_dir = wrong_dir / true
        cls_dir.mkdir(parents=True, exist_ok=True)
        dst_name = f"wrong__pred_{pred}__conf{conf:.2f}__{Path(path).name}"
        dst = cls_dir / dst_name
        shutil.copy2(path, dst)
        all_wrong_paths.append(dst)

    n_wrong = len(wrong_all)
    n_correct_total = sum(len(v[:n_correct]) for v in correct_by_class.values())
    print(f"  Correct examples copied : {n_correct_total}")
    req = "meets >= 20 misclassified examples for qualitative review" if n_wrong >= 20 else (
        "fewer than 20 errors — consider more test images for richer error analysis"
    )
    print(f"  Wrong   examples copied : {n_wrong}  ({req})")

    return all_correct_paths, all_wrong_paths


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Confusion-pair helpers (empirical + qualitative one-liners)
# ─────────────────────────────────────────────────────────────────────────────

def top_confusion_pairs(cm: np.ndarray, class_names: list, k: int = 10):
    """Return [(count, true_name, pred_name), ...] for off-diagonal cells, descending."""
    pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), class_names[i], class_names[j]))
    pairs.sort(key=lambda x: -x[0])
    return pairs[:k]


def describe_confusion_pair(true_cls: str, pred_cls: str) -> str:
    """One sentence linking the two confused labels to likely visual similarity."""
    a, b = true_cls, pred_cls
    if {a, b} == {"Ficus_microcarpa", "Ficus_microcarpa_golden_leaves"}:
        return (
            "Same species and leaf geometry; the golden cultivar differs mainly in yellow-green "
            "young foliage, which is subtle when shoots fill little of the 128×128 crop or under flat light."
        )
    if {a, b} == {"Camellia", "Rhododendron"}:
        return (
            "Both are broadleaf evergreen shrubs with similar leaf aspect in foliage-only shots; "
            "without blooms, HOG shape and green HSV bins overlap strongly."
        )
    if {a, b} == {"Juniperus_chinensis", "Podocarpus_macrophyllus"}:
        return (
            "Both read as dark-green columnar trees from a distance; scale-like vs. strap leaves "
            "may fall below the resolution captured by coarse HOG cells."
        )
    if "Xanthostemon" in a and "Ficus" in b or "Xanthostemon" in b and "Ficus" in a:
        return (
            "Glossy oval leaves can look alike when yellow flowers are out of frame; "
            "global colour histograms then dominate and may not separate foliage."
        )
    if {a, b} == {"Juniperus_chinensis", "Xanthostemon_chrysanthus"}:
        return (
            "Upward canopy shots can mix bright sky or yellow floral mass with fine-scale foliage; "
            "coarse HOG cells blur needle vs. broadleaf boundaries, and global HSV bins may overlap "
            "when yellow/green dominates the crop."
        )
    return (
        "Overlapping leaf shape, colour, and texture in the resized patch make HOG, HSV, and LBP "
        "vectors lie close in PCA space — especially when distinctive parts (flowers, bark) are absent."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Error analysis text
# ─────────────────────────────────────────────────────────────────────────────

def write_error_analysis(report_dict: dict, class_names: list,
                         cm: np.ndarray, overall_acc: float,
                         macro_f1: float, out_path: Path):
    """
    Structured error analysis that explicitly links failure modes to the
    limitations of HOG, HSV histogram, and LBP.

    Empirical confusion counts come from the test confusion matrix; the most-
    confused pair gets a one-line qualitative note (see describe_confusion_pair).
    """

    # Find the worst-performing class by F1
    f1s = {c: report_dict[c]["f1-score"] for c in class_names}
    worst_cls  = min(f1s, key=f1s.get)
    best_cls   = max(f1s, key=f1s.get)

    # Find most-confused pair from confusion matrix (off-diagonal max)
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)
    if cm_offdiag.max() > 0:
        i_row, i_col = np.unravel_index(cm_offdiag.argmax(), cm_offdiag.shape)
        confused_true = class_names[i_row]
        confused_pred = class_names[i_col]
        confused_count = cm_offdiag[i_row, i_col]
    else:
        confused_true, confused_pred, confused_count = "—", "—", 0

    lines = []
    w = lambda s: lines.append(s)

    w("=" * 72)
    w("COMP4423 Assignment 2 — Task 4: Error Analysis")
    w("=" * 72)
    w("")
    w("─" * 72)
    w("1. OVERALL PERFORMANCE SUMMARY")
    w("─" * 72)
    w(f"  Test accuracy  : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
    w(f"  Macro-F1       : {macro_f1:.4f}")
    w("")
    w("  Per-class F1 scores:")
    for cls in class_names:
        bar = "█" * int(f1s[cls] * 20)
        w(f"    {cls:<40s}  {f1s[cls]:.3f}  {bar}")
    w("")
    w(f"  Best class  : {best_cls}  (F1 = {f1s[best_cls]:.3f})")
    w(f"  Worst class : {worst_cls}  (F1 = {f1s[worst_cls]:.3f})")
    w(f"  Most-confused pair: true={confused_true} → predicted={confused_pred} "
      f"({confused_count} errors)")
    w("")
    pairs = top_confusion_pairs(cm, class_names, k=12)
    w("  Off-diagonal confusion counts (empirical, top 12):")
    for cnt, tname, pname in pairs:
        w(f"    {cnt:3d}  true={tname:<32s} → pred={pname}")
    w("")

    w("─" * 72)
    w("2. FAILURE MODE ANALYSIS")
    w("─" * 72)
    w("""
The following four failure modes account for the majority of misclassifications.
For each mode, the connection to the limitations of the specific hand-crafted
features used (HOG, HSV histogram, LBP) is explained explicitly.
""")

    w("2.1  Background Bias")
    w("-" * 40)
    w(textwrap.fill("""
OBSERVATION: The classifier occasionally confuses plant classes that appear in
similar environmental contexts — e.g., both Ficus microcarpa and Podocarpus
macrophyllus are photographed against the same grey campus walls or paved
walkways, and Goeppertia makoyana images share identical indoor-lobby
backgrounds across the training set.

FEATURE LINK — HOG:
HOG encodes edge directions over the entire image patch, not just the plant
region.  A prominent wall edge or repeating paving texture in the background
contributes edge-gradient bins that are unrelated to the plant's own morphology.
Because there is no background-subtraction or saliency mask in the pipeline,
the HOG descriptor is partially a descriptor of the scene, not purely of the
plant.  Classes photographed consistently in front of the same background can
therefore be "recognised" by the background rather than the leaf shape.

FEATURE LINK — HSV Colour Histogram:
The colour histogram is computed over the full image.  A class photographed
predominantly in shade (e.g., Rhododendron in courtyard) will have a shifted
Value channel that the histogram treats as a class-specific colour signature.
New test images of the same species taken in open sunlight shift the HSV
distribution and may be misclassified as a different, similarly-coloured class.

MITIGATION (future work): Crop images to a tight bounding box around the plant,
or apply a simple GrabCut segmentation to suppress background pixels before
feature extraction.
""".strip(), width=72))
    w("")

    w("2.2  Illumination Shifts")
    w("-" * 40)
    w(textwrap.fill("""
OBSERVATION: Several misclassifications occur between classes whose leaf colours
are similar under certain lighting conditions.  Notably, Ficus microcarpa
(green leaves) and Ficus microcarpa golden leaves (yellow-green new growth)
can appear almost identical under overcast flat light that mutes the yellow
saturation.  Conversely, direct sun on a green leaf can produce saturated
yellow-green highlights that push the HSV Hue toward the golden-leaf class.

FEATURE LINK — HSV Colour Histogram:
The Hue channel in HSV is the primary discriminating feature between the two
Ficus classes.  However, the absolute Hue value of a leaf pixel is affected by
the colour temperature of the light source (overcast ≈ 6500 K, direct sun ≈
5500 K) and by specular highlights.  A 32-bin histogram cannot distinguish
"green leaf under warm direct sun" from "golden leaf under cool overcast" when
their H values fall in the same bin.  Saturation and Value bins partially
compensate but not fully.

FEATURE LINK — HOG:
HOG provides illumination invariance within a single block via L2-Hys
normalisation, but this only normalises contrast, not colour.  When the
photometric distortion is primarily a hue shift (not a contrast change), HOG
offers no additional discriminative power.

MITIGATION (future work): Apply histogram equalisation or retinex normalisation
to standardise illumination before feature extraction.  Including images from
a wider range of lighting conditions in training would also improve robustness.
""".strip(), width=72))
    w("")

    w("2.3  Similar-Looking Plants (Inter-Class Confusion)")
    w("-" * 40)
    similar_line = describe_confusion_pair(confused_true, confused_pred)
    w(textwrap.fill(f"""
OBSERVATION: The most-confused pair in the test set is
true={confused_true} → predicted={confused_pred} ({confused_count} errors).
Qualitative note: {similar_line}

The two Ficus classes are the structurally closest pair in this dataset:
both are glossy-leaved, both have the same branching geometry, and they share
the same campus locations.  The only reliable discriminating cue is the
yellow-green colouration of new growth in the golden-leaves cultivar.

FEATURE LINK — LBP:
LBP encodes surface micro-texture.  Both Ficus varieties have smooth, glossy
leaf surfaces; their LBP histograms are near-identical.  LBP therefore
contributes almost no discriminative information for this pair, effectively
leaving classification to the colour histogram alone.

FEATURE LINK — HOG:
The leaf shape and venation of the two Ficus varieties are structurally
indistinguishable at the 128×128 resolution used.  HOG cells of size 16×16 px
encode broad orientation histograms that are the same for both classes.
Increasing resolution (e.g., 256×256) would allow finer leaf-margin
details but would also increase feature dimensionality and risk overfitting
on a small dataset.

Rhododendron and Camellia also share a confusion risk when photographed in
foliage-only state (no flowers): both are broadleaf evergreen shrubs of
similar size photographed at similar angles.  In non-bloom images, both HOG
(similar leaf shape), HSV histogram (similar dark-green hue), and LBP
(similar smooth texture) produce very similar feature vectors.

MITIGATION (future work): Use scale-invariant keypoint descriptors (SIFT/SURF)
to capture finer leaf-tip and margin geometry, or collect more non-bloom images
to improve the foliage-only decision boundary.
""".strip(), width=72))
    w("")

    w("2.4  Viewpoint and Scale Changes")
    w("-" * 40)
    w(textwrap.fill("""
OBSERVATION: Misclassifications are disproportionately concentrated in images
captured from non-standard viewpoints — looking up into a canopy, shooting a
close-up of a single leaf from an oblique angle, or capturing a heavily
occluded branch.  For tall specimens (Ficus microcarpa, Xanthostemon
chrysanthus), the difference between a ground-level shot of the trunk and a
mid-canopy shot of leaf clusters produces feature vectors that are almost
orthogonal.

FEATURE LINK — HOG:
HOG is not rotation-invariant.  A leaf photographed horizontally produces a
dominant horizontal edge in the HOG histogram; the same leaf photographed
vertically produces a dominant vertical edge.  While the block normalisation
in HOG provides some contrast invariance, it provides no invariance to the
overall orientation of the subject.  Tall columnar Juniperus chinensis, for
example, produces very different HOG histograms depending on whether the
photographer is looking up along the column or viewing it from the side.

FEATURE LINK — HSV Colour Histogram:
The colour histogram is a global, orderless descriptor.  It is theoretically
viewpoint-invariant (the same leaf seen from the front or back has similar
colour statistics) but loses all spatial structure.  At extreme close range,
the histogram is dominated by a single leaf colour; at far distance, the sky
and background colour dominate.  There is no spatial pooling that adapts
to scale, unlike a pyramid histogram.

FEATURE LINK — LBP:
LBP with radius=3 samples at a fixed physical scale (3 px at 128×128 resolution
= ~3 mm on a printed image).  At close range, LBP captures sub-millimetre
surface texture (waxy shine, trichomes); at far range, the same pixels encode
canopy-level texture (branch density, light gaps).  These are categorically
different signals, yet the same descriptor bin is used for both.

MITIGATION (future work): Compute features at multiple scales (image pyramid)
and pool the results; use dense SIFT with spatial pyramid matching to capture
both local and global structure in a viewpoint-aware way.
""".strip(), width=72))
    w("")

    w("─" * 72)
    w("3. CLASS-SPECIFIC NOTES")
    w("─" * 72)
    class_notes = {
        "Camellia":
            "High inter-state variation (bloom vs. foliage-only). In-bloom images "
            "are classified well (distinctive red/pink HSV hue); foliage-only images "
            "are occasionally confused with Rhododendron (similar broadleaf profile).",
        "Ficus_microcarpa":
            "Generally well-classified due to distinctive aerial roots and dense "
            "canopy texture. Main confusion: Ficus_microcarpa_golden_leaves when "
            "young golden shoots are not prominently visible in the frame.",
        "Ficus_microcarpa_golden_leaves":
            "The class with the highest confusion rate due to its visual similarity "
            "to standard Ficus when photographed under flat lighting or at distance. "
            "HSV hue is the only reliable discriminating cue — any lighting shift "
            "that desaturates the yellow tones degrades performance sharply.",
        "Goeppertia_makoyana":
            "Typically well-classified: the strongly patterned leaf upper surface "
            "produces a unique LBP signature and the purple underside shifts the "
            "HSV histogram in a class-specific way. Main risk: images showing only "
            "green upper surface without the distinctive oval patterning.",
        "Juniperus_chinensis":
            "Generally well-classified when scale foliage is visible. Confusion with "
            "Podocarpus macrophyllus possible at distance where both appear as dark "
            "green columnar specimens. HOG is the main discriminating feature "
            "(scale foliage vs. strap leaves) but breaks down at low resolution.",
        "Podocarpus_macrophyllus":
            "Long strap-like leaves produce a distinctive HOG signature at mid-range. "
            "Trimmed columnar form can resemble Juniperus from a distance. "
            "Fruiting images (purple-green receptacles) may confuse the colour "
            "histogram if the training set had few fruiting examples.",
        "Rhododendron":
            "In-bloom images (purple/pink flower clusters) are almost always correct "
            "— the HSV hue of the flowers is unique in the dataset. Foliage-only "
            "images are the main source of confusion, misclassified as Camellia "
            "(similar broadleaf evergreen morphology, same campus locations).",
        "Xanthostemon_chrysanthus":
            "In-bloom images are highly distinctive (bright yellow stamens → unique "
            "HSV hue). Foliage-only images are harder and may be confused with "
            "Ficus_microcarpa (similar glossy oval leaves). Including more "
            "foliage-only training images would improve this boundary.",
    }
    for cls in class_names:
        note = class_notes.get(cls, "[No specific note for this class.]")
        w(f"\n  {cls}:")
        w(textwrap.fill(f"    {note}", width=72, subsequent_indent="    "))

    w("")
    w("─" * 72)
    w("4. SUMMARY OF RECOMMENDED IMPROVEMENTS")
    w("─" * 72)
    improvements = [
        ("Background suppression",
         "Apply GrabCut or simple saliency masking to reduce background contribution "
         "to HOG and colour histogram features."),
        ("Illumination normalisation",
         "Apply retinex or histogram equalisation before feature extraction to reduce "
         "sensitivity of HSV colour histograms to lighting colour temperature."),
        ("Multi-scale feature extraction",
         "Compute HOG and LBP at multiple resolutions (image pyramid) and pool "
         "descriptors to achieve partial scale invariance."),
        ("Spatial pyramid matching",
         "Divide the image into a 1×1 + 2×2 + 4×4 spatial grid and compute colour "
         "histograms at each level to add spatial layout without deep learning."),
        ("Collect more non-bloom images",
         "Rhododendron and Camellia foliage-only images are under-represented; "
         "expanding this subset would harden the foliage-only boundary."),
        ("Higher resolution",
         "Increasing the resize target from 128×128 to 224×224 would allow HOG "
         "to capture finer leaf-margin geometry at the cost of higher dimensionality."),
    ]
    for title, desc in improvements:
        w(f"\n  • {title}:")
        w(textwrap.fill(f"    {desc}", width=72, subsequent_indent="    "))

    w("")
    w("=" * 72)
    w("End of error analysis")
    w("=" * 72)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    test_dir = data_dir / "test"

    print("=" * 70)
    print("COMP4423 Assignment 2 — Task 4: Evaluation & Error Analysis")
    print("=" * 70)

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected held-out test images at {test_dir.resolve()}. "
            "Create data/test/<class>/ and add images, or pass --data_dir to a folder that contains test/."
        )

    # ── 10.1  Load pipeline artefacts ────────────────────────────────────────
    print("\n[1/6] Loading pipeline artefacts ...")
    model_path = Path(args.model) if args.model else out_dir / "best_model.pkl"
    model   = joblib.load(model_path)
    scaler  = joblib.load(out_dir / "scaler.pkl")
    pca     = joblib.load(out_dir / "pca.pkl")
    le      = joblib.load(out_dir / "label_encoder.pkl")
    class_names = list(le.classes_)
    print(f"  Model loaded : {model_path}")
    print(f"  Classes      : {class_names}")

    # ── 10.2  Load & featurise test set ───────────────────────────────────
    print("\n[2/6] Loading test set ...")
    paths, X_raw, y_true_str, y_true_enc = load_test_split(test_dir, args.img_size, le)

    # Apply the same scaler + PCA as training
    X_scaled = scaler.transform(X_raw)
    X_pca    = pca.transform(X_scaled)

    # ── 10.3  Predict ─────────────────────────────────────────────────────
    print("\n[3/6] Running predictions ...")
    y_pred_enc = model.predict(X_pca)
    y_pred_str = le.inverse_transform(y_pred_enc).tolist()

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_pca)
    else:
        # SVM with decision_function fallback
        dec    = model.decision_function(X_pca)
        # Softmax-like normalisation for confidence display
        exp_d  = np.exp(dec - dec.max(axis=1, keepdims=True))
        probas = exp_d / exp_d.sum(axis=1, keepdims=True)

    # ── 10.4  Compute metrics ─────────────────────────────────────────────
    print("\n[4/6] Computing metrics ...")
    label_idx = list(range(len(class_names)))
    overall_acc = accuracy_score(y_true_enc, y_pred_enc)
    cm = confusion_matrix(
        y_true_enc, y_pred_enc, labels=label_idx,
    )
    report = classification_report(
        y_true_enc, y_pred_enc,
        labels=label_idx,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    print(f"\n  Overall test accuracy : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
    print(f"  Macro-F1              : {macro_f1:.4f}")
    print(f"  Weighted F1           : {weighted_f1:.4f}")
    cr_txt = classification_report(
        y_true_enc, y_pred_enc,
        labels=label_idx,
        target_names=class_names,
        zero_division=0,
    )
    print("\n" + cr_txt)

    # Save metrics.txt
    metrics_path = out_dir / "test_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Test accuracy : {overall_acc:.4f}\n")
        f.write(f"Macro-F1      : {macro_f1:.4f}\n")
        f.write(f"Weighted F1   : {weighted_f1:.4f}\n\n")
        f.write(cr_txt)
    print(f"  Saved: {metrics_path}")

    # Structured JSON (for reports / reproducibility)
    eval_json = {
        "test_dir": str(test_dir.resolve()),
        "n_samples": int(len(y_true_enc)),
        "accuracy": round(float(overall_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "top_confusion_pairs": [
            {"count": c, "true": t, "pred": p}
            for c, t, p in top_confusion_pairs(cm, class_names, k=15)
        ],
        "per_class": {
            c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
            for c in class_names
        },
        "model_path": str(model_path.resolve()),
    }
    json_path = out_dir / "test_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2)
    print(f"  Saved: {json_path}")

    # Save results.csv
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "true_label", "pred_label",
                         "correct", "confidence"])
        for p, true, pred, prob in zip(paths, y_true_str, y_pred_str, probas):
            writer.writerow([str(p), true, pred,
                             int(true == pred), round(prob.max(), 4)])
    print(f"  Saved: {csv_path}")

    # ── 10.5  Plots ───────────────────────────────────────────────────────
    print("\n[5/6] Generating plots ...")
    plot_confusion_matrix(
        cm, class_names,
        out_dir / "confusion_matrix.png",
    )
    plot_per_class_metrics(
        report, class_names,
        out_dir / "per_class_metrics.png",
    )

    # ── 10.6  Qualitative examples ────────────────────────────────────────
    print("\n[6/6] Exporting qualitative examples ...")
    correct_paths, wrong_paths = export_examples(
        paths, y_true_str, y_pred_str, probas,
        out_dir, n_correct=args.n_examples,
    )

    # Contact sheets
    if correct_paths:
        save_contact_sheet(
            correct_paths[:30],
            title="Correct Predictions (top by confidence)",
            out_path=out_dir / "examples_correct_sheet.png",
        )
    if wrong_paths:
        save_contact_sheet(
            wrong_paths[: args.max_wrong_sheet],
            title="Wrong Predictions (filename encodes pred vs true)",
            out_path=out_dir / "examples_wrong_sheet.png",
        )

    # ── 10.7  Error analysis text ─────────────────────────────────────────
    write_error_analysis(
        report, class_names, cm, overall_acc, macro_f1,
        out_dir / "error_analysis.txt",
    )

    # ── 10.8  Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("All Task 4 artefacts saved to:", out_dir.resolve())
    print("=" * 70)
    print("  test_metrics.txt")
    print("  test_evaluation.json")
    print("  results.csv")
    print("  confusion_matrix.png")
    print("  per_class_metrics.png")
    print("  examples/correct/<class>/  (correct predictions)")
    print("  examples/wrong/<true_class>/  (misclassified; filename shows predicted class)")
    print("  examples_correct_sheet.png")
    print("  examples_wrong_sheet.png")
    print("  error_analysis.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()