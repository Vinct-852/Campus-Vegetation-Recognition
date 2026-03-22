#!/usr/bin/env python3
"""
Split each class folder into train (70%), validation (15%), and test (15%).

Assumes all images currently live under data/train/<class_name>/.
After running, images are moved to:
  data/train/<class_name>/  (70%)
  data/val/<class_name>/    (15%)
  data/test/<class_name>/   (15% — remainder after floor splits)

Uses a fixed random seed for reproducibility. Image extensions: same as count_train_images.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from random import Random

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_counts(n: int) -> tuple[int, int, int]:
    """70% / 15% / remainder to test (handles rounding)."""
    if n == 0:
        return 0, 0, 0
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    # Ensure at least one sample in train when n >= 1
    if n_train == 0 and n > 0:
        n_train = 1
        n_test = max(0, n_test - 1)
    return n_train, n_val, n_test


def dir_has_images(d: Path) -> bool:
    if not d.is_dir():
        return False
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in d.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split data/train/<class>/ into train 70%, val 15%, test 15%"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root data folder containing train/ (default: project data/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves only; do not move files",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move (originals stay in train; use for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow non-empty val/test class folders (still errors if a target file exists)",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root
    train_root = data_root / "train"
    val_root = data_root / "val"
    test_root = data_root / "test"

    if not train_root.is_dir():
        print(f"Error: missing train directory: {train_root}", file=sys.stderr)
        sys.exit(1)

    class_dirs = sorted(d for d in train_root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not class_dirs:
        print(f"No class subdirectories under {train_root}", file=sys.stderr)
        sys.exit(1)

    # Preflight: val/test must be empty per class unless --force
    if not args.force:
        for d in class_dirs:
            name = d.name
            for split_root, label in ((val_root, "val"), (test_root, "test")):
                target = split_root / name
                if dir_has_images(target):
                    print(
                        f"Error: {target} already contains images. "
                        f"Empty it or use --force.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    rng = Random(args.seed)
    summary: list[tuple[str, int, int, int, int]] = []

    for class_dir in class_dirs:
        name = class_dir.name
        files = list_images(class_dir)
        n = len(files)
        if n == 0:
            print(f"Warning: no images in {class_dir}", file=sys.stderr)
            summary.append((name, 0, 0, 0, 0))
            continue

        rng.shuffle(files)
        n_train, n_val, n_test = split_counts(n)
        part_train = files[:n_train]
        part_val = files[n_train : n_train + n_val]
        part_test = files[n_train + n_val :]

        val_class = val_root / name
        test_class = test_root / name
        if not args.dry_run:
            val_class.mkdir(parents=True, exist_ok=True)
            test_class.mkdir(parents=True, exist_ok=True)

        op = shutil.copy2 if args.copy else shutil.move

        for p in part_val:
            dest = val_class / p.name
            if dest.exists() and not args.dry_run:
                print(f"Error: destination exists: {dest}", file=sys.stderr)
                sys.exit(1)
            if args.dry_run:
                print(f"DRY  {p} -> {dest}")
            else:
                op(p, dest)

        for p in part_test:
            dest = test_class / p.name
            if dest.exists() and not args.dry_run:
                print(f"Error: destination exists: {dest}", file=sys.stderr)
                sys.exit(1)
            if args.dry_run:
                print(f"DRY  {p} -> {dest}")
            else:
                op(p, dest)

        summary.append((name, n, n_train, n_val, n_test))

    # Table
    w = max(len(t[0]) for t in summary)
    print()
    print(f"{'class':{w}}  total  train  val  test")
    print("-" * (w + 28))
    tot = [0, 0, 0, 0]
    for name, n, nt, nv, ns in summary:
        print(f"{name:{w}}  {n:5d}  {nt:5d}  {nv:3d}  {ns:4d}")
        tot[0] += n
        tot[1] += nt
        tot[2] += nv
        tot[3] += ns
    print("-" * (w + 28))
    print(f"{'TOTAL':{w}}  {tot[0]:5d}  {tot[1]:5d}  {tot[2]:3d}  {tot[3]:4d}")

    if args.dry_run:
        print("\nDry run only; no files were moved.")
    elif args.copy:
        print("\nCopied val/test images; train folder still has 100% (remove duplicates manually if needed).")
    else:
        print(f"\nDone. Splits are under {train_root}, {val_root}, {test_root}")


if __name__ == "__main__":
    main()
