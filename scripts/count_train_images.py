#!/usr/bin/env python3
"""Count images per class across data/train, data/val, and data/test."""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def count_images_in_dir(directory: Path) -> int:
    """Count files with common image extensions (non-recursive)."""
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def count_images_recursive(directory: Path) -> int:
    """Count image files recursively under directory."""
    n = 0
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            n += 1
    return n


def get_class_names(split_dir: Path) -> set[str]:
    """Return class folder names under a split directory."""
    if not split_dir.is_dir():
        return set()
    return {d.name for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith(".")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count images per class across data/train, data/val, data/test"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Path to data root containing train/ val/ test/ (default: project data/)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Count images recursively in each class folder",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    if not data_root.is_dir():
        print(f"Error: not a directory: {data_root}")
        raise SystemExit(1)

    if not train_dir.is_dir() and not val_dir.is_dir() and not test_dir.is_dir():
        print(f"Error: expected at least one of these directories under {data_root}: train, val, test")
        raise SystemExit(1)

    class_names = sorted(get_class_names(train_dir) | get_class_names(val_dir) | get_class_names(test_dir))
    if not class_names:
        print(f"No class subdirectories found under {data_root}/train, {data_root}/val, or {data_root}/test")
        raise SystemExit(0)

    counter = count_images_recursive if args.recursive else count_images_in_dir
    rows: list[tuple[str, int, int, int, int]] = []
    total_train = 0
    total_val = 0
    total_test = 0

    for class_name in class_names:
        class_train = train_dir / class_name
        class_val = val_dir / class_name
        class_test = test_dir / class_name
        n_train = counter(class_train) if class_train.is_dir() else 0
        n_val = counter(class_val) if class_val.is_dir() else 0
        n_test = counter(class_test) if class_test.is_dir() else 0
        n_total = n_train + n_val + n_test
        rows.append((class_name, n_train, n_val, n_test, n_total))
        total_train += n_train
        total_val += n_val
        total_test += n_test

    w = max(len(name) for name, _, _, _, _ in rows)
    print(f"{'class':{w}}  train  val  test  total")
    print("-" * (w + 29))
    for name, n_train, n_val, n_test, n_total in rows:
        print(f"{name:{w}}  {n_train:5d}  {n_val:3d}  {n_test:4d}  {n_total:5d}")
    print("-" * (w + 29))
    grand_total = total_train + total_val + total_test
    print(f"{'TOTAL':{w}}  {total_train:5d}  {total_val:3d}  {total_test:4d}  {grand_total:5d}")


if __name__ == "__main__":
    main()
