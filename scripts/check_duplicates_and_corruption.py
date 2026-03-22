#!/usr/bin/env python3
"""
Dataset quality checks:
1) Corrupted / unreadable files using PIL.Image.verify()
2) Exact duplicates using MD5 hash (identical bytes)
3) Near-duplicates using perceptual hashing (pHash) with Hamming distance threshold <= 8 bits

Scans data/train, data/val, data/test (default) and writes reports under --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

try:
    import imagehash  # type: ignore
except Exception:  # pragma: no cover
    imagehash = None


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}


def iter_image_files(root_dirs: Iterable[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for root in root_dirs:
        if not root.is_dir():
            continue
        if recursive:
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    files.append(p)
        else:
            # Common dataset layout: data/<split>/<class_name>/*.jpg
            # Scan one level down from the split directory.
            for child in root.iterdir():
                if child.is_dir():
                    for p in child.iterdir():
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                            files.append(p)
                elif child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                    files.append(child)
    return sorted(set(files))


def compute_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute MD5 over file bytes."""
    md5 = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def check_corrupted_files(paths: list[Path]) -> dict[str, str]:
    """
    Verify each file with PIL.Image.verify(). Returns:
      path_str -> error_message
    """
    corrupted: dict[str, str] = {}
    for p in paths:
        try:
            img = Image.open(p)
            # verify() checks file integrity but does not decode full pixels.
            img.verify()
        except Exception as e:  # PIL raises a variety of exceptions
            corrupted[str(p)] = f"{type(e).__name__}: {e}"
    return corrupted


@dataclass(frozen=True)
class DuplicateReports:
    exact_groups_rows: list[dict[str, str]]
    near_pairs_rows: list[dict[str, str]]


def check_duplicates(
    paths_all: list[Path],
    readable_paths: list[Path],
    phash_threshold_bits: int,
    phash_hash_size: int,
) -> DuplicateReports:
    """
    Exact duplicates via MD5 across all files (including corrupted ones),
    near-duplicates via pHash on readable files only.
    """
    # --- Exact duplicates (MD5) across *all* files ---
    md5_to_paths: dict[str, list[str]] = {}
    for p in paths_all:
        md5 = compute_md5(p)
        md5_to_paths.setdefault(md5, []).append(str(p))

    exact_groups_rows: list[dict[str, str]] = []
    group_id = 0
    for md5, group_paths in sorted(md5_to_paths.items(), key=lambda x: (-len(x[1]), x[0])):
        if len(group_paths) < 2:
            continue
        group_id += 1
        for sp in sorted(group_paths):
            exact_groups_rows.append(
                {
                    "exact_group_id": str(group_id),
                    "md5": md5,
                    "path": sp,
                }
            )

    # --- Near duplicates (pHash) on readable files only ---
    if imagehash is None:
        raise RuntimeError(
            "Missing dependency 'imagehash'. Install with: pip install imagehash pillow"
        )

    phash_values: dict[str, object] = {}
    for p in readable_paths:
        # Re-open for actual hashing (verify() does not guarantee decodability)
        img = Image.open(p).convert("RGB")
        ph = imagehash.phash(img, hash_size=phash_hash_size)
        phash_values[str(p)] = ph

    readable_strs = sorted([str(p) for p in readable_paths])

    near_pairs_rows: list[dict[str, str]] = []
    n = len(readable_strs)
    # Naive pairwise comparisons; with ~600 images this is typically fine.
    for i in range(n):
        a = readable_strs[i]
        ha = phash_values[a]
        for j in range(i + 1, n):
            b = readable_strs[j]
            hb = phash_values[b]
            # imagehash supports: distance = hash1 - hash2 (Hamming distance)
            dist = int(ha - hb)
            if dist <= phash_threshold_bits:
                near_pairs_rows.append(
                    {
                        "near_pair_id": str(len(near_pairs_rows) + 1),
                        "path_a": a,
                        "path_b": b,
                        "phash_hamming_distance_bits": str(dist),
                    }
                )

    return DuplicateReports(
        exact_groups_rows=exact_groups_rows,
        near_pairs_rows=near_pairs_rows,
    )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(rows: list[dict[str, str]], out_path: Path, fieldnames: list[str]) -> None:
    ensure_parent_dir(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find exact duplicates (MD5) and near-duplicates (pHash) and detect corrupted images."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Data root containing train/ val/ test (default: project data/)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Subdirectories under data-root to scan (default: train val test)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan inside split folders (uses rglob).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "reports" / "quality_check_duplicates",
        help="Where to write CSV/JSON reports.",
    )
    parser.add_argument(
        "--phash-threshold-bits",
        type=int,
        default=8,
        help="Near-duplicate threshold: keep pairs with Hamming distance <= this bit count.",
    )
    parser.add_argument(
        "--phash-hash-size",
        type=int,
        default=8,
        help="pHash hash size (default 8 => 8x8 => 64-bit hash).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit for quick testing (0 means no limit).",
    )

    args = parser.parse_args()

    scan_dirs = [args.data_root / s for s in args.splits]
    all_paths = iter_image_files(scan_dirs, recursive=args.recursive)
    if args.max_files and args.max_files > 0:
        all_paths = all_paths[: args.max_files]

    if not all_paths:
        print(f"No image files found under {', '.join(str(d) for d in scan_dirs)}", file=sys.stderr)
        raise SystemExit(1)

    corrupted_map = check_corrupted_files(all_paths)
    readable_paths = [p for p in all_paths if str(p) not in corrupted_map]

    try:
        dup_reports = check_duplicates(
            paths_all=all_paths,
            readable_paths=readable_paths,
            phash_threshold_bits=args.phash_threshold_bits,
            phash_hash_size=args.phash_hash_size,
        )
    except RuntimeError as e:
        # Keep output user-friendly for common missing-dependency scenarios.
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)

    ensure_parent_dir(args.output_dir / "dummy.txt")

    # CSV outputs
    corrupted_rows = [
        {"path": p, "error": err} for p, err in sorted(corrupted_map.items(), key=lambda x: x[0])
    ]
    write_csv(
        corrupted_rows,
        args.output_dir / "corrupted_files.csv",
        fieldnames=["path", "error"],
    )

    write_csv(
        dup_reports.exact_groups_rows,
        args.output_dir / "exact_duplicates_md5_groups.csv",
        fieldnames=["exact_group_id", "md5", "path"],
    )

    write_csv(
        dup_reports.near_pairs_rows,
        args.output_dir / "near_duplicates_phash_pairs.csv",
        fieldnames=["near_pair_id", "path_a", "path_b", "phash_hamming_distance_bits"],
    )

    # Summary
    summary = {
        "scan": {
            "data_root": str(args.data_root),
            "splits": args.splits,
            "recursive": args.recursive,
            "num_files_scanned": len(all_paths),
            "num_files_readable": len(readable_paths),
            "num_files_corrupted": len(corrupted_map),
        },
        "exact_duplicates_md5": {
            "num_exact_duplicate_rows": len(dup_reports.exact_groups_rows),
        },
        "near_duplicates_phash": {
            "phash_threshold_bits": args.phash_threshold_bits,
            "phash_hash_size": args.phash_hash_size,
            "num_near_duplicate_pairs": len(dup_reports.near_pairs_rows),
        },
        "notes": {
            "exact_duplicates": "Grouped by identical MD5 bytes (includes corrupted files because MD5 computed for all).",
            "near_duplicates": "pHash computed only for readable images after PIL verify.",
        },
    }

    ensure_parent_dir(args.output_dir / "summary.json")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Quality check complete.")
    print(f"Scanned:  {len(all_paths)} image files")
    print(f"Readable: {len(readable_paths)}")
    print(f"Corrupted:{len(corrupted_map)}")
    print(f"Exact dup rows: {len(dup_reports.exact_groups_rows)}")
    print(f"Near dup pairs: {len(dup_reports.near_pairs_rows)}")
    print(f"Reports written to: {args.output_dir}")


if __name__ == "__main__":
    main()

