#!/usr/bin/env python3
"""Read-only Phase 1 CSV audit. Does not open images or rebuild data splits.

Run from the repository root with Python 3.10 or later:
    python tools/verify_phase1_metadata.py --data-root PATH --repo-root .
Optionally use --out phase1_audit_local.json to save a NEW result file.
No TensorFlow, pandas, or other third-party packages are required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

REFERENCE_COMMIT = "9a7ed84f3681d9e13d4d48a43ebdddbd5e94d591"
REFERENCE_BLOBS = {
    "data/splits/train_split.csv": "05851b5173130d45b78f1057ec7f0e78cf827a06",
    "data/splits/val_split.csv": "54c0018bb7fbb08c59dd26c511928836e5ea46ba",
}


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def read_csv(path: Path, expected_columns: list[str]) -> dict:
    """Load a small metadata CSV and report its exact source-byte identity."""
    raw = path.read_bytes()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if columns != expected_columns:
            raise ValueError(f"{path.name}: columns {columns!r}, expected {expected_columns!r}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name}: malformed CSV row")
    return {
        "columns": columns, "rows": rows, "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob_sha1": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
        "lf_git_blob_sha1": git_blob_sha1(raw.replace(b"\r\n", b"\n")),
    }


def parse_prices(rows: list[dict]) -> list[Decimal]:
    """Reject blank, non-numeric, infinite, and non-positive prices."""
    values = []
    for index, row in enumerate(rows, start=2):
        try:
            value = Decimal(row["price"])
        except (InvalidOperation, KeyError) as exc:
            raise ValueError(f"Invalid price at CSV line {index}") from exc
        if not value.is_finite() or value <= 0:
            raise ValueError(f"Non-finite or non-positive price at CSV line {index}")
        values.append(value)
    return values


def price_stats(values: list[Decimal]) -> dict:
    if not values:
        raise ValueError("No prices available")
    numbers = [float(value) for value in values]
    return {
        "n": len(numbers), "min": min(numbers), "mean": statistics.mean(numbers),
        "median": statistics.median(numbers), "max": max(numbers),
        "sample_std_ddof1": statistics.stdev(numbers) if len(numbers) > 1 else None,
        "population_std_ddof0": statistics.pstdev(numbers),
    }


def audit(data_root: Path, repo_root: Path) -> dict:
    files = {
        "train.csv": read_csv(data_root / "train.csv", ["imageid", "price"]),
        "test.csv": read_csv(data_root / "test.csv", ["imageid"]),
        "sample_solution.csv": read_csv(data_root / "sample_solution.csv", ["imageid", "price"]),
        "data/splits/train_split.csv": read_csv(repo_root / "data/splits/train_split.csv", ["imageid", "price"]),
        "data/splits/val_split.csv": read_csv(repo_root / "data/splits/val_split.csv", ["imageid", "price"]),
    }
    checks = []

    def check(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "details": details})

    expected_counts = {"train.csv": 8000, "test.csv": 3000, "sample_solution.csv": 3000,
                       "data/splits/train_split.csv": 7000, "data/splits/val_split.csv": 1000}
    ids, values = {}, {}
    for name, item in files.items():
        rows = item["rows"]
        ids[name] = [row["imageid"] for row in rows]
        check(f"{name}: row count", len(rows) == expected_counts[name], f"{len(rows)} rows")
        valid_ids = all(re.fullmatch(r"[1-9][0-9]*\.jpg", value or "") for value in ids[name])
        check(f"{name}: image ID format", valid_ids, "IDs must be non-blank numeric .jpg filenames; images are not opened")
        duplicates = len(ids[name]) - len(set(ids[name]))
        check(f"{name}: unique image IDs", duplicates == 0, f"{duplicates} duplicate IDs")
        if "price" in item["columns"]:
            values[name] = parse_prices(rows)
            check(f"{name}: valid prices", True, "All prices are finite, numeric and positive")
    full, test = set(ids["train.csv"]), set(ids["test.csv"])
    train, val = set(ids["data/splits/train_split.csv"]), set(ids["data/splits/val_split.csv"])
    check("Training-source vs test IDs", not (full & test), f"{len(full & test)} shared IDs")
    check("Train-partition vs validation IDs", not (train & val), f"{len(train & val)} shared IDs")
    check("Partition coverage", train | val == full, f"{len(train | val)} unique IDs in union; {len(full)} source IDs")
    check("Sample vs test ID set", set(ids["sample_solution.csv"]) == test, "Exact ID-set comparison")
    check("Sample vs test row order", ids["sample_solution.csv"] == ids["test.csv"], "Exact ordered-list comparison")
    truth = dict(zip(ids["train.csv"], values["train.csv"]))
    for name in REFERENCE_BLOBS:
        mismatches = sum(truth.get(i) != p for i, p in zip(ids[name], values[name]))
        check(f"{name}: label agreement", mismatches == 0, f"{mismatches} labels differ from uploaded train.csv (Decimal comparison)")
        observed = files[name]["lf_git_blob_sha1"]
        check(f"{name}: repository snapshot identity", observed == REFERENCE_BLOBS[name],
              f"LF-normalized Git blob SHA-1 {observed}; reference commit {REFERENCE_COMMIT}; source-byte hashes are also retained")
    unique_sample = sorted(set(values["sample_solution.csv"]))
    check("Sample placeholder values", unique_sample == [Decimal("500")],
          f"{len(unique_sample)} unique value(s); this reference sample uses 500, NOT test labels")
    train_stats = price_stats(values["data/splits/train_split.csv"])
    val_stats = price_stats(values["data/splits/val_split.csv"])
    meta_path = repo_root / "data/splits/split_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_settings = {"seed": 42, "val_fraction": 0.125, "n_price_bins": 10,
                             "stratified_by": "price decile", "n_train": 7000, "n_val": 1000}
        setting_ok = all(meta.get(key) == value for key, value in expected_settings.items())
        check("Split metadata settings", setting_ok, "seed 42; validation fraction 0.125; 10 price bins; 7000/1000")
        for label, stats in [("train", train_stats), ("val", val_stats)]:
            expected = {key: round(stats[key], 2) for key in ["mean", "min", "median", "max"]}
            expected["std"] = round(stats["sample_std_ddof1"], 2)
            meta_stats = meta.get(f"{label}_price_stats", {})
            ok = all(abs(float(meta_stats.get(key, float("inf"))) - value) < 1e-8 for key, value in expected.items())
            check(f"{label}: metadata statistics", ok, "Mean, sample std, min, median and max agree at 2 decimal places")
    else:
        checks.append({"check": "Split metadata file", "status": "NOT_CHECKED", "details": "split_meta.json not provided"})
    audit_files = {name: {key: value for key, value in item.items() if key != "rows"} | {"row_count": len(item["rows"])}
                   for name, item in files.items()}
    return {
        "audit_date": datetime.now(timezone.utc).date().isoformat(),
        "scope": "CSV metadata and existing split files only; no images opened, no model run, no source files modified",
        "repository": "mingyang-sun1/computer-vision-final-assignment",
        "reference_commit": REFERENCE_COMMIT,
        "all_executed_checks_passed": all(c["status"] != "FAIL" for c in checks),
        "executed_checks": sum(c["status"] in ("PASS", "FAIL") for c in checks),
        "checks": checks, "files": audit_files,
        "price_stats_usd_1000": {"full_training_source": price_stats(values["train.csv"]),
                                 "training_partition": train_stats, "validation_partition": val_stats},
        "sample_solution_note": "3000 constant placeholder predictions of 500; not ground-truth labels and not the team's model predictions",
        "not_verified": ["Image-file presence, decoding, resolution and image-content duplicates", "Local machine paths and training environment", "Checkpoint availability, original training logs and Kaggle submission status", "Team reviewer sign-off"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CV_DATA_ROOT"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, help="Write a NEW JSON result file; existing files are never overwritten")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("Provide --data-root or set CV_DATA_ROOT")
    try:
        result = audit(Path(args.data_root), args.repo_root)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if result["all_executed_checks_passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(f"Phase 1 audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
