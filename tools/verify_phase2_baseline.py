"""Read-only Phase 2 baseline evidence checks (Python 3.10+, standard library).

No images, TensorFlow, training, prediction, pickle loading or network access.
CSV checks and B0 arithmetic are real checks; model-file presence is NOT proof
that a model can run or that it produced a reported score. Missing artifacts
are PENDING, not fabricated results. Output creation is exclusive (no overwrite).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REFERENCE_COMMIT = "8cb91dc12b29035eee70d3c3caf7690c7540399d"
EXPECTED_SPLITS = {
    "train_split.csv": "05851b5173130d45b78f1057ec7f0e78cf827a06",
    "val_split.csv": "54c0018bb7fbb08c59dd26c511928836e5ea46ba",
}
REPORTED = {
    "B0_mean": {"mse": 154431.7, "mae": 295.9, "r2": -0.000},
    "B1_ridge": {"mse": 114853.7, "mae": 249.0, "r2": 0.256},
    "B2_mlp": {"mse": 133378.6, "mae": 270.8, "r2": 0.136},
}


def file_identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    # Git on Windows may check text files out with CRLF. Check the canonical
    # LF blob as well as the exact working-file SHA-256; never rewrite input.
    canonical = raw.replace(b"\r\n", b"\n")
    header = f"blob {len(canonical)}\0".encode("ascii")
    return {
        "file_name": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob_sha1_lf": hashlib.sha1(header + canonical).hexdigest(),
    }


def read_table(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f"{path.name}: expected columns {columns}, got {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: no data rows")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path.name}: malformed row width")
    ids = [row["imageid"] for row in rows]
    if any(not x.strip() or x != x.strip() for x in ids):
        raise ValueError(f"{path.name}: blank or whitespace-padded image ID")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate image IDs")
    if "price" in columns:
        prices(rows)  # Validate numeric values before any arithmetic.
    return rows


def prices(rows: list[dict[str, str]]) -> list[float]:
    values = [float(row["price"]) for row in rows]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Prices or predictions contain NaN or infinity")
    return values


def metric_summary(actual: list[float], predicted: list[float]) -> dict[str, float | None]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Metric inputs must be nonempty and have equal length")
    if not all(math.isfinite(x) for x in actual + predicted):
        raise ValueError("Metric inputs must be finite")
    mean = math.fsum(actual) / len(actual)
    sq_error = math.fsum((a - b) ** 2 for a, b in zip(actual, predicted))
    total = math.fsum((a - mean) ** 2 for a in actual)
    return {
        "mse": sq_error / len(actual),
        "mae": math.fsum(abs(a - b) for a, b in zip(actual, predicted)) / len(actual),
        "r2": 1.0 - sq_error / total if total > 0 else None,
    }


def b0_reference(train_prices: list[float], val_prices: list[float]) -> dict[str, Any]:
    if not train_prices:
        raise ValueError("Empty training prices")
    mean = math.fsum(train_prices) / len(train_prices)
    std = math.sqrt(math.fsum((x - mean) ** 2 for x in train_prices) / len(train_prices))
    return {
        "precision": "Python float / float64-style CSV arithmetic; no float32 input cast",
        "training_mean": mean,
        "training_std_population": std,
        "training_min": min(train_prices),
        "training_max": max(train_prices),
        "validation": metric_summary(val_prices, [mean] * len(val_prices)),
    }


def load_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")
    result = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return result


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def audit(repo: Path, data_root: Path, results_path: Path | None = None,
          submission_path: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_repository_commit": REFERENCE_COMMIT,
        "scope": "CSV arithmetic, supplied log consistency and artifact inventory only",
        "training_executed": False,
        "images_read": False,
        "kaggle_score_verified": False,
        "checks": [], "files_read": [],
    }
    checks = report["checks"]

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    def pending(name: str, detail: str) -> None:
        checks.append({"name": name, "status": "PENDING", "detail": detail})

    def table(path: Path, cols: list[str]) -> list[dict[str, str]]:
        rows = read_table(path, cols)
        report["files_read"].append(file_identity(path))
        check(f"csv_{path.name}", True, f"{len(rows)} rows; exact schema, unique nonblank IDs, finite prices where present")
        return rows

    train_path = repo / "data/splits/train_split.csv"
    val_path = repo / "data/splits/val_split.csv"
    train, val = table(train_path, ["imageid", "price"]), table(val_path, ["imageid", "price"])
    for path in (train_path, val_path):
        found = file_identity(path)["git_blob_sha1_lf"]
        check(f"identity_{path.name}", found == EXPECTED_SPLITS[path.name],
              f"LF-normalised blob {found}; compare with inspected split, never regenerate it")
    train_ids = [r["imageid"] for r in train]
    val_ids = [r["imageid"] for r in val]
    check("split_counts", (len(train), len(val)) == (7000, 1000), f"train={len(train)}, validation={len(val)}")
    check("split_disjoint", not set(train_ids) & set(val_ids), "Training and validation IDs do not overlap")
    tr_y, va_y = prices(train), prices(val)
    check("positive_labels", all(x > 0 for x in tr_y + va_y), "Only supplied labelled partitions checked")
    reference = b0_reference(tr_y, va_y)
    report["B0_recomputed"] = reference
    check("b0_agrees_with_documented_rounding",
          abs(reference["validation"]["mse"] - REPORTED["B0_mean"]["mse"]) <= 0.05,
          "Compare with rounded historical MSE; this does not verify B1/B2")

    source = data_root / "train.csv"
    if source.is_file():
        original = table(source, ["imageid", "price"])
        lookup = {r["imageid"]: float(r["price"]) for r in original}
        combined = train + val
        coverage = set(train_ids + val_ids) == set(lookup)
        check("source_coverage", coverage and len(combined) == len(original), "Existing split covers every supplied training row exactly once")
        matches = all(r["imageid"] in lookup and
                      math.isclose(float(r["price"]), lookup[r["imageid"]], rel_tol=0, abs_tol=1e-7)
                      for r in combined)
        check("source_labels_match", matches, "Split labels compared by image ID with supplied train.csv")
    else:
        pending("source_labels", "train.csv not provided at the resolved data root; B0 still uses committed partitions")

    test_path, sample_path = data_root / "test.csv", data_root / "sample_solution.csv"
    test = table(test_path, ["imageid"]) if test_path.is_file() else None
    if test is None:
        pending("test_ids", "test.csv unavailable; submission ID coverage cannot be checked")
    else:
        test_ids = [r["imageid"] for r in test]
        check("test_count", len(test_ids) == 3000, f"test={len(test_ids)}")
        check("test_isolation", not set(test_ids) & set(train_ids + val_ids), "Test IDs do not appear in the labelled partitions")
        if sample_path.is_file():
            sample = table(sample_path, ["imageid", "price"])
            check("sample_id_order", [r["imageid"] for r in sample] == test_ids, "Template is used for format only, not ground truth")
        else:
            pending("sample_format", "sample_solution.csv unavailable")

    log_path = results_path or repo / "outputs/logs/phase2_results.json"
    if not log_path.is_file():
        pending("phase2_run_log", "Original phase2_results.json not supplied; no B1/B2 metric reproduction claimed")
    else:
        log = load_object(log_path)
        report["files_read"].append(file_identity(log_path))
        expected = {"phase": 2, "seed": 42, "n_train": len(train), "n_val": len(val), "feature_dim": 2048}
        check("log_setup", all(log.get(k) == v for k, v in expected.items()), f"Required setup: {expected}")
        models = ["B0_mean", "B1_ridge", "B2_mlp"]
        valid = all(isinstance(log.get(k), dict) and
                    all(finite_number(log[k].get(m)) for m in ("mse", "mae", "r2")) and
                    log[k]["mse"] >= 0 and log[k]["mae"] >= 0 and log[k]["r2"] <= 1
                    for k in models)
        check("log_metrics", valid, "Finite MSE/MAE/R2 with valid ranges; reported B1/B2 values are not independently reproduced")
        if valid:
            check("log_b0", abs(log["B0_mean"]["mse"] - reference["validation"]["mse"]) <= 0.02,
                  "0.02 MSE tolerance allows the existing float32-input baseline computation")
            winner = "B2_mlp" if log["B2_mlp"]["mse"] < log["B1_ridge"]["mse"] else "B1_ridge"
            check("selection_rule", log.get("selected") == winner, f"Lowest B1/B2 validation MSE selects {winner}; ties use B1")
            report["supplied_log_metrics"] = {k: {m: log[k][m] for m in ("mse", "mae", "r2")} for k in models}
            report["mse_difference_from_historical_record"] = {k: log[k]["mse"] - REPORTED[k]["mse"] for k in models}
        stats = log.get("target_stats", {})
        stats = stats if isinstance(stats, dict) else {}
        expected_stats = {"mean": reference["training_mean"], "std": reference["training_std_population"],
                          "min": reference["training_min"], "max": reference["training_max"]}
        check("log_target_stats", all(finite_number(stats.get(k)) and abs(stats[k] - v) < 0.01 for k, v in expected_stats.items()),
              "Population std and bounds are computed from training labels only")
        ridge, mlp = log.get("B1_ridge", {}), log.get("B2_mlp", {})
        ridge = ridge if isinstance(ridge, dict) else {}
        mlp = mlp if isinstance(mlp, dict) else {}
        check("ridge_alpha", finite_number(ridge.get("alpha")) and 0.01 <= ridge.get("alpha", -1) <= 10000,
              "Selected alpha lies within the baseline candidate range")
        check("mlp_config", mlp.get("hidden") == 256 and mlp.get("dropout") == 0.2 and mlp.get("learning_rate") == 0.001,
              "Match the inspected MLP configuration")
        epochs = mlp.get("epochs_run")
        check("mlp_epochs", isinstance(epochs, int) and not isinstance(epochs, bool) and 1 <= epochs <= 200,
              "Actual epoch count is recorded and within the cap")

    for name in ("x_scaler.joblib", "ridge.joblib", "mlp.keras", "target_stats.json"):
        path = repo / "outputs/models" / name
        if not path.is_file():
            pending(f"artifact_{name}", "Not supplied in this workspace; request the matching original run artifact")
        else:
            check(f"artifact_{name}", path.stat().st_size > 0, "File-presence check only; model is NOT loaded or executed")
            if name == "target_stats.json":
                stats = load_object(path)
                expected = {"mean": reference["training_mean"], "std": reference["training_std_population"],
                            "min": reference["training_min"], "max": reference["training_max"]}
                check("saved_target_stats", all(finite_number(stats.get(k)) and abs(stats[k] - v) < 0.01 for k, v in expected.items()),
                      "Saved training target statistics match the current split")

    submission = submission_path or repo / "outputs/predictions/submission_phase2_ridge.csv"
    if not submission.is_file():
        pending("submission_file", "Baseline submission CSV not provided to this audit; repository presence may be recorded separately")
    else:
        rows = table(submission, ["imageid", "price"])
        check("submission_count", len(rows) == 3000, f"{len(rows)} rows")
        if test is not None:
            check("submission_ids_and_order", [r["imageid"] for r in rows] == [r["imageid"] for r in test],
                  "Exact test ID list and order; no performance is inferred")
        check("submission_training_bounds", all(reference["training_min"] <= v <= reference["training_max"] for v in prices(rows)),
              "Checks this baseline's clipping policy, not a universal rule about true house prices")

    report["historical_values_source"] = f"docs/experiments.md at {REFERENCE_COMMIT}; not newly measured B1/B2 scores"
    counts = Counter(c["status"] for c in checks)
    report["summary"] = {key.lower(): counts[key] for key in ("PASS", "FAIL", "PENDING")}
    report["status"] = "FAILED" if counts["FAIL"] else ("SCOPED_CHECKS_PASS_ARTIFACTS_PENDING" if counts["PENDING"] else "ARTIFACT_CHECKS_PASS_NOT_A_TRAINING_REPRODUCTION")
    return report


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results", type=Path, help="Original phase2_results.json from the matching run")
    parser.add_argument("--submission", type=Path, help="Submission CSV to inspect; default is the historical Ridge CSV")
    parser.add_argument("--out", type=Path, help="NEW JSON file; an existing path is refused")
    parser.add_argument("--require-artifacts", action="store_true", help="Return exit 2 when any item is pending")
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    data = (args.data_root or Path(os.environ.get("CV_DATA_ROOT", repo.parent / "comp-90086-2026/house_dataset"))).resolve()
    try:
        if args.out is not None and args.out.exists():
            raise FileExistsError(f"Refusing to overwrite {args.out}")
        report = audit(repo, data, args.results, args.submission)
        if args.out is not None:
            write_new_json(args.out, report)
        for c in report["checks"]:
            print(f"{c['status']:<7} {c['name']}: {c['detail']}")
        print(json.dumps(report["summary"]))
        print(report["status"])
        print("No images read; no models trained, loaded or executed; no Kaggle score verified.")
        if report["summary"]["fail"]:
            return 1
        return 2 if args.require_artifacts and report["summary"]["pending"] else 0
    except (OSError, ValueError, TypeError, OverflowError, csv.Error) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
