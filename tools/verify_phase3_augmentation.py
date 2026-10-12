"""Read-only Member A Phase 3 (P05) E2/E4 evidence audit, standard library only.

Join committed validation exports by image ID and recompute CSV-decimal metrics.
Original training logs are optional evidence; checkpoint checks are inventory
only. No images, TensorFlow, model loading, training, prediction or network.
This audit cannot establish that a checkpoint produced a particular CSV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, DecimalException, localcontext
from pathlib import Path
from typing import Any

REFERENCE_COMMIT = "4ba5d22751a529ac7f4ec56947ea86d7cb531159"
EXPECTED_SPLITS = {
    "train_split.csv": "05851b5173130d45b78f1057ec7f0e78cf827a06",
    "val_split.csv": "54c0018bb7fbb08c59dd26c511928836e5ea46ba",
}
TAGS = ("E2_finetune", "E4_augment")
HISTORICAL_MSE = {"E2_finetune": 102745.7, "E4_augment": 108784.5}
PRED_COLUMNS = ["imageid", "price_true", "price_pred", "residual", "abs_error"]
PRECISION = (
    "Metrics use the exact decimal values in the exported CSVs, with Decimal "
    "arithmetic (50 significant digits) before JSON float serialization. "
    "Predictions, residuals and absolute errors were independently exported "
    "to one decimal place. Original full-precision predictions are not recovered."
)
ROUNDING_ASSUMPTIONS = (
    "Assume the same exact labels and each original prediction differs from its "
    "CSV value by at most 0.05. With MAE computed from the rounded predictions, "
    "abs(MSE_csv - MSE_original) <= 0.1 * MAE_csv + 0.0025. "
    "MAE differs by at most 0.05 and R2 by at most the MSE bound / label variance. "
    "Historical MSEs displayed to one decimal add a further 0.05 reporting bound. "
    "The original pipeline also casts targets/operations to float32. Original-log "
    "compatibility uses a separately stated 0.0001 price-unit numerical allowance "
    "in addition to the ideal CSV rounding bound; it is a comparison tolerance, "
    "not a verified maximum error for every original floating-point operation. "
    "Compatibility does not recover or verify the original run."
)
INPUT_ERRORS = (OSError, ValueError, TypeError, OverflowError, csv.Error,
                DecimalException, RecursionError)


def file_identity(path: Path, raw: bytes | None = None) -> dict[str, Any]:
    raw = path.read_bytes() if raw is None else raw
    canonical = raw.replace(b"\r\n", b"\n")
    header = f"blob {len(canonical)}\0".encode("ascii")
    return {"path": str(path.resolve()), "file_name": path.name, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1_lf": hashlib.sha1(header + canonical).hexdigest()}


def decimal_number(value: str) -> Decimal:
    number = Decimal(value)
    if not number.is_finite() or not math.isfinite(float(number)):
        raise ValueError("Numeric values must be finite and representable as JSON numbers")
    return number


def parse_table(raw: bytes, columns: list[str]) -> dict[str, dict[str, Decimal]]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != columns:
        raise ValueError(f"Expected columns {columns}; got {reader.fieldnames}")
    rows: dict[str, dict[str, Decimal]] = {}
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed CSV row width")
        imageid = row["imageid"]
        if not imageid.strip() or imageid != imageid.strip():
            raise ValueError("Blank or whitespace-padded image ID")
        if imageid in rows:
            raise ValueError(f"Duplicate image ID: {imageid}")
        rows[imageid] = {key: decimal_number(row[key]) for key in columns[1:]}
    if not rows:
        raise ValueError("CSV contains no data rows")
    return rows


def metric_summary(actual: list[Decimal], predicted: list[Decimal]) -> dict[str, Any]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Metric inputs must be nonempty and have equal length")
    if not all(x.is_finite() for x in actual + predicted):
        raise ValueError("Metric inputs must be finite")
    with localcontext() as ctx:
        ctx.prec = 50
        n = Decimal(len(actual))
        mean = sum(actual) / n
        mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n
        mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / n
        variance = sum((a - mean) ** 2 for a in actual) / n
        r2 = 1 - mse / variance if variance > 0 else None
    values = {"mse": mse, "mae": mae, "r2": r2}
    if not all(v is None or math.isfinite(float(v)) for v in values.values()):
        raise ValueError("Computed metrics exceed finite JSON numeric range")
    return {"metrics": {k: float(v) if v is not None else None for k, v in values.items()},
            "metrics_decimal": {k: str(v) if v is not None else None for k, v in values.items()},
            "label_variance": float(variance)}


def load_object(raw: bytes) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"Duplicate JSON key: {key}")
            obj[key] = value
        return obj

    result = json.loads(raw.decode("utf-8-sig"), parse_constant=reject,
                        object_pairs_hook=unique_pairs)

    def finite_tree(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite JSON number")
        if isinstance(value, dict):
            for child in value.values():
                finite_tree(child)
        elif isinstance(value, list):
            for child in value:
                finite_tree(child)

    finite_tree(result)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def on_tenth_grid(value: Decimal) -> bool:
    digits, exponent = value.as_tuple().digits, value.as_tuple().exponent
    places_beyond_tenths = max(0, -exponent - 1)
    return not places_beyond_tenths or not any(digits[-places_beyond_tenths:])


def validate_log(log: dict[str, Any], tag: str) -> None:
    """Validate the schema actually emitted by src/exp_finetune.py."""
    expected = {"experiment": tag, "seed": 42, "img_size": 224,
                "augmentation": tag == "E4_augment",
                "unfrozen": "layer4 (conv5) + regression head",
                "learning_rate": 1e-4, "batch_size": 32,
                "epochs_cap": 15, "patience": 4}
    for key, value in expected.items():
        found = log.get(key)
        if type(found) is not type(value) or found != value:
            raise ValueError(f"{key}: expected {value!r}, got {found!r}")
    created = log.get("created_utc")
    if not isinstance(created, str):
        raise ValueError("created_utc must be an ISO timestamp with UTC offset")
    timestamp = datetime.fromisoformat(created.replace("Z", "+00:00"))
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("created_utc must use UTC")
    for key in ("epochs_cap", "epochs_run", "best_epoch", "trainable_params", "total_params"):
        if type(log.get(key)) is not int or log[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(log.get("patience")) is not int or log["patience"] < 0:
        raise ValueError("patience must be a nonnegative integer")
    if log["trainable_params"] > log["total_params"]:
        raise ValueError("trainable_params cannot exceed total_params")
    if log["epochs_run"] > log["epochs_cap"]:
        raise ValueError("epochs_run cannot exceed epochs_cap")
    history = log.get("history")
    if not isinstance(history, dict):
        raise ValueError("history must contain loss and val_loss arrays")
    for key in ("loss", "val_loss"):
        values = history.get(key)
        if not isinstance(values, list) or len(values) != log["epochs_run"]:
            raise ValueError(f"history.{key} length must equal epochs_run")
        if not all(finite_number(x) and x >= 0 for x in values):
            raise ValueError(f"history.{key} values must be finite and nonnegative")
    best_epoch = min(range(log["epochs_run"]), key=history["val_loss"].__getitem__) + 1
    if log["best_epoch"] != best_epoch:
        raise ValueError("best_epoch must be the first minimum of history.val_loss (1-based)")
    val = log.get("val")
    if not isinstance(val, dict) or not all(finite_number(val.get(k)) for k in ("mse", "mae", "r2")):
        raise ValueError("val must contain finite mse, mae and r2")
    if val["mse"] < 0 or val["mae"] < 0 or val["r2"] > 1:
        raise ValueError("val metric ranges are invalid")
    if val["mae"] > math.sqrt(val["mse"]) + 1e-9:
        raise ValueError("val.mae cannot exceed sqrt(val.mse)")


def input_paths(repo: Path, e2_log: Path | None, e4_log: Path | None,
                e2_model: Path | None, e4_model: Path | None) -> dict[str, Path]:
    paths = {name: repo / "data/splits" / name for name in EXPECTED_SPLITS}
    for tag, log, model in zip(TAGS, (e2_log, e4_log), (e2_model, e4_model)):
        paths[f"csv_{tag}"] = repo / "outputs/predictions" / f"val_predictions_{tag}.csv"
        paths[f"log_{tag}"] = log if log is not None else repo / "outputs/logs" / f"phase3_{tag}.json"
        paths[f"model_{tag}"] = model if model is not None else repo / "outputs/models" / f"finetuned_{tag}.keras"
    return paths


def audit(repo: Path, e2_log: Path | None = None, e4_log: Path | None = None,
          e2_model: Path | None = None, e4_model: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_repository_commit": REFERENCE_COMMIT,
        "scope": "Member A P05: E2 no-augmentation versus E4 moderate-augmentation CSV evidence and original artifact inventory",
        "training_executed": False, "model_loading_executed": False,
        "images_read": False, "kaggle_score_verified": False,
        "precision_caveat": PRECISION, "rounding_assumptions": ROUNDING_ASSUMPTIONS,
        "historical_values_source": f"docs/experiments.md at {REFERENCE_COMMIT}; historical values, not new training results",
        "checks": [], "files_read": [], "runs": {}, "comparison": None,
    }
    paths = input_paths(repo, e2_log, e4_log, e2_model, e4_model)

    def check(name: str, passed: bool, detail: str) -> None:
        report["checks"].append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    def pending(name: str, detail: str) -> None:
        report["checks"].append({"name": name, "status": "PENDING", "detail": detail})

    def read(path: Path) -> bytes:
        raw = path.read_bytes()
        identity = file_identity(path, raw)
        identity["repository_relative_path"] = path.resolve().relative_to(repo.resolve()).as_posix() if path.resolve().is_relative_to(repo.resolve()) else None
        report["files_read"].append(identity)
        return raw

    def table(key: str, columns: list[str]) -> dict[str, dict[str, Decimal]] | None:
        try:
            rows = parse_table(read(paths[key]), columns)
            check(key, True, f"{len(rows)} rows; exact schema, unique nonblank IDs and finite numeric values")
            return rows
        except INPUT_ERRORS as exc:
            check(key, False, f"{paths[key]}: {exc}")
            return None

    train = table("train_split.csv", ["imageid", "price"])
    val = table("val_split.csv", ["imageid", "price"])
    for name in EXPECTED_SPLITS:
        found = next((x for x in report["files_read"] if x["path"] == str(paths[name].resolve())), None)
        if found:
            check(f"identity_{name}", found["git_blob_sha1_lf"] == EXPECTED_SPLITS[name],
                  f"LF-normalised Git blob {found['git_blob_sha1_lf']}; expected {EXPECTED_SPLITS[name]}")
    if train is not None and val is not None:
        check("split_counts", (len(train), len(val)) == (7000, 1000), f"train={len(train)}, validation={len(val)}")
        overlap = set(train) & set(val)
        check("split_disjoint", not overlap, f"Training/validation overlap: {len(overlap)} IDs")
        check("positive_labels", all(x["price"] > 0 for x in list(train.values()) + list(val.values())),
              "All committed training and validation labels must be positive")
    tables = {tag: table(f"csv_{tag}", PRED_COLUMNS) for tag in TAGS}
    for tag, rows in tables.items():
        report["runs"][tag] = {"csv_path": str(paths[f"csv_{tag}"].resolve()),
                               "metrics": None, "row_count": len(rows) if rows is not None else None,
                               "precision": PRECISION, "supplied_log": None}
        if rows is None:
            continue
        if val is not None:
            missing, extra = set(val) - set(rows), set(rows) - set(val)
            check(f"{tag}_coverage", not missing and not extra and len(rows) == 1000,
                  f"Join by image ID; missing={len(missing)}, extra={len(extra)}, rows={len(rows)}")
            mismatch = [i for i in set(rows) & set(val) if rows[i]["price_true"] != val[i]["price"]]
            check(f"{tag}_source_labels", not mismatch, f"Exact decimal source-label mismatches by ID: {len(mismatch)}")
        inconsistent = [i for i, r in rows.items()
                        if r["abs_error"] < 0
                        or r["abs_error"] != abs(r["residual"])
                        or not all(on_tenth_grid(r[k]) for k in ("price_pred", "residual", "abs_error"))
                        or abs(r["abs_error"] - abs(r["price_true"] - r["price_pred"])) > Decimal("0.100000001")
                        or abs(r["residual"] - (r["price_pred"] - r["price_true"])) > Decimal("0.100000001")]
        check(f"{tag}_rounded_errors", not inconsistent,
              f"{len(inconsistent)} inconsistent rows; prediction/residual/abs_error must be on the 0.1 grid, abs_error=abs(residual); separately rounded residual versus prediction allows 0.1 plus 1e-9 arithmetic tolerance")
        if train is not None:
            lo, hi = min(r["price"] for r in train.values()), max(r["price"] for r in train.values())
            check(f"{tag}_clipping", all(lo <= r["price_pred"] <= hi for r in rows.values()),
                  f"Exported predictions must be within training-only price range [{lo}, {hi}]")

    csv_valid = not any(c["status"] == "FAIL" for c in report["checks"])
    if csv_valid:
        try:
            with localcontext() as ctx:
                ctx.prec = 50
                ids = sorted(val)
                actual = [val[i]["price"] for i in ids]
                for tag in TAGS:
                    result = metric_summary(actual, [tables[tag][i]["price_pred"] for i in ids])
                    run = report["runs"][tag]
                    run.update(result)
                    mse, mae = Decimal(result["metrics_decimal"]["mse"]), Decimal(result["metrics_decimal"]["mae"])
                    bound = Decimal("0.1") * mae + Decimal("0.0025")
                    delta = mse - Decimal(str(HISTORICAL_MSE[tag]))
                    compatible = abs(delta) <= bound + Decimal("0.05")
                    run["rounding"] = {"prediction_max_error": 0.05, "mse_error_bound": float(bound),
                                       "historical_reporting_max_error": 0.05,
                                       "historical_combined_mse_bound": float(bound + Decimal("0.05")),
                                       "log_price_numerical_allowance": 0.0001,
                                       "log_mse_numerical_allowance": float(Decimal("0.0002") * mae + Decimal("0.00001001"))}
                    run["historical"] = {"mse": HISTORICAL_MSE[tag],
                                         "difference_csv_minus_historical_mse": float(delta),
                                         "compatible_with_rounding": compatible}
                    check(f"{tag}_historical_rounding", compatible,
                          f"CSV minus historical MSE={delta}; combined bound={bound + Decimal('0.05')}; compatibility only")
                a, b = (report["runs"][tag]["metrics_decimal"] for tag in TAGS)
                mse_delta = Decimal(b["mse"]) - Decimal(a["mse"])
                e2_error = [abs(actual[j] - tables[TAGS[0]][i]["price_pred"]) for j, i in enumerate(ids)]
                e4_error = [abs(actual[j] - tables[TAGS[1]][i]["price_pred"]) for j, i in enumerate(ids)]
                report["comparison"] = {
                    "mse_delta_e4_minus_e2": float(mse_delta),
                    "mse_change_percent": float(100 * mse_delta / Decimal(a["mse"])) if Decimal(a["mse"]) else None,
                    "mae_delta_e4_minus_e2": float(Decimal(b["mae"]) - Decimal(a["mae"])),
                    "e4_better_count": sum(b < a for a, b in zip(e2_error, e4_error)),
                    "e4_worse_count": sum(b > a for a, b in zip(e2_error, e4_error)),
                    "equal_count": sum(b == a for a, b in zip(e2_error, e4_error)),
                    "basis": "Absolute errors recomputed from rounded CSV predictions joined by image ID; positive MSE delta means E4 is worse",
                }
        except INPUT_ERRORS as exc:
            csv_valid = False
            report["comparison"] = None
            for run in report["runs"].values():
                run["metrics"] = None
            check("csv_metric_computation", False, str(exc))
    report["csv_analysis_status"] = "PASS" if csv_valid else "FAIL"

    logs = {}
    for tag in TAGS:
        path = paths[f"log_{tag}"]
        try:
            if not path.exists():
                pending(f"{tag}_original_log", f"Original run log missing: {path}; no training history reconstructed")
            else:
                log = load_object(read(path))
                validate_log(log, tag)
                logs[tag] = log
                report["runs"][tag]["supplied_log"] = {k: v for k, v in log.items() if k != "history"}
                report["runs"][tag]["supplied_log"]["history"] = log["history"]
                check(f"{tag}_log_schema", True, "Original schema, fixed settings, finite metrics/history, epoch lengths and first argmin best epoch checked")
                if csv_valid:
                    run = report["runs"][tag]
                    mse_limit = run["rounding"]["mse_error_bound"] + run["rounding"]["log_mse_numerical_allowance"]
                    limits = {"mse": mse_limit, "mae": 0.0501,
                              "r2": mse_limit / run["label_variance"] if run["label_variance"] else 0.0}
                    differences = {k: abs(log["val"][k] - run["metrics"][k]) if run["metrics"][k] is not None else None for k in limits}
                    check(f"{tag}_log_csv_metrics", all(differences[k] is not None and differences[k] <= limits[k] + 1e-8 for k in limits),
                          f"Original-log/CSV metric differences={differences}; ideal rounding plus declared numerical allowances={limits}; no original predictions recovered")
                else:
                    pending(f"{tag}_log_csv_metrics", "CSV metric compatibility unavailable because required input validation failed")
        except INPUT_ERRORS as exc:
            check(f"{tag}_original_log", False, f"{path}: {exc}")
        model_path = paths[f"model_{tag}"]
        inventory = {"path": str(model_path.resolve()), "present": False,
                     "loaded": False, "checkpoint_provenance_verified": False}
        report["runs"][tag]["model_inventory"] = inventory
        try:
            if not model_path.exists():
                pending(f"{tag}_checkpoint_inventory", f"Original checkpoint missing: {model_path}; inventory only")
            else:
                inventory.update({"present": model_path.is_file(), "bytes": model_path.stat().st_size})
                check(f"{tag}_checkpoint_inventory", inventory["present"] and inventory["bytes"] > 0,
                      "Nonempty regular-file presence only; no model loading, prediction or checkpoint-to-CSV provenance verification")
        except INPUT_ERRORS as exc:
            check(f"{tag}_checkpoint_inventory", False, f"{model_path}: {exc}")
    if len(logs) == 2:
        fields = ("seed", "img_size", "unfrozen", "learning_rate", "batch_size", "epochs_cap", "patience", "trainable_params", "total_params")
        mismatches = [k for k in fields if logs[TAGS[0]][k] != logs[TAGS[1]][k]]
        check("matched_run_conditions", not mismatches,
              f"Matched fixed conditions, epoch cap, patience and parameter counts; differing fields: {mismatches}")
    else:
        pending("matched_run_conditions", "Both valid original logs are required to compare the recorded run conditions")
    counts = Counter(c["status"] for c in report["checks"])
    status = "FAILED" if counts["FAIL"] else ("CSV_CHECKS_PASS_ORIGINAL_EVIDENCE_PENDING" if counts["PENDING"] else "AVAILABLE_EVIDENCE_CHECKS_PASS_NOT_A_TRAINING_REPRODUCTION")
    report["status"] = status
    report["summary"] = {key.lower(): counts[key] for key in ("PASS", "FAIL", "PENDING")}
    report["summary"]["status"] = status
    return report


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    for flag in ("e2-log", "e4-log", "e2-model", "e4-model"):
        parser.add_argument(f"--{flag}", type=Path, help="Optional path to the matching original artifact")
    parser.add_argument("--out", type=Path, required=True, help="NEW audit JSON file; existing paths and source aliases are refused")
    parser.add_argument("--require-artifacts", action="store_true", help="Exit 2 when original evidence remains pending")
    args = parser.parse_args(argv)
    try:
        repo = args.repo_root.resolve()
        paths = input_paths(repo, args.e2_log, args.e4_log, args.e2_model, args.e4_model)
        if args.out.resolve() in {p.resolve() for p in paths.values()}:
            raise ValueError("Output must not alias any audit input, including a missing original artifact")
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {args.out}")
        report = audit(repo, args.e2_log, args.e4_log, args.e2_model, args.e4_model)
        write_new_json(args.out, report)
        for c in report["checks"]:
            print(f"{c['status']:<7} {c['name']}: {c['detail']}")
        print(json.dumps(report["summary"]))
        if report["summary"]["fail"]:
            print("Evidence checks failed; inspect the FAIL entries. A valid CSV comparison, if present, is separately labelled.")
        elif report["summary"]["pending"]:
            print("CSV checks passed; original evidence pending.")
        else:
            print("Available evidence checks passed; no training reproduction or checkpoint provenance verified.")
        print("No images read; no models trained, loaded or executed; no Kaggle score verified.")
        return 1 if report["summary"]["fail"] else (2 if args.require_artifacts and report["summary"]["pending"] else 0)
    except INPUT_ERRORS as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
