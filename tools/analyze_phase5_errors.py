"""Analyse the existing Phase 4 E5 validation export without loading a model.

Python 3.10+; standard library only. Inputs are pinned and joined by imageid.
The command writes a NEW evidence folder and never changes its source CSVs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from decimal import Decimal, DecimalException, localcontext
from pathlib import Path
from typing import Any, Sequence

# Load the adjacent standard-library checker without depending on the cwd or
# modifying sys.path. Its CSV, identity and ensemble rules remain authoritative.
_spec = importlib.util.spec_from_file_location(
    "phase4_input_primitives", Path(__file__).with_name("verify_phase4_selection.py")
)
assert _spec is not None and _spec.loader is not None
_phase4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase4)
EvidenceError = _phase4.EvidenceError
blob_sha = _phase4.blob_sha
TRAIN, VAL, PRED, TAGS = _phase4.TRAIN, _phase4.VAL, _phase4.PRED, _phase4.TAGS

REFERENCE_COMMIT = "ae74826454627addbeabdec5905595a9599fc105"
ENSEMBLE_PATH = "docs/evidence/phase4_run_20261016T124243_097811Z/validation_ensemble.csv"
PINNED_BLOBS = {
    path: _phase4.PINNED_BLOBS[path]
    for path in (TRAIN, VAL, *(PRED + f"val_predictions_{tag}.csv" for tag in TAGS))
}
PINNED_BLOBS[ENSEMBLE_PATH] = "0798660701be8503df445ca51b2bc629a8bea904"
MODEL_TAG = "E5_mean_export_1dp"
METRIC_FIELDS = [
    "count", "mean_true", "mean_pred", "mse", "rmse", "mae", "bias", "sse", "sse_share",
    "median_abs_error", "p90_abs_error", "overprediction_fraction",
    "underprediction_fraction", "exact_prediction_fraction", "r2",
]
PER_IMAGE_FIELDS = [
    "imageid", "price_true", "price_pred", "residual", "abs_error", "squared_error",
    "relative_error", "abs_relative_error", "price_bucket",
]
BUCKET_FIELDS = ["bucket", "price_lo", "price_hi", "upper_inclusive", *METRIC_FIELDS]
EXAMPLE_FIELDS = ["category", "rank", *PER_IMAGE_FIELDS]


def _float(value: Decimal, context: str = "result") -> float:
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{context} cannot be represented as a finite JSON number")
    return result


def _validate_vectors(labels: Sequence[Decimal], predictions: Sequence[Decimal]) -> None:
    if not labels or len(labels) != len(predictions):
        raise EvidenceError("Nonempty, equally sized labels and predictions are required")
    if any(not value.is_finite() for value in (*labels, *predictions)):
        raise EvidenceError("All labels and predictions must be finite")


def linear_quantile(values: Sequence[Decimal], q: Decimal | float | int) -> Decimal:
    """Linear interpolation at position (n - 1) * q, including both endpoints."""
    q = Decimal(str(q))
    if not values or not q.is_finite() or not Decimal(0) <= q <= Decimal(1):
        raise EvidenceError("A quantile needs nonempty values and 0 <= q <= 1")
    if any(not value.is_finite() for value in values):
        raise EvidenceError("Quantiles require finite values")
    ordered = sorted(values)
    with localcontext() as ctx:
        ctx.prec = 50
        position = Decimal(len(ordered) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def metrics(labels: Sequence[Decimal], predictions: Sequence[Decimal],
            total_sse: Decimal | None = None) -> dict[str, Any]:
    """Unrounded descriptive statistics; residual means predicted minus true."""
    _validate_vectors(labels, predictions)
    with localcontext() as ctx:
        ctx.prec = 50
        n = Decimal(len(labels))
        mean_true, mean_pred = sum(labels) / n, sum(predictions) / n
        residuals = [p - y for y, p in zip(labels, predictions)]
        absolute = [abs(e) for e in residuals]
        sse = sum((e * e for e in residuals), Decimal(0))
        mse = sse / n
        denominator = sse if total_sse is None else total_sse
        if not denominator.is_finite() or denominator < 0:
            raise EvidenceError("Total SSE must be finite and nonnegative")
        variance_sum = sum(((y - mean_true) ** 2 for y in labels), Decimal(0))
        return {
            "count": len(labels), "mean_true": _float(mean_true), "mean_pred": _float(mean_pred),
            "mse": _float(mse), "rmse": _float(mse.sqrt()),
            "mae": _float(sum(absolute) / n), "bias": _float(sum(residuals) / n),
            "sse": _float(sse), "sse_share": _float(sse / denominator) if denominator else 0.0,
            "median_abs_error": _float(linear_quantile(absolute, Decimal("0.5"))),
            "p90_abs_error": _float(linear_quantile(absolute, Decimal("0.9"))),
            "overprediction_fraction": _float(Decimal(sum(e > 0 for e in residuals)) / n),
            "underprediction_fraction": _float(Decimal(sum(e < 0 for e in residuals)) / n),
            "exact_prediction_fraction": _float(Decimal(sum(e == 0 for e in residuals)) / n),
            "r2": _float(1 - sse / variance_sum) if variance_sum else None,
        }


def _partitions(labels: Sequence[Decimal], n: int) -> tuple[list[Decimal], list[list[int]]]:
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 100:
        raise EvidenceError("The requested bucket count must be an integer from 1 to 100")
    with localcontext() as ctx:
        ctx.prec = 50
        edges = sorted(set(linear_quantile(labels, Decimal(i) / n) for i in range(n + 1)))
    if len(edges) == 1:
        return [edges[0], edges[0]], [list(range(len(labels)))]
    groups = [
        [j for j, y in enumerate(labels)
         if lo <= y and (y < hi or (i == len(edges) - 2 and y <= hi))]
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))
    ]
    if sum(map(len, groups)) != len(labels):
        raise EvidenceError("Price bins must assign every validation row exactly once")
    return edges, groups


def bucket_table(labels: Sequence[Decimal], predictions: Sequence[Decimal],
                 n: int = 10) -> list[dict[str, Any]]:
    """True-price quantile intervals; final upper edge inclusive; empty bins omitted."""
    _validate_vectors(labels, predictions)
    edges, groups = _partitions(labels, n)
    with localcontext() as ctx:
        ctx.prec = 50
        total_sse = sum(((p - y) ** 2 for y, p in zip(labels, predictions)), Decimal(0))
        rows = []
        for i, indices in enumerate(groups):
            if not indices:
                continue
            rows.append({
                "bucket": f"B{len(rows) + 1:02d}", "price_lo": _float(edges[i]),
                "price_hi": _float(edges[i + 1]), "upper_inclusive": i == len(groups) - 1,
                **metrics([labels[j] for j in indices], [predictions[j] for j in indices], total_sse),
            })
        return rows


def select_examples(rows: Sequence[dict[str, Any]], per_category: int = 3) -> list[dict[str, Any]]:
    """Select distinct extreme examples deterministically, resolving ties by imageid."""
    if not isinstance(per_category, int) or isinstance(per_category, bool) or per_category < 1:
        raise EvidenceError("Examples per category must be a positive integer")
    selections = [
        ("best_absolute", sorted(rows, key=lambda r: (r["abs_error"], r["imageid"]))),
        ("largest_overprediction", sorted((r for r in rows if r["residual"] > 0),
                                         key=lambda r: (-r["residual"], r["imageid"]))),
        ("largest_underprediction", sorted((r for r in rows if r["residual"] < 0),
                                          key=lambda r: (r["residual"], r["imageid"]))),
    ]
    selected, used = [], set()
    for category, candidates in selections:
        rank = 0
        for row in candidates:
            if row["imageid"] in used:
                continue
            rank += 1
            selected.append({"category": category, "rank": rank, **row})
            used.add(row["imageid"])
            if rank == per_category:
                break
    return selected


def _paired(labels: list[Decimal], e3: list[Decimal], e5: list[Decimal], n: int) -> dict[str, Any]:
    with localcontext() as ctx:
        ctx.prec = 50
        e3_sq = [(p - y) ** 2 for y, p in zip(labels, e3)]
        e5_sq = [(p - y) ** 2 for y, p in zip(labels, e5)]
        deltas = [b - a for a, b in zip(e3_sq, e5_sq)]
        e3_sse, e5_sse = sum(e3_sq), sum(e5_sq)
        e3_bins, e5_bins = bucket_table(labels, e3, n), bucket_table(labels, e5, n)
        _, groups = _partitions(labels, n)
        groups = [group for group in groups if group]
        return {
            "reference": "E3_320px", "candidate": MODEL_TAG,
            "interpretation": "Descriptive paired comparison on the same split; not a significance test.",
            "lower_absolute_error": sum(d < 0 for d in deltas),
            "equal_absolute_error": sum(d == 0 for d in deltas),
            "higher_absolute_error": sum(d > 0 for d in deltas),
            "sse_reduction_from_improved_rows": _float(-sum(d for d in deltas if d < 0)),
            "sse_increase_from_worsened_rows": _float(sum(d for d in deltas if d > 0)),
            "net_sse_change": _float(e5_sse - e3_sse),
            "net_mse_change": _float((e5_sse - e3_sse) / len(labels)),
            "mse_change_percent": _float(100 * (e5_sse / e3_sse - 1)) if e3_sse else None,
            "overall_e3": metrics(labels, e3), "overall_e5": metrics(labels, e5),
            "price_buckets": [{
                "bucket": b["bucket"], "price_lo": b["price_lo"], "price_hi": b["price_hi"],
                "upper_inclusive": b["upper_inclusive"], "count": b["count"],
                "e3_mse": a["mse"], "e5_mse": b["mse"],
                "mse_change": _float(sum(deltas[j] for j in group) / len(group)),
                "e3_bias": a["bias"], "e5_bias": b["bias"],
                "e3_sse_share": a["sse_share"], "e5_sse_share": b["sse_share"],
                "lower_absolute_error": sum(deltas[j] < 0 for j in group),
                "equal_absolute_error": sum(deltas[j] == 0 for j in group),
                "higher_absolute_error": sum(deltas[j] > 0 for j in group),
            } for a, b, group in zip(e3_bins, e5_bins, groups)],
        }


def analyze(root: Path, *, pins: dict[str, str] | None = None,
            counts: tuple[int, int] = (7000, 1000), buckets: int = 10) -> dict[str, Any]:
    """Validate all six inputs and return diagnostics; alternate pins/counts serve tests.

    The CLI always uses the production source identities and 7000/1000 split.
    A validation failure raises EvidenceError before any output is created.
    """
    root = root.resolve()
    pins = PINNED_BLOBS if pins is None else pins
    if set(pins) != set(PINNED_BLOBS):
        raise EvidenceError("Pins must cover exactly the six Phase 5 input files")
    if len(counts) != 2 or any(not isinstance(c, int) or c < 1 for c in counts):
        raise EvidenceError("Positive training and validation counts are required")
    contents, sources, checks = {}, [], []
    for relative, expected in pins.items():
        data = _phase4.safe_file(root, relative).read_bytes()
        normalized = data.replace(b"\r\n", b"\n")
        observed = blob_sha(data)
        if observed != expected:
            raise EvidenceError(f"Source identity mismatch for {relative}: expected {expected}, found {observed}")
        contents[relative] = data
        sources.append({"path": relative, "bytes": len(data), "bytes_lf": len(normalized),
                        "sha256_lf": hashlib.sha256(normalized).hexdigest(),
                        "git_blob_sha1_lf": observed})
        checks.append({"name": "identity:" + relative, "status": "PASS"})
    train_order, train = _phase4.read_csv(contents[TRAIN], {"imageid", "price"}, {"price"}, exact=True)
    order, truth = _phase4.read_csv(contents[VAL], {"imageid", "price"}, {"price"}, exact=True)
    if (len(train_order), len(order)) != counts:
        raise EvidenceError(f"Expected train/validation counts {counts}; found {(len(train_order), len(order))}")
    if set(train) & set(truth):
        raise EvidenceError("Training and validation identifiers overlap")
    train_labels = [train[i]["price"] for i in train_order]
    labels = [truth[i]["price"] for i in order]
    if min(train_labels + labels) <= 0:
        raise EvidenceError("All source true prices must be positive")
    bounds = min(train_labels), max(train_labels)
    checks.append({"name": "fixed_split", "status": "PASS", "train_count": len(train),
                   "validation_count": len(truth), "disjoint": True})
    vectors = {}
    for tag, path in [*((tag, PRED + f"val_predictions_{tag}.csv") for tag in TAGS),
                      (MODEL_TAG, ENSEMBLE_PATH)]:
        _, rows = _phase4.read_csv(contents[path], {"imageid", "price_true", "price_pred"},
                                   {"price_true", "price_pred"}, exact=(tag == MODEL_TAG))
        vectors[tag] = _phase4.validate_predictions(order, rows, truth, "price_pred", bounds)
        checks.append({"name": "validation:" + tag, "status": "PASS", "matched_rows": len(rows)})
    expected_ensemble = _phase4.ensemble([vectors[tag] for tag in TAGS])
    predictions = vectors[MODEL_TAG]
    for ident, expected, actual in zip(order, expected_ensemble, predictions):
        if expected != actual:
            raise EvidenceError(f"E5 ensemble mismatch for {ident}: expected {expected}, found {actual}")
    checks.append({"name": "e5_equal_mean_half_even_1dp", "status": "PASS", "matched_rows": len(order)})

    price_buckets = bucket_table(labels, predictions, buckets)
    _, groups = _partitions(labels, buckets)
    bucket_by_index = {}
    for bucket, group in zip(price_buckets, (g for g in groups if g)):
        for j in group:
            bucket_by_index[j] = bucket["bucket"]
    with localcontext() as ctx:
        ctx.prec = 50
        rows = []
        for j, (ident, y, p) in enumerate(zip(order, labels, predictions)):
            residual = p - y
            rows.append({"imageid": ident, "price_true": _float(y), "price_pred": _float(p),
                         "residual": _float(residual), "abs_error": _float(abs(residual)),
                         "squared_error": _float(residual ** 2), "relative_error": _float(residual / y),
                         "abs_relative_error": _float(abs(residual / y)), "price_bucket": bucket_by_index[j]})
        absolute = [abs(p - y) for y, p in zip(labels, predictions)]
        threshold_rows = [
            {"threshold": threshold, "count": sum(error <= threshold for error in absolute),
             "fraction": _float(Decimal(sum(error <= threshold for error in absolute)) / len(labels))}
            for threshold in (50, 100)
        ]
    rows.sort(key=lambda r: r["imageid"])
    return {
        "schema_version": 1, "analyzer_version": "1.0.0", "phase": 5,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_commit": REFERENCE_COMMIT, "phase4_reference_commit": _phase4.REFERENCE_COMMIT,
        "local_git": _phase4.git_state(root), "tag": MODEL_TAG,
        "component_tags": list(TAGS), "ensemble_weights": ["1/3", "1/3", "1/3"],
        "source_prediction_path": ENSEMBLE_PATH, "input_files": sources, "checks": checks,
        "train_count": len(train), "validation_count": len(truth),
        "units": {"price_and_error": "USD 1,000", "mse_and_sse": "(USD 1,000)^2",
                  "fractions": "0 to 1", "relative_error": "(predicted - true) / true"},
        "precision": "Decimal(50) arithmetic on exact CSV label fields and one-decimal prediction exports; "
                     "E5 equals the three-component mean rounded with ROUND_HALF_EVEN to 0.1. "
                     "JSON/CSV results are finite floats; original model precision is not recovered.",
        "scope": "Descriptive analysis of the existing E5 candidate on the fixed validation split.",
        "training_executed": False, "model_loading_executed": False, "images_read": False,
        "approval_recorded_by_this_analysis": False,
        "bucket_method": {
            "requested_count": buckets, "actual_count": len(price_buckets),
            "method": "Linear quantiles at (n - 1) * q; duplicate edges collapsed; empty intervals omitted.",
            "membership": "Lower inclusive, upper exclusive, except the final upper edge is inclusive.",
            "purpose": "Data-derived diagnostic bins based on validation true prices. Do not use these "
                       "results to tune a model and then claim an independent validation estimate.",
            "zero_sse_share": "Shares are 0 when total SSE is 0.",
        },
        "overall": metrics(labels, predictions), "price_buckets": price_buckets,
        "absolute_error_thresholds": threshold_rows,
        "threshold_interpretation": "Counts with absolute error <= 50 or <= 100 (USD 1,000); "
                                    "diagnostic summaries, not course acceptance criteria.",
        "per_image_errors": rows, "representative_examples": select_examples(rows),
        "example_selection": "Up to three per category, in order: smallest absolute error, largest "
                             "positive residual, largest negative residual. No repeated imageid; ties "
                             "use lexical imageid order. These are extremes, not typical cases.",
        "paired_e3_comparison": _paired(labels, vectors["E3_320px"], predictions, buckets),
        "limitations": [
            "One fixed split and previously selected models; no repeated-seed uncertainty or independent test estimate.",
            "Aggregate improvement does not imply improvement for every price range or image.",
            "Images were not opened by this numerical analysis; image-content explanations require visual inspection.",
            "Training/validation learning curves were not re-evaluated; these residuals alone do not establish overfitting.",
        ],
    }


def analysis_markdown(report: dict[str, Any]) -> str:
    overall, paired = report["overall"], report["paired_e3_comparison"]
    lines = [
        "# Phase 5 — E5 Validation Error Analysis", "",
        f"Generated: {report['created_utc']}  ",
        f"Source: `{report['source_prediction_path']}`  ",
        f"Model: `{report['tag']}`; equal weights for E2, E3 and E4.  ",
        f"Rows: {report['validation_count']:,} validation; {report['train_count']:,} disjoint training IDs.", "",
        "## Scope and precision", "", report["scope"], "", report["precision"], "",
        "Prices, RMSE, MAE and signed bias use USD 1,000. MSE and SSE use squared USD 1,000 units. "
        "A positive residual means overprediction; a negative residual means underprediction.", "",
        "## Overall errors", "", "| Statistic | Value |", "|---|---:|",
    ]
    for label, key in (("MSE", "mse"), ("RMSE", "rmse"), ("MAE", "mae"),
                       ("Mean signed bias", "bias"), ("Median absolute error", "median_abs_error"),
                       ("90th percentile absolute error", "p90_abs_error")):
        lines.append(f"| {label} | {overall[key]:,.6f} |")
    r2_text = "Undefined (constant true prices)" if overall["r2"] is None else f"{overall['r2']:.6f}"
    lines.append(f"| R2 | {r2_text} |")
    for label, key in (("Overprediction", "overprediction_fraction"),
                       ("Underprediction", "underprediction_fraction"),
                       ("Exact prediction", "exact_prediction_fraction")):
        lines.append(f"| {label} | {overall[key]:.2%} |")
    lines.extend(["", "## Errors by true-price interval", "", report["bucket_method"]["method"], "",
                  report["bucket_method"]["membership"], "", report["bucket_method"]["purpose"], "",
                  "| Bin | True-price interval | Count | MAE | MSE | Bias | Share of total SSE |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for row in report["price_buckets"]:
        close = "]" if row["upper_inclusive"] else ")"
        lines.append(f"| {row['bucket']} | [{row['price_lo']:.12g}, {row['price_hi']:.12g}{close} | "
                     f"{row['count']} | {row['mae']:,.3f} | {row['mse']:,.3f} | "
                     f"{row['bias']:+,.3f} | {row['sse_share']:.2%} |")
    lines.extend(["", "All interval statistics, means, RMSE, error-direction fractions and SSE are in "
                  "`price_buckets.csv`. Unequal interval counts can result from tied prices.", "",
                  "## Paired E3 comparison", "", paired["interpretation"], "",
                  f"E5 has lower absolute error on **{paired['lower_absolute_error']}** rows, equal error "
                  f"on **{paired['equal_absolute_error']}**, and higher error on "
                  f"**{paired['higher_absolute_error']}**. E5 minus E3 MSE is "
                  f"**{paired['net_mse_change']:+,.6f}**.", "",
                  "| Bin | E3 MSE | E5 MSE | E5 minus E3 MSE | E3 bias | E5 bias |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in paired["price_buckets"]:
        lines.append(f"| {row['bucket']} | {row['e3_mse']:,.3f} | {row['e5_mse']:,.3f} | "
                     f"{row['mse_change']:+,.3f} | {row['e3_bias']:+,.3f} | {row['e5_bias']:+,.3f} |")
    top = paired["price_buckets"][-1]
    lines.extend(["", f"In the highest true-price interval, E5 minus E3 MSE is "
                  f"**{top['mse_change']:+,.3f}**. Aggregate performance should therefore be assessed "
                  "alongside the interval results, not treated as a guarantee for every price range.", "",
                  "## Error thresholds", "", report["threshold_interpretation"], ""])
    for row in report["absolute_error_thresholds"]:
        lines.append(f"- Absolute error <= {row['threshold']}: {row['count']} rows ({row['fraction']:.1%}).")
    lines.extend(["", "## Examples for subsequent visual inspection", "", report["example_selection"], "",
                  "| Category | Rank | Image ID | True | Predicted | Residual | Absolute error |",
                  "|---|---:|---|---:|---:|---:|---:|"])
    for row in report["representative_examples"]:
        lines.append(f"| {row['category']} | {row['rank']} | `{row['imageid']}` | "
                     f"{row['price_true']:,.3f} | {row['price_pred']:,.3f} | "
                     f"{row['residual']:+,.3f} | {row['abs_error']:,.3f} |")
    lines.extend(["", "No visual explanation is asserted by this numerical report. `representative_examples.csv` "
                  "also records signed and absolute relative errors as fractions.", "", "## Limits", ""])
    lines.extend("- " + item for item in report["limitations"])
    lines.extend(["", "## Source identities", "",
                  "Both hashes below normalize Windows CRLF to LF before hashing. "
                  "Source bytes and input identity checks are recorded in `summary.json`.", "",
                  "| Input path | LF-normalized SHA-256 |", "|---|---|"])
    lines.extend(f"| `{source['path']}` | `{source['sha256_lf']}` |" for source in report["input_files"])
    lines.extend(["", f"Reference repository commit: `{report['reference_commit']}`.  ",
                  f"Phase 4 reference commit: `{report['phase4_reference_commit']}`.  ",
                  f"Current local HEAD: `{report['local_git'].get('head')}`.", ""])
    return "\n".join(lines)


def write_report(report: dict[str, Any], root: Path, out_dir: Path) -> None:
    """Write only to a new evidence directory, protecting the Phase 4 source folder."""
    root = root.resolve()
    out_dir = (root / out_dir).resolve() if not out_dir.is_absolute() else out_dir.resolve()
    evidence_root = (root / "docs/evidence").resolve()
    if (not evidence_root.is_relative_to(root) or not out_dir.is_relative_to(evidence_root)
            or out_dir == evidence_root):
        raise EvidenceError("Output must be a NEW child folder under <repo>/docs/evidence/")
    source_dir = (root / ENSEMBLE_PATH).resolve().parent
    if out_dir.is_relative_to(source_dir):
        raise EvidenceError("Output cannot be inside the originating Phase 4 evidence folder")
    # Serialize before creating a directory, so invalid numbers cannot publish a partial report.
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    markdown = analysis_markdown(report)
    out_dir.mkdir(parents=True, exist_ok=False)
    for filename, text in (("summary.json", encoded), ("analysis.md", markdown)):
        with (out_dir / filename).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    for filename, fields, rows in (
        ("per_image_errors.csv", PER_IMAGE_FIELDS, report["per_image_errors"]),
        ("price_buckets.csv", BUCKET_FIELDS, report["price_buckets"]),
        ("representative_examples.csv", EXAMPLE_FIELDS, report["representative_examples"]),
    ):
        with (out_dir / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path,
                        help="A new folder beneath docs/evidence; relative paths use the repository root")
    parser.add_argument("--buckets", type=int, default=10, help="Diagnostic quantile intervals (default: 10)")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out_dir = args.output_dir or Path("docs/evidence") / ("phase5_run_" + stamp)
    try:
        report = analyze(root, buckets=args.buckets)
        write_report(report, root, out_dir)
    except (OSError, EvidenceError, DecimalException, ValueError) as exc:
        print(f"Phase 5 analysis failed: {exc}", file=sys.stderr)
        return 1
    print(f"Validated {report['validation_count']} E5 rows against the fixed split and all three components.")
    print(f"MSE {report['overall']['mse']:.6f}; MAE {report['overall']['mae']:.6f}; "
          f"bias {report['overall']['bias']:+.6f} (predicted - true).")
    print("Written:", out_dir if out_dir.is_absolute() else root / out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
