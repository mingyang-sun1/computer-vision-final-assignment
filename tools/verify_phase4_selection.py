"""Read-only Phase 4 comparison of existing, pinned repository predictions.

Python 3.10+; standard library only. No training, model loading, downloads,
image access, Kaggle access, or source-file changes. Only a NEW evidence
folder is written. Candidate selection is provisional, never team approval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, DecimalException, InvalidOperation, ROUND_HALF_EVEN, localcontext
from html import escape
from pathlib import Path
from typing import Any, Iterable

REFERENCE_COMMIT = "90959c41384c7bd3c8c5a48147e8272246071bbe"
TAGS = ("E2_finetune", "E3_320px", "E4_augment")
TRAIN = "data/splits/train_split.csv"
VAL = "data/splits/val_split.csv"
PRED = "outputs/predictions/"
# Git blob identities from the inspected main snapshot. Normalize CRLF only.
PINNED_BLOBS = {
    TRAIN: "05851b5173130d45b78f1057ec7f0e78cf827a06",
    VAL: "54c0018bb7fbb08c59dd26c511928836e5ea46ba",
    PRED + "val_predictions_E2_finetune.csv": "6ed8deb3847516e33345eab6ef40646142c06986",
    PRED + "val_predictions_E3_320px.csv": "1b838b037d504375033644ee0a84c6760dd41f24",
    PRED + "val_predictions_E4_augment.csv": "6260839aede5cc7e59de6de6179782582fdd5a7a",
    PRED + "submission_phase2_ridge.csv": "856a86573a39ee4c6e24b55dc046632420906f3d",
    PRED + "submission_phase3_E2_finetune.csv": "dcaefb73744451a25bd5cdcbb45548407352de87",
    PRED + "submission_phase3_E3_320px.csv": "4c12698c2793f198e7d95a80de541cea28d3d58a",
    PRED + "submission_phase3_E4_augment.csv": "e9cab1f3d75311c50b69462268a41caca25fcee7",
    PRED + "submission_ensemble.csv": "bceaab97bb957dade32f85e16d55daf3e20192c5",
    PRED + "v3_val_predictions.csv": "edfa94a798d953803000b07334a8525b36c47090",
    "src/ensemble.py": "618a36b99eb2487fda30bd19a707cc21ccc2b5b9",
}
CENT = Decimal("0.1")


class EvidenceError(ValueError):
    """An input cannot support the proposed comparison."""


def blob_sha(data: bytes) -> str:
    data = data.replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def number(value: str, context: str) -> Decimal:
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise EvidenceError(f"{context}: invalid number {value!r}") from exc
    if not result.is_finite():
        raise EvidenceError(f"{context}: non-finite number")
    if abs(result) > Decimal("1e12"):
        raise EvidenceError(f"{context}: magnitude exceeds the supported price range")
    return result


def read_csv(data: bytes, required: set[str], numeric: set[str], *, exact: bool = False
             ) -> tuple[list[str], dict[str, dict[str, Any]]]:
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise EvidenceError(f"Invalid CSV columns: {fields}; required: {sorted(required)}")
        if exact and set(fields) != required:
            raise EvidenceError(f"Unexpected CSV columns: {fields}")
        order, rows = [], {}
        for line, row in enumerate(reader, start=2):
            if None in row or any(v is None for v in row.values()):
                raise EvidenceError(f"Malformed CSV row {line}")
            ident = row["imageid"]
            if not ident or ident != ident.strip() or ident in rows:
                raise EvidenceError(f"Blank, padded, or duplicate imageid at row {line}")
            parsed = dict(row)
            for field in numeric:
                parsed[field] = number(row[field], f"row {line}/{field}")
            order.append(ident)
            rows[ident] = parsed
        if not rows:
            raise EvidenceError("Empty CSV")
        return order, rows
    except (UnicodeError, csv.Error) as exc:
        raise EvidenceError(f"Unreadable CSV: {exc}") from exc


def same_ids(reference: Iterable[str], other: Iterable[str]) -> None:
    ref, actual = set(reference), set(other)
    if ref != actual:
        raise EvidenceError(f"Identifier mismatch: missing={len(ref-actual)}, extra={len(actual-ref)}")


def metrics(labels: list[Decimal], predictions: list[Decimal]) -> dict[str, float]:
    if any(not x.is_finite() for x in labels + predictions):
        raise EvidenceError("Metrics require finite labels and predictions")
    if not labels or len(labels) != len(predictions):
        raise EvidenceError("Nonempty, equally sized labels and predictions are required")
    with localcontext() as ctx:
        ctx.prec = 50
        n = Decimal(len(labels))
        mean = sum(labels) / n
        ss = sum((y - mean) ** 2 for y in labels)
        if not ss:
            raise EvidenceError("R2 is undefined for constant validation labels")
        errors = [p-y for y, p in zip(labels, predictions)]
        sse = sum(e*e for e in errors)
        return {"mse": float(sse/n), "mae": float(sum(abs(e) for e in errors)/n),
                "r2": float(1-sse/ss)}


def ensemble(values: list[list[Decimal]], *, rounded: bool = True) -> list[Decimal]:
    if len(values) != 3 or not values[0] or len({len(v) for v in values}) != 1:
        raise EvidenceError("Exactly three nonempty equally sized prediction vectors are required")
    with localcontext() as ctx:
        ctx.prec = 50
        result = [sum(parts)/Decimal(3) for parts in zip(*values)]
        return [x.quantize(CENT, rounding=ROUND_HALF_EVEN) for x in result] if rounded else result


def validate_predictions(order: list[str], rows: dict[str, dict[str, Any]],
                         truth: dict[str, dict[str, Any]], column: str,
                         bounds: tuple[Decimal, Decimal], *, grid: bool = True) -> list[Decimal]:
    same_ids(order, rows)
    result = []
    for ident in order:
        row = rows[ident]
        if row["price_true"] != truth[ident]["price"]:
            raise EvidenceError(f"Label mismatch for {ident}")
        p = row[column]
        if not bounds[0] <= p <= bounds[1]:
            raise EvidenceError(f"Prediction outside training-label bounds for {ident}")
        if grid and p != p.quantize(CENT):
            raise EvidenceError(f"Prediction not on the 0.1 export grid for {ident}")
        result.append(p)
    return result


def safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise EvidenceError(f"Missing file or unsafe path: {relative}")
    return path


def git_state(root: Path) -> dict[str, Any]:
    result = {}
    for key, args in (("head", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])):
        try:
            p = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                               text=True, timeout=10, check=False)
            result[key] = p.stdout.strip() if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result[key] = None
    return result


def audit(root: Path, *, pins: dict[str, str] | None = None,
          counts: tuple[int, int, int] = (7000, 1000, 3000),
          test_manifest: Path | None = None) -> dict[str, Any]:
    """Audit inputs. Alternate pins/counts are for synthetic unit fixtures only.

    The command-line interface always uses the production pins and counts.
    """
    root = root.resolve()
    pins = PINNED_BLOBS if pins is None else pins
    report: dict[str, Any] = {
        "schema_version": 2, "checker_version": "2.0.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_commit": REFERENCE_COMMIT, "local_git": git_state(root),
        "scope": "Revised Phase 4 / Member A: existing model comparison and provisional selection",
        "training_executed": False, "model_loading_executed": False, "images_read": False,
        "kaggle_score_verified": False, "test_scores_used_for_selection": False,
        "checks": [], "input_files": [], "metrics": {}, "artifacts": [],
        "reported_reference_mse": {"B1": 114853.7, "E3": 99051.0, "E5": 95850.6},
        "reported_references_recomputed": False,
        "artifact_inventory_is_phase7_follow_up": True,
        "candidate": None, "joint_approval": "PENDING",
        "precision": "Decimal(50) arithmetic on exported predictions; JSON metrics are floats. "
                     "The deployed-style ensemble rounds the mean to one decimal. "
                     "Historical full-precision run scores are not recovered.",
    }
    checks = report["checks"]

    def check(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    contents = {}
    for relative, expected in pins.items():
        try:
            data = safe_file(root, relative).read_bytes()
            observed = blob_sha(data)
            report["input_files"].append({"path": relative, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "git_blob_sha1_lf": observed})
            if observed != expected:
                raise EvidenceError(f"Expected {expected}; found {observed}; original not modified")
            contents[relative] = data
            check("identity:" + relative, "PASS", "LF-normalized identity matches inspected snapshot")
        except (OSError, EvidenceError) as exc:
            check("identity:" + relative, "FAIL", str(exc))
    if any(c["status"] == "FAIL" for c in checks):
        return finish(report)

    try:
        train_order, train = read_csv(contents[TRAIN], {"imageid", "price"}, {"price"}, exact=True)
        order, truth = read_csv(contents[VAL], {"imageid", "price"}, {"price"}, exact=True)
        if (len(train_order), len(order)) != counts[:2]:
            raise EvidenceError(f"Expected train/validation counts {counts[:2]}")
        if set(train) & set(truth):
            raise EvidenceError("Training and validation identifiers overlap")
        labels = [truth[i]["price"] for i in order]
        train_labels = [train[i]["price"] for i in train_order]
        if min(train_labels + labels) <= 0:
            raise EvidenceError("All source prices must be positive")
        bounds = min(train_labels), max(train_labels)
        check("fixed_split", "PASS", f"{len(train)} training / {len(truth)} validation; disjoint IDs")
        vectors = {}
        for tag in TAGS:
            _, rows = read_csv(contents[PRED+f"val_predictions_{tag}.csv"],
                               {"imageid", "price_true", "price_pred"}, {"price_true", "price_pred"})
            vectors[tag] = validate_predictions(order, rows, truth, "price_pred", bounds)
            report["metrics"][tag] = metrics(labels, vectors[tag])
            check("validation:"+tag, "PASS", "Unique IDs aligned by imageid; labels, range and export grid checked")
        with localcontext() as ctx:
            ctx.prec = 50
            mean = sum(train_labels)/Decimal(len(train_labels))
        report["metrics"]["B0_mean"] = metrics(labels, [mean]*len(order))
        mean_raw = ensemble([vectors[t] for t in TAGS], rounded=False)
        mean_export = ensemble([vectors[t] for t in TAGS])
        report["metrics"]["E5_mean_unrounded"] = metrics(labels, mean_raw)
        report["metrics"]["E5_mean_export_1dp"] = metrics(labels, mean_export)
        report["paired_comparisons"] = {}
        for left, right in (("E3_320px", "E2_finetune"), ("E4_augment", "E2_finetune"),
                            ("E5_mean_export_1dp", "E3_320px")):
            lhs = mean_export if left.startswith("E5") else vectors[left]
            rhs = vectors[right]
            comparisons = [abs(a-y).compare(abs(b-y)) for a,b,y in zip(lhs,rhs,labels)]
            report["paired_comparisons"][left+"_vs_"+right] = {
                "lower_absolute_error": sum(x < 0 for x in comparisons),
                "equal_absolute_error": sum(x == 0 for x in comparisons),
                "higher_absolute_error": sum(x > 0 for x in comparisons),
                "mse_change_percent": 100*(report["metrics"][left]["mse"] /
                                           report["metrics"][right]["mse"] - 1)
                                      if report["metrics"][right]["mse"] else None,
                "interpretation": "Descriptive paired comparison, not a significance test.",
            }

        report["ensemble_validation_rows"] = [
            {"imageid": i, "price_true": str(y), "price_pred": str(p)}
            for i, y, p in zip(order, labels, mean_export)]
        # Include the existing alternative CSV as diagnostic evidence, not an invented new run.
        _, v3 = read_csv(contents[PRED+"v3_val_predictions.csv"],
            {"imageid", "price_true", "price_pred_224", "price_pred_320"},
            {"price_true", "price_pred_224", "price_pred_320"}, exact=True)
        for size in (224, 320):
            predictions = validate_predictions(order, v3, truth, f"price_pred_{size}", bounds, grid=False)
            report["metrics"][f"V3_stored_{size}"] = metrics(labels, predictions)
        check("v3_validation", "PASS", "Both existing columns checked; original run identity remains unconfirmed")

        ref_order, baseline_test = read_csv(contents[PRED+"submission_phase2_ridge.csv"],
                                           {"imageid", "price"}, {"price"}, exact=True)
        if len(ref_order) != counts[2]:
            raise EvidenceError(f"Expected {counts[2]} submission rows")
        if set(ref_order) & (set(train_order) | set(order)):
            raise EvidenceError("Test submission identifiers overlap labelled splits")
        submission_vectors = []
        for tag in TAGS:
            ids, rows = read_csv(contents[PRED+f"submission_phase3_{tag}.csv"],
                                 {"imageid", "price"}, {"price"}, exact=True)
            if ids != ref_order:
                raise EvidenceError(f"{tag}: submission ID order differs from pinned baseline submission")
            values = [rows[i]["price"] for i in ref_order]
            if any(p != p.quantize(CENT) or not bounds[0] <= p <= bounds[1] for p in values):
                raise EvidenceError(f"{tag}: invalid range or one-decimal export grid")
            submission_vectors.append(values)
        ids, existing = read_csv(contents[PRED+"submission_ensemble.csv"],
                                 {"imageid", "price"}, {"price"}, exact=True)
        if ids != ref_order:
            raise EvidenceError("Ensemble submission ID order mismatch")
        expected = ensemble(submission_vectors)
        mismatches = sum(existing[i]["price"] != p for i, p in zip(ref_order, expected))
        if mismatches:
            raise EvidenceError(f"Ensemble submission differs from mean-and-round policy on {mismatches} rows")
        check("submission_arithmetic", "PASS", f"{len(ids)} rows match three-component mean rounded to 1 dp")
        if test_manifest is None:
            check("test_manifest", "PENDING", "Optional course test.csv not supplied. "
                  "The stored submissions agree internally; external manifest verification is deferred.")
        else:
            manifest_path = test_manifest.expanduser().resolve()
            manifest_data = manifest_path.read_bytes()
            manifest_ids, _ = read_csv(manifest_data, {"imageid"}, set(), exact=True)
            if manifest_ids != ref_order:
                raise EvidenceError("Course test.csv identifier order/population differs from submissions")
            report["external_test_manifest"] = {
                "path": str(manifest_path), "rows": len(manifest_ids),
                "sha256": hashlib.sha256(manifest_data).hexdigest(),
                "labels_read": False,
            }
            check("test_manifest", "PASS", "Course test.csv matches submission IDs and order; no labels read")
        # Deliberately never use a Kaggle/test score for ranking.
        core = list(TAGS) + ["E5_mean_export_1dp"]
        winner = min(core, key=lambda tag: (report["metrics"][tag]["mse"], tag))
        best_mse = report["metrics"][winner]["mse"]
        tied = [t for t in core if report["metrics"][t]["mse"] == best_mse]
        if len(tied) > 1:
            check("selection", "PENDING", f"Tied validation MSE: {tied}; team decision required")
        elif report["metrics"]["B0_mean"]["mse"] <= best_mse:
            check("selection", "PENDING", "Mean-price baseline is at least as good; inspect before selecting")
        elif min(report["metrics"][f"V3_stored_{s}"]["mse"] for s in (224, 320)) < best_mse:
            check("selection", "PENDING", "A V3 stored column scores better; identify its original run before final selection")
        else:
            report["candidate"] = {"status": "PROVISIONAL", "id": winner,
                "selection_metric": "validation MSE from the pinned rounded CSV evidence",
                "validation_mse": best_mse, "scope": core,
                "review_date": None, "approved_by": [],
                "weights": ["1/3", "1/3", "1/3"] if winner.startswith("E5") else None,
                "source_tags": list(TAGS) if winner.startswith("E5") else [winner],
                "joint_approval": "PENDING", "checkpoint_provenance_verified": False}
            check("selection", "PASS", f"Numerical candidate only: {winner}; not final approval")
    except (OSError, KeyError, EvidenceError, DecimalException, OverflowError) as exc:
        report["candidate"] = None
        check("csv_audit", "FAIL", str(exc))

    # Inventory is intentionally separate from successful model loading/provenance.
    for tag in TAGS:
        for rel in (f"outputs/logs/phase3_{tag}.json", f"outputs/models/finetuned_{tag}.keras"):
            path = root / rel
            present = path.is_file() and path.resolve().is_relative_to(root) and path.stat().st_size > 0
            report["artifacts"].append({"path": rel, "present": present,
                "bytes": path.stat().st_size if present else None,
                "loaded_or_provenance_verified": False})
            check("inventory:"+rel, "PASS" if present else "PENDING",
                  "Nonempty local file only; not executed or matched to predictions" if present else
                  "Not found locally; an existing shared-storage artifact is acceptable for manual handoff")
    check("original_artifacts", "PENDING", "Phase 7 reproducibility follow-up. "
          "E2/E4 receipt is already recorded in Phase3_MemberB_Artifact_Update.md. "
          "Do not request those files again solely because they are absent from GitHub. "
          "Inventory is not checkpoint-to-prediction provenance.")
    check("v3_run_identity", "PENDING", "Supplementary V3 comparison requires its original run identity. "
          "Do not infer backbone or preprocessing from the column names alone.")
    check("joint_review", "PENDING", "Member A and Member B must approve the candidate and record actual review date")
    return finish(report)


def finish(report: dict[str, Any]) -> dict[str, Any]:
    counter = Counter(c["status"] for c in report["checks"])
    report["summary"] = {s: counter[s] for s in ("PASS", "FAIL", "PENDING")}
    report["status"] = "FAILED" if counter["FAIL"] else "PROVISIONAL_REVIEW_REQUIRED"
    if counter["FAIL"]:
        report["candidate"] = None
    return report


def comparison_markdown(report: dict[str, Any]) -> str:
    lines = ["# Phase 4: executed CSV comparison", "", f"Reference snapshot: `{REFERENCE_COMMIT}`.",
        f"Audit status: **{report['status']}**. Joint approval: **PENDING**.", "",
        "Scores below are recomputed from stored CSVs, not new training results. MSE is in squared",
        "USD-1,000 units; MAE is in USD 1,000; R2 is unitless. No test score is used for selection.", "",
        "| Candidate / reference | Validation MSE | MAE | R2 |", "|---|---:|---:|---:|"]
    for tag, m in report["metrics"].items():
        lines.append(f"| {tag} | {m['mse']:,.6f} | {m['mae']:,.6f} | {m['r2']:.6f} |")
    lines += ["", "The unrounded E5 row is a precision reference. Selection uses E5_mean_export_1dp,",
        "which matches the existing submission mean-then-round policy. V3 columns are diagnostic",
        "until their original model/run identity is confirmed. Historical B1/B2/E1 scores are in",
        "`docs/tables/Phase4_Historical_Comparison.md`; they are not silently treated as recomputed.", "",
        "## Candidate", "", "```json", json.dumps(report["candidate"], indent=2), "```", "", "## Checks", ""]
    for c in report["checks"]:
        lines.append(f"- **{c['status']}** `{c['name']}`: {c['detail']}")
    return "\n".join(lines) + "\n"


def svg_chart(values: dict[str, dict[str, float]]) -> str:
    """Generate a self-contained quantitative SVG, with no external assets."""
    rows = [(k, v["mse"]) for k, v in values.items() if k != "E5_mean_unrounded"]
    width, height, left, plot_width = 1000, 130 + len(rows)*48, 245, 615
    maximum = max((v for _, v in rows), default=1) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
             '<title>Phase 4 validation MSE from pinned CSVs</title>',
             '<g font-family="sans-serif" font-size="16">',
             '<text x="24" y="28" font-size="21">Phase 4 — executed CSV validation comparison</text>',
             '<text x="24" y="53" font-size="13">Lower is better. Stored predictions only; no new training or test-set selection.</text>']
    for index, (name, value) in enumerate(rows):
        y, length = 83+index*48, plot_width*value/maximum
        parts += [f'<text x="12" y="{y+17}">{escape(name)}</text>',
                  f'<rect x="{left}" y="{y}" width="{length:.2f}" height="24"/>',
                  f'<text x="{left+length+10:.2f}" y="{y+17}">{value:,.1f}</text>']
    parts += [f'<text x="{left}" y="{height-18}" font-size="13">MSE in squared (USD 1,000) units; zero-length bars represent zero MSE.</text>',
              '</g></svg>']
    return "\n".join(parts)


def write_report(report: dict[str, Any], root: Path, out_dir: Path) -> None:
    root, out_dir = root.resolve(), out_dir.resolve()
    evidence_root = (root / "docs/evidence").resolve()
    if not out_dir.is_relative_to(evidence_root) or out_dir == evidence_root:
        raise EvidenceError("Output must be a NEW child folder under <repo>/docs/evidence/")
    # mkdir(exist_ok=False) rejects repeats, including prior successful audit folders.
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (out_dir / "comparison.md").write_text(comparison_markdown(report), encoding="utf-8")
    if report["summary"]["FAIL"]:
        return  # Never publish a success chart or candidate following a failed audit.
    (out_dir / "validation_mse.svg").write_text(svg_chart(report["metrics"]), encoding="utf-8")
    with (out_dir / "comparison.csv").open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate", "mse", "mae", "r2"])
        w.writeheader()
        w.writerows({"candidate": name, **m} for name, m in report["metrics"].items())
    with (out_dir / "validation_ensemble.csv").open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["imageid", "price_true", "price_pred"])
        w.writeheader()
        w.writerows(report.get("ensemble_validation_rows", []))
    (out_dir / "selection_candidate.json").write_text(
        json.dumps(report["candidate"], indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, help="New folder beneath docs/evidence; relative to repo root")
    parser.add_argument("--test-manifest", type=Path, help="Optional original course test.csv (imageid only); no images or labels")
    parser.add_argument("--require-complete", action="store_true", help="Exit 2 when manual/artifact items remain pending")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = args.out_dir or Path("docs/evidence") / ("phase4_run_"+stamp)
    if not out.is_absolute():
        out = root/out
    print("Checking pinned Phase 4 inputs in:", root, flush=True)
    report = audit(root, test_manifest=args.test_manifest)
    try:
        write_report(report, root, out)
    except (OSError, EvidenceError) as exc:
        print(f"Output error: {exc}", file=sys.stderr)
        return 1
    print("Phase 4:", report["summary"])
    print("Candidate:", report["candidate"]["id"] if report["candidate"] else "NOT DETERMINED")
    print("Written:", out)
    print("No training, images, model loading, source changes or automatic final approval.")
    if report["summary"]["FAIL"]:
        return 1
    return 2 if args.require_complete and report["summary"]["PENDING"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
