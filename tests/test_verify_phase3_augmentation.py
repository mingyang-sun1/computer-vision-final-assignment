"""Synthetic evidence-checker tests; no scientific training results or models.

All synthetic logs/checkpoint markers live in temporary test directories only.
These tests exercise data validation and audit logic, not model performance.
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/verify_phase3_augmentation.py"
spec = importlib.util.spec_from_file_location("phase3_checker", MODULE_PATH)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class CheckerUnitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_metrics_known_values_and_undefined_r2(self):
        result = checker.metric_summary([Decimal(1), Decimal(3)], [Decimal(2), Decimal(2)])
        self.assertEqual(result["metrics"], {"mse": 1.0, "mae": 1.0, "r2": 0.0})
        self.assertEqual(result["metrics_decimal"]["mse"], "1")
        self.assertIsNone(checker.metric_summary([Decimal(1)], [Decimal(2)])["metrics"]["r2"])
        for a, p in (([], []), ([Decimal(1)], []), ([Decimal("NaN")], [Decimal(1)])):
            with self.subTest(a=a, p=p), self.assertRaises(ValueError):
                checker.metric_summary(a, p)

    def test_csv_counterexamples_are_rejected(self):
        invalid = [
            "imageid,value\na.jpg,1\n", "imageid,price\n ,1\n",
            "imageid,price\n a.jpg,1\n", "imageid,price\na.jpg,1\na.jpg,2\n",
            "imageid,price\na.jpg\n", "imageid,price\na.jpg,1,extra\n",
            "imageid,price\na.jpg,NaN\n", "imageid,price\na.jpg,inf\n",
            "imageid,price\na.jpg,1e309\n", "imageid,price\na.jpg,words\n",
            "imageid,price\n",
        ]
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(checker.INPUT_ERRORS):
                checker.parse_table(content.encode(), ["imageid", "price"])

    def test_lf_identity_matches_crlf_and_preserves_exact_hash(self):
        lf, crlf = self.root / "lf.csv", self.root / "crlf.csv"
        lf.write_bytes(b"imageid,price\na.jpg,1\n")
        crlf.write_bytes(b"imageid,price\r\na.jpg,1\r\n")
        a, b = checker.file_identity(lf), checker.file_identity(crlf)
        self.assertEqual(a["git_blob_sha1_lf"], b["git_blob_sha1_lf"])
        self.assertNotEqual(a["sha256"], b["sha256"])
        self.assertEqual(a["path"], str(lf.resolve()))

    def test_json_rejects_malformed_duplicate_and_nonfinite_values(self):
        for raw in (b"{", b"[]", b'{"x":1,"x":2}', b'{"x":NaN}',
                    b'{"x":[Infinity]}', b'{"nested":{"value":1e309}}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                checker.load_object(raw)
        self.assertEqual(checker.load_object(b'{"a":[1,2]}'), {"a": [1, 2]})

    def test_exclusive_json_write_and_nonfinite_output_rejection(self):
        path = self.root / "audit.json"
        checker.write_new_json(path, {"first": True})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            checker.write_new_json(path, {"second": True})
        self.assertEqual(path.read_bytes(), before)
        invalid = self.root / "invalid.json"
        with self.assertRaises(ValueError):
            checker.write_new_json(invalid, {"bad": float("nan")})
        self.assertFalse(invalid.exists())


class SyntheticIntegrationTests(unittest.TestCase):
    """8,000 synthetic labels; temporary fixture files are never trained/loaded."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "synthetic_repo"
        self.train = [[f"train_{i}.jpg", 100 + 2 * (i % 100)] for i in range(7000)]
        self.val = [[f"val_{i}.jpg", 150 + 2 * (i % 50)] for i in range(1000)]
        self.train_path = self.repo / "data/splits/train_split.csv"
        self.val_path = self.repo / "data/splits/val_split.csv"
        self.write_table(self.train_path, ["imageid", "price"], self.train)
        self.write_table(self.val_path, ["imageid", "price"], self.val)
        self.csv_paths, self.rows = {}, {}
        for tag in checker.TAGS:
            rows = []
            for i, (imageid, target) in enumerate(self.val):
                error = 5.0 if tag == "E2_finetune" else (3.0 if i % 2 == 0 else 7.0)
                rows.append([imageid, target, target + error, error, error])
            if tag == "E4_augment":
                rows.reverse()  # The real E2/E4 exports are sorted independently.
            self.rows[tag] = rows
            self.csv_paths[tag] = self.repo / "outputs/predictions" / f"val_predictions_{tag}.csv"
            self.save_rows(tag)
        expected = {p.name: checker.file_identity(p)["git_blob_sha1_lf"] for p in (self.train_path, self.val_path)}
        split_patch = patch.dict(checker.EXPECTED_SPLITS, expected)
        split_patch.start()
        self.addCleanup(split_patch.stop)
        historical_patch = patch.dict(checker.HISTORICAL_MSE, {"E2_finetune": 25.0, "E4_augment": 29.0})
        historical_patch.start()
        self.addCleanup(historical_patch.stop)

    @staticmethod
    def write_table(path, columns, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)

    def save_rows(self, tag):
        self.write_table(self.csv_paths[tag], checker.PRED_COLUMNS, self.rows[tag])

    def add_synthetic_artifacts(self):
        """Schema fixtures and labelled marker bytes, never real training evidence."""
        report = checker.audit(self.repo)
        self.logs = {}
        for tag in checker.TAGS:
            log = {"experiment": tag, "created_utc": "2026-01-01T00:00:00+00:00",
                   "seed": 42, "img_size": 224, "augmentation": tag == "E4_augment",
                   "unfrozen": "layer4 (conv5) + regression head", "learning_rate": 1e-4,
                   "batch_size": 32, "epochs_cap": 15, "patience": 4,
                   "epochs_run": 3, "best_epoch": 2,
                   "trainable_params": 5, "total_params": 10,
                   "val": report["runs"][tag]["metrics"],
                   "history": {"loss": [0.5, 0.4, 0.3], "val_loss": [0.6, 0.45, 0.47]}}
            self.logs[tag] = log
            self.save_log(tag)
            path = self.repo / "outputs/models" / f"finetuned_{tag}.keras"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"SYNTHETIC INVENTORY FIXTURE; NOT A MODEL")

    def log_path(self, tag):
        return self.repo / "outputs/logs" / f"phase3_{tag}.json"

    def save_log(self, tag):
        path = self.log_path(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs[tag]), encoding="utf-8")

    def assert_failed_check(self, result, name):
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(any(c["name"] == name and c["status"] == "FAIL" for c in result["checks"]), result["checks"])

    def test_missing_original_artifacts_are_pending_and_inputs_untouched(self):
        before = {p: p.read_bytes() for p in (self.train_path, self.val_path, *self.csv_paths.values())}
        result = checker.audit(self.repo)
        self.assertEqual(result["summary"]["fail"], 0)
        self.assertEqual(result["summary"]["pending"], 5)
        self.assertEqual(result["csv_analysis_status"], "PASS")
        for flag in ("training_executed", "model_loading_executed", "images_read", "kaggle_score_verified"):
            self.assertFalse(result[flag])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(len(result["files_read"]), 4)

    def test_comparison_joins_ids_independently_of_row_order(self):
        first = checker.audit(self.repo)
        expected = first["comparison"]
        self.assertEqual(expected["mse_delta_e4_minus_e2"], 4.0)
        self.assertEqual(expected["mse_change_percent"], 16.0)
        self.assertEqual((expected["e4_better_count"], expected["e4_worse_count"], expected["equal_count"]), (500, 500, 0))
        self.rows["E2_finetune"].reverse()
        self.rows["E4_augment"] = self.rows["E4_augment"][317:] + self.rows["E4_augment"][:317]
        for tag in checker.TAGS:
            self.save_rows(tag)
        second = checker.audit(self.repo)
        self.assertEqual(second["comparison"], expected)
        self.assertEqual(second["runs"]["E2_finetune"]["metrics"], first["runs"]["E2_finetune"]["metrics"])

    def test_source_label_mismatch_blocks_comparison(self):
        self.rows["E2_finetune"][0][1] += 1
        self.save_rows("E2_finetune")
        result = checker.audit(self.repo)
        self.assert_failed_check(result, "E2_finetune_source_labels")
        self.assertIsNone(result["comparison"])

    def test_missing_and_extra_validation_ids_block_comparison(self):
        original = [r[:] for r in self.rows["E4_augment"]]
        for rows in (original[:-1], original + [["extra.jpg", 200, 203, 3, 3]]):
            with self.subTest(row_count=len(rows)):
                self.rows["E4_augment"] = rows
                self.save_rows("E4_augment")
                result = checker.audit(self.repo)
                self.assert_failed_check(result, "E4_augment_coverage")
                self.assertIsNone(result["comparison"])

    def test_frozen_split_count_and_disjointness_are_enforced(self):
        original = self.train_path.read_bytes()
        self.write_table(self.train_path, ["imageid", "price"], self.train[:-1])
        result = checker.audit(self.repo)
        self.assert_failed_check(result, "identity_train_split.csv")
        self.assert_failed_check(result, "split_counts")
        self.assertIsNone(result["comparison"])
        self.train_path.write_bytes(original)
        overlap = [r[:] for r in self.val]
        overlap[0][0] = self.train[0][0]
        self.write_table(self.val_path, ["imageid", "price"], overlap)
        result = checker.audit(self.repo)
        self.assert_failed_check(result, "split_disjoint")

    def test_independently_rounded_errors_accept_tolerance_but_reject_corruption(self):
        row = self.rows["E2_finetune"][0]
        row[3], row[4] = 4.9, 4.9
        self.save_rows("E2_finetune")
        result = checker.audit(self.repo)
        self.assertEqual(result["summary"]["fail"], 0)
        self.assertEqual(result["runs"]["E2_finetune"]["metrics"]["mae"], 5.0)
        row[4] = 5.2
        self.save_rows("E2_finetune")
        result = checker.audit(self.repo)
        self.assert_failed_check(result, "E2_finetune_rounded_errors")
        self.assertIsNone(result["comparison"])
        for residual, absolute in ((5.1, 4.9), (5.05, 5.05)):
            with self.subTest(residual=residual, absolute=absolute):
                row[3], row[4] = residual, absolute
                self.save_rows("E2_finetune")
                result = checker.audit(self.repo)
                self.assert_failed_check(result, "E2_finetune_rounded_errors")
                self.assertIsNone(result["comparison"])

    def test_training_range_clipping_is_checked(self):
        row = self.rows["E4_augment"][0]
        row[2] = 300.0
        row[3] = row[2] - row[1]
        row[4] = abs(row[3])
        self.save_rows("E4_augment")
        result = checker.audit(self.repo)
        self.assert_failed_check(result, "E4_augment_clipping")
        self.assertIsNone(result["comparison"])

    def test_supplied_schema_fixtures_pass_without_claiming_model_verification(self):
        self.add_synthetic_artifacts()
        result = checker.audit(self.repo)
        self.assertEqual(result["summary"]["fail"], 0, result["checks"])
        self.assertEqual(result["summary"]["pending"], 0)
        self.assertIn("NOT_A_TRAINING_REPRODUCTION", result["status"])
        self.assertFalse(result["runs"]["E2_finetune"]["model_inventory"]["loaded"])
        self.assertEqual(len(result["files_read"]), 6)  # Checkpoints are statted, not read.

    def test_legacy_member_b_schema_passes_with_explicit_metadata_gaps(self):
        """Legacy log fields are never fabricated to pass the audit."""
        self.add_synthetic_artifacts()
        e2 = self.logs["E2_finetune"]
        e2["experiment"] = "E2_partial_finetuning"
        for key in ("augmentation", "epochs_cap", "patience", "best_epoch"):
            e2.pop(key)
        self.save_log("E2_finetune")
        e4 = self.logs["E4_augment"]
        e4.pop("patience")
        self.save_log("E4_augment")
        result = checker.audit(self.repo)
        self.assertEqual(result["summary"]["fail"], 0, result["checks"])
        self.assertEqual(result["summary"]["pending"], 3, result["checks"])
        self.assertEqual(result["runs"]["E2_finetune"]["supplied_log"], e2)
        self.assertEqual(result["runs"]["E2_finetune"]["log_observations"]["best_epoch_from_history"], 2)
        for tag in checker.TAGS:
            self.assertTrue(any(c["name"] == f"{tag}_legacy_metadata_gaps" and
                                c["status"] == "PENDING" for c in result["checks"]))
        self.assertTrue(any(c["name"] == "matched_run_conditions" and
                            c["status"] == "PASS" for c in result["checks"]))
        self.assertTrue(any(c["name"] == "matched_run_conditions_unrecorded" and
                            c["status"] == "PENDING" for c in result["checks"]))

    def test_missing_required_or_conflicting_legacy_metadata_still_fails(self):
        self.add_synthetic_artifacts()
        for tag, changed in (("E4_augment", {"augmentation": None}),
                             ("E2_finetune", {"augmentation": True}),
                             ("E2_finetune", {"seed": None}),
                             ("E2_finetune", {"experiment": "unrelated"})):
            with self.subTest(tag=tag, changed=changed):
                before = self.logs[tag].copy()
                for key, value in changed.items():
                    if value is None:
                        self.logs[tag].pop(key)
                    else:
                        self.logs[tag][key] = value
                self.save_log(tag)
                result = checker.audit(self.repo)
                self.assert_failed_check(result, f"{tag}_original_log")
                self.logs[tag] = before
                self.save_log(tag)

    def test_wrong_settings_history_and_best_epoch_fail_without_losing_csv_analysis(self):
        self.add_synthetic_artifacts()
        original = json.loads(json.dumps(self.logs["E4_augment"]))
        invalid = [
            {"augmentation": False}, {"seed": True}, {"img_size": 320},
            {"epochs_run": 16}, {"best_epoch": 1},
            {"history": {"loss": [0.5], "val_loss": [0.6, 0.45, 0.47]}},
            {"history": {"loss": [0.5, -0.1, 0.3], "val_loss": [0.6, 0.45, 0.47]}},
            {"history": {"loss": [0.5, 0.4, 0.3], "val_loss": [0.4, 0.4, 0.5]}},
            {"trainable_params": 11}, {"created_utc": "2026-01-01T12:00:00"},
        ]
        for change in invalid:
            with self.subTest(change=change):
                self.logs["E4_augment"] = dict(original, **change)
                self.save_log("E4_augment")
                result = checker.audit(self.repo)
                self.assert_failed_check(result, "E4_augment_original_log")
                self.assertEqual(result["csv_analysis_status"], "PASS")
                self.assertIsNotNone(result["comparison"])

    def test_malformed_and_nonfinite_original_logs_are_failed_reports(self):
        self.add_synthetic_artifacts()
        for content in ('{', '{"bad":NaN}', '{"bad":1e309}'):
            with self.subTest(content=content):
                self.log_path("E2_finetune").write_text(content, encoding="utf-8")
                result = checker.audit(self.repo)
                self.assert_failed_check(result, "E2_finetune_original_log")
                json.dumps(result, allow_nan=False)

    def test_log_metrics_outside_rounding_bound_are_failed(self):
        self.add_synthetic_artifacts()
        self.logs["E4_augment"]["val"] = dict(self.logs["E4_augment"]["val"], mse=31.0)
        self.save_log("E4_augment")
        result = checker.audit(self.repo)
        self.assert_failed_check(result, "E4_augment_log_csv_metrics")
        self.assertEqual(result["comparison"]["mse_delta_e4_minus_e2"], 4.0)

    def test_paired_epoch_cap_patience_and_parameter_counts_must_match(self):
        self.add_synthetic_artifacts()
        for key, changed in (("epochs_cap", 16), ("patience", 5), ("trainable_params", 6)):
            with self.subTest(key=key):
                original = self.logs["E4_augment"][key]
                self.logs["E4_augment"][key] = changed
                self.save_log("E4_augment")
                result = checker.audit(self.repo)
                self.assert_failed_check(result, "matched_run_conditions" if key == "trainable_params" else "E4_augment_original_log")
                self.logs["E4_augment"][key] = original
                self.save_log("E4_augment")
        for tag in checker.TAGS:
            self.logs[tag]["epochs_cap"] = 20
            self.save_log(tag)
        result = checker.audit(self.repo)
        for tag in checker.TAGS:
            self.assert_failed_check(result, f"{tag}_original_log")

    def test_missing_or_nonfinite_required_csv_yields_clean_failed_report(self):
        path = self.csv_paths["E2_finetune"]
        original = path.read_bytes()
        path.unlink()
        result = checker.audit(self.repo)
        self.assert_failed_check(result, "csv_E2_finetune")
        self.assertIsNone(result["comparison"])
        path.write_bytes(original)
        self.rows["E2_finetune"][0][2] = "NaN"
        self.save_rows("E2_finetune")
        result = checker.audit(self.repo)
        self.assert_failed_check(result, "csv_E2_finetune")
        self.assertIsNone(result["comparison"])
        json.dumps(result, allow_nan=False)

    def test_cli_round_trip_require_artifacts_and_no_overwrite(self):
        out = self.root / "audit.json"
        args = ["--repo-root", str(self.repo), "--out", str(out), "--require-artifacts"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(checker.main(args), 2)
            original = out.read_bytes()
            self.assertEqual(checker.main(args), 1)
        self.assertEqual(out.read_bytes(), original)
        self.assertEqual(json.loads(original)["comparison"]["mse_delta_e4_minus_e2"], 4.0)
        self.assertIn("CSV checks passed; original evidence pending", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())

    def test_cli_cannot_write_to_present_or_missing_source_alias(self):
        before = self.train_path.read_bytes()
        for out in (self.train_path, self.log_path("E4_augment")):
            with self.subTest(out=out), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(["--repo-root", str(self.repo), "--out", str(out)]), 1)
        self.assertEqual(self.train_path.read_bytes(), before)
        self.assertFalse(self.log_path("E4_augment").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
