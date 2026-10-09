"""Synthetic tests of the evidence checker, NOT model-training tests."""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/verify_phase2_baseline.py"
spec = importlib.util.spec_from_file_location("phase2_checker", MODULE_PATH)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class CheckerUnitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def table(self, name, columns, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def test_metrics_known_values(self):
        result = checker.metric_summary([1.0, 3.0], [2.0, 2.0])
        self.assertEqual(result, {"mse": 1.0, "mae": 1.0, "r2": 0.0})

    def test_b0_uses_training_mean_not_validation_mean(self):
        result = checker.b0_reference([1.0, 3.0], [10.0, 12.0])
        self.assertEqual(result["training_mean"], 2.0)
        self.assertEqual(result["validation"]["mse"], 82.0)

    def test_bad_metric_lengths_rejected(self):
        with self.assertRaises(ValueError):
            checker.metric_summary([1.0], [1.0, 2.0])

    def test_constant_target_has_no_defined_r2(self):
        self.assertIsNone(checker.metric_summary([1.0, 1.0], [2.0, 2.0])["r2"])

    def test_duplicate_ids_rejected(self):
        path = self.table("duplicate.csv", ["imageid", "price"], [["1.jpg", 1], ["1.jpg", 2]])
        with self.assertRaises(ValueError):
            checker.read_table(path, ["imageid", "price"])

    def test_wrong_schema_rejected(self):
        path = self.table("schema.csv", ["imageid", "value"], [["1.jpg", 1]])
        with self.assertRaises(ValueError):
            checker.read_table(path, ["imageid", "price"])

    def test_nonfinite_prices_rejected(self):
        for value in ("NaN", "inf", "-inf"):
            with self.subTest(value=value):
                path = self.table("bad.csv", ["imageid", "price"], [["1.jpg", value]])
                with self.assertRaises(ValueError):
                    checker.read_table(path, ["imageid", "price"])

    def test_malformed_row_width_rejected(self):
        path = self.table("wide.csv", ["imageid", "price"], [["1.jpg", 1, "extra"]])
        with self.assertRaises(ValueError):
            checker.read_table(path, ["imageid", "price"])

    def test_crlf_identity_equivalent_but_exact_hash_differs(self):
        a, b = self.root / "lf.csv", self.root / "crlf.csv"
        a.write_bytes(b"imageid,price\n1.jpg,1\n")
        b.write_bytes(b"imageid,price\r\n1.jpg,1\r\n")
        self.assertEqual(checker.file_identity(a)["git_blob_sha1_lf"], checker.file_identity(b)["git_blob_sha1_lf"])
        self.assertNotEqual(checker.file_identity(a)["sha256"], checker.file_identity(b)["sha256"])

    def test_refuse_output_overwrite(self):
        path = self.root / "audit.json"
        checker.write_new_json(path, {"first": 1})
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            checker.write_new_json(path, {"second": 2})
        self.assertEqual(path.read_bytes(), original)

    def test_nonfinite_json_rejected(self):
        path = self.root / "log.json"
        path.write_text('{"value":NaN}', encoding="utf-8")
        with self.assertRaises(ValueError):
            checker.load_object(path)


class SyntheticIntegrationTests(unittest.TestCase):
    """8,000 synthetic labels; dummy model files are never loaded."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo, self.data = self.root / "repo", self.root / "dataset"
        train = [[f"{i}.jpg", 200.0 + (i % 100)] for i in range(1, 7001)]
        val = [[f"{i}.jpg", 200.0 + (i % 100)] for i in range(7001, 8001)]
        test = [[f"{i}.jpg"] for i in range(8001, 11001)]
        self.write(self.repo / "data/splits/train_split.csv", ["imageid", "price"], train)
        self.write(self.repo / "data/splits/val_split.csv", ["imageid", "price"], val)
        self.write(self.data / "train.csv", ["imageid", "price"], train + val)
        self.write(self.data / "test.csv", ["imageid"], test)
        self.write(self.data / "sample_solution.csv", ["imageid", "price"], [[r[0], 500] for r in test])
        self.submission = self.repo / "outputs/predictions/submission_phase2_ridge.csv"
        self.write(self.submission, ["imageid", "price"], [[r[0], 249.5] for r in test])
        expected = {p.name: checker.file_identity(p)["git_blob_sha1_lf"] for p in (self.repo / "data/splits").glob("*.csv")}
        self.ref = checker.b0_reference([r[1] for r in train], [r[1] for r in val])
        self.addCleanup(patch.stopall)
        patch.dict(checker.EXPECTED_SPLITS, expected).start()
        patch.dict(checker.REPORTED, {"B0_mean": self.ref["validation"]}).start()
        self.stats = {"mean": self.ref["training_mean"], "std": self.ref["training_std_population"], "min": 200.0, "max": 299.0}
        self.log = {"phase": 2, "seed": 42, "n_train": 7000, "n_val": 1000, "feature_dim": 2048,
                    "target_stats": self.stats, "B0_mean": self.ref["validation"],
                    "B1_ridge": {"mse": 100.0, "mae": 8.0, "r2": 0.4, "alpha": 10.0},
                    "B2_mlp": {"mse": 120.0, "mae": 9.0, "r2": 0.3, "hidden": 256, "dropout": 0.2, "learning_rate": 0.001, "epochs_run": 19},
                    "selected": "B1_ridge"}
        self.log_path = self.repo / "outputs/logs/phase2_results.json"

    @staticmethod
    def write(path, columns, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            w = csv.writer(stream); w.writerow(columns); w.writerows(rows)

    def add_artifacts(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(self.log), encoding="utf-8")
        model_dir = self.repo / "outputs/models"; model_dir.mkdir(parents=True, exist_ok=True)
        for name in ("x_scaler.joblib", "ridge.joblib", "mlp.keras"):
            (model_dir / name).write_bytes(b"SYNTHETIC FILE-PRESENCE FIXTURE; NOT A MODEL")
        (model_dir / "target_stats.json").write_text(json.dumps(self.stats), encoding="utf-8")

    def test_missing_artifacts_are_pending_not_pass(self):
        result = checker.audit(self.repo, self.data)
        self.assertEqual(result["summary"]["fail"], 0)
        self.assertEqual(result["summary"]["pending"], 5)
        self.assertFalse(result["training_executed"])

    def test_full_artifact_checks_do_not_claim_training(self):
        self.add_artifacts()
        before = {p: p.read_bytes() for p in (self.repo / "data/splits").glob("*.csv")}
        result = checker.audit(self.repo, self.data)
        self.assertEqual(result["summary"]["fail"], 0)
        self.assertEqual(result["summary"]["pending"], 0)
        self.assertIn("NOT_A_TRAINING_REPRODUCTION", result["status"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_wrong_winner_is_failed(self):
        self.log["selected"] = "B2_mlp"; self.add_artifacts()
        result = checker.audit(self.repo, self.data)
        self.assertTrue(any(c["name"] == "selection_rule" and c["status"] == "FAIL" for c in result["checks"]))

    def test_mismatched_target_statistics_are_failed(self):
        self.stats["mean"] += 5; self.add_artifacts()
        result = checker.audit(self.repo, self.data)
        self.assertTrue(any(c["name"] == "log_target_stats" and c["status"] == "FAIL" for c in result["checks"]))

    def test_bad_submission_order_is_failed(self):
        rows = [[f"{i}.jpg", 249.5] for i in range(11000, 8000, -1)]
        self.write(self.submission, ["imageid", "price"], rows)
        result = checker.audit(self.repo, self.data)
        self.assertTrue(any(c["name"] == "submission_ids_and_order" and c["status"] == "FAIL" for c in result["checks"]))

    def test_require_artifacts_returns_two_and_does_not_overwrite(self):
        out = self.root / "audit.json"
        args = ["--repo-root", str(self.repo), "--data-root", str(self.data), "--out", str(out), "--require-artifacts"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main(args), 2)
            raw = out.read_bytes()
            self.assertEqual(checker.main(args), 1)
            self.assertEqual(raw, out.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
