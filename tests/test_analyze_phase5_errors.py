"""Synthetic correctness checks for the Member A Phase 5 CSV analysis.

No course images, model checkpoints, network access, or ML runtime are used.
The small fixtures have independently calculable errors and deliberately
different CSV orderings, so a positional join cannot accidentally pass.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import math
import tempfile
import unittest
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools/analyze_phase5_errors.py"
SPEC = importlib.util.spec_from_file_location("phase5_analysis", SCRIPT)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class Phase5AnalysisTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.ids = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"]
        self.truth = [100, 200, 300, 400, 500, 600]
        self.predictions = [110, 160, 300, 430, 450, 670]
        self.write(m.TRAIN, ["imageid", "price"], [
            ["train_a.jpg", 50], ["train_b.jpg", 250],
            ["train_c.jpg", 450], ["train_d.jpg", 650],
            ["train_e.jpg", 800],
        ])
        self.write(m.VAL, ["imageid", "price"], list(zip(self.ids, self.truth)))
        for tag, offset in zip(m.TAGS, (-10, 0, 10)):
            self.write(self.component(tag), ["imageid", "price_true", "price_pred"], [
                [ident, truth, prediction + offset]
                for ident, truth, prediction in zip(self.ids, self.truth, self.predictions)
            ])
        self.write(m.ENSEMBLE_PATH, ["imageid", "price_true", "price_pred"],
                   list(zip(self.ids, self.truth, self.predictions)))
        self.refresh_pins()

    @staticmethod
    def component(tag):
        return m.PRED + f"val_predictions_{tag}.csv"

    def write(self, relative, header, rows):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def refresh_pins(self):
        self.pins = {
            relative: m.blob_sha((self.root / relative).read_bytes())
            for relative in m.PINNED_BLOBS
        }

    def mutate(self, relative, transform):
        rows = list(csv.reader(io.StringIO((self.root / relative).read_text(encoding="utf-8"))))
        transform(rows)
        self.write(relative, rows[0], rows[1:])
        self.refresh_pins()

    @contextmanager
    def changed(self, relative, transform):
        path = self.root / relative
        original = path.read_bytes()
        self.mutate(relative, transform)
        try:
            yield
        finally:
            path.write_bytes(original)
            self.refresh_pins()

    def analyze(self, buckets=3):
        return m.analyze(self.root, pins=self.pins, counts=(5, 6), buckets=buckets)

    def test_known_signed_errors_and_metrics(self):
        report = self.analyze()
        metric = report["overall"]
        # Residuals are [10, -40, 0, 30, -50, 70], with SSE = 10,000.
        self.assertEqual(metric["count"], 6)
        self.assertEqual(metric["sse"], 10000)
        self.assertAlmostEqual(metric["mse"], 10000 / 6)
        self.assertAlmostEqual(metric["rmse"], math.sqrt(10000 / 6))
        self.assertAlmostEqual(metric["mae"], 200 / 6)
        self.assertAlmostEqual(metric["bias"], 20 / 6)
        self.assertAlmostEqual(metric["r2"], 1 - 10000 / 175000)
        self.assertEqual(metric["overprediction_fraction"], 0.5)
        self.assertAlmostEqual(metric["underprediction_fraction"], 2 / 6)
        self.assertAlmostEqual(metric["exact_prediction_fraction"], 1 / 6)
        errors = {row["imageid"]: row for row in report["per_image_errors"]}
        self.assertEqual(errors["b.jpg"]["residual"], -40)
        self.assertEqual(errors["b.jpg"]["abs_error"], 40)
        self.assertEqual(errors["b.jpg"]["squared_error"], 1600)
        self.assertEqual(errors["f.jpg"]["residual"], 70)

    def test_each_prediction_file_is_joined_by_identifier(self):
        original = self.analyze()
        for index, tag in enumerate(m.TAGS):
            def reorder(rows, index=index):
                body = rows[1:]
                rows[1:] = body[index + 1:] + body[:index + 1]
            self.mutate(self.component(tag), reorder)
        self.mutate(m.ENSEMBLE_PATH,
                    lambda rows: rows.__setitem__(slice(1, None), rows[1:][::-1]))
        reordered = self.analyze()
        for key in ("overall", "price_buckets", "per_image_errors", "representative_examples"):
            with self.subTest(result=key):
                self.assertEqual(original[key], reordered[key])

    def test_bucket_totals_reconstruct_overall_errors(self):
        report = self.analyze()
        buckets = report["price_buckets"]
        overall = report["overall"]
        self.assertEqual(sum(row["count"] for row in buckets), overall["count"])
        self.assertAlmostEqual(sum(row["sse"] for row in buckets), overall["sse"])
        self.assertAlmostEqual(sum(row["sse_share"] for row in buckets), 1)
        self.assertAlmostEqual(sum(row["count"] * row["mse"] for row in buckets) / 6,
                               overall["mse"])
        self.assertAlmostEqual(sum(row["count"] * row["bias"] for row in buckets) / 6,
                               overall["bias"])

    def test_paired_comparison_counts_absolute_errors_and_reconciles_sse(self):
        # E3 has two smaller, two equal, and two larger absolute errors.
        # The equal pair includes opposite signed residuals of -30 and +30.
        e3_predictions = [120, 170, 300, 370, 400, 650]
        other_predictions = [(3 * e5 - e3) / 2
                             for e5, e3 in zip(self.predictions, e3_predictions)]
        for tag in m.TAGS:
            predictions = e3_predictions if tag == "E3_320px" else other_predictions
            self.write(self.component(tag), ["imageid", "price_true", "price_pred"],
                       list(zip(self.ids, self.truth, predictions)))
        self.refresh_pins()
        paired = self.analyze()["paired_e3_comparison"]
        self.assertEqual(paired["lower_absolute_error"], 2)
        self.assertEqual(paired["equal_absolute_error"], 2)
        self.assertEqual(paired["higher_absolute_error"], 2)
        self.assertEqual(paired["sse_reduction_from_improved_rows"], 7800)
        self.assertEqual(paired["sse_increase_from_worsened_rows"], 3100)
        self.assertEqual(paired["net_sse_change"], -4700)
        self.assertAlmostEqual(paired["net_mse_change"], -4700 / 6)
        self.assertAlmostEqual(sum(row["mse_change"] * row["count"]
                                   for row in paired["price_buckets"]), -4700)

    def assert_partition(self, labels, predictions, number_of_buckets):
        buckets = m.bucket_table(labels, predictions, n=number_of_buckets)
        self.assertEqual(sum(row["count"] for row in buckets), len(labels))
        for label in labels:
            matches = [row for row in buckets
                       if Decimal(str(row["price_lo"])) <= label
                       and (label <= Decimal(str(row["price_hi"]))
                            if row["upper_inclusive"]
                            else label < Decimal(str(row["price_hi"])))]
            self.assertEqual(len(matches), 1, f"Price {label} must belong to exactly one bucket")
        expected_sse = sum((prediction - label) ** 2 for label, prediction in zip(labels, predictions))
        self.assertAlmostEqual(sum(row["sse"] for row in buckets), float(expected_sse))
        self.assertAlmostEqual(sum(row["sse_share"] for row in buckets), 1)
        return buckets

    def test_tied_quantiles_cover_all_prices_once(self):
        labels = [Decimal(value) for value in (100, 100, 100, 200, 200, 300, 300, 300, 400, 400)]
        predictions = [value + Decimal(index - 5) for index, value in enumerate(labels)]
        self.assert_partition(labels, predictions, 10)

    def test_constant_prices_form_one_valid_bucket(self):
        labels = [Decimal(200)] * 6
        predictions = [Decimal(value) for value in (210, 160, 200, 230, 150, 270)]
        buckets = self.assert_partition(labels, predictions, 10)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0]["count"], 6)
        self.assertIsNone(buckets[0]["r2"])

    def test_missing_duplicate_and_extra_prediction_ids_are_rejected(self):
        transforms = {
            "missing": lambda rows: rows.pop(),
            "duplicate": lambda rows: rows[2].__setitem__(0, rows[1][0]),
            "extra": lambda rows: rows.append(["extra.jpg", 300, 310]),
        }
        for source in (m.ENSEMBLE_PATH, self.component(m.TAGS[1])):
            for defect, transform in transforms.items():
                with self.subTest(source=source, defect=defect):
                    with self.changed(source, transform):
                        with self.assertRaises(m.EvidenceError):
                            self.analyze()

    def test_wrong_ground_truth_is_rejected_in_ensemble_and_components(self):
        for source in (m.ENSEMBLE_PATH, self.component(m.TAGS[1])):
            with self.subTest(source=source):
                with self.changed(source, lambda rows: rows[1].__setitem__(1, "101")):
                    with self.assertRaises(m.EvidenceError):
                        self.analyze()

    def test_nonfinite_predictions_and_labels_are_rejected(self):
        columns = ((m.TRAIN, 1), (m.VAL, 1), (m.ENSEMBLE_PATH, 1),
                   (m.ENSEMBLE_PATH, 2), (self.component(m.TAGS[1]), 2))
        for source, column in columns:
            for value in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(source=source, column=column, value=value):
                    with self.changed(source, lambda rows, column=column, value=value:
                                      rows[1].__setitem__(column, value)):
                        with self.assertRaises(m.EvidenceError):
                            self.analyze()

    def test_training_validation_overlap_is_rejected(self):
        with self.changed(m.TRAIN, lambda rows: rows[1].__setitem__(0, "a.jpg")):
            with self.assertRaises(m.EvidenceError):
                self.analyze()

    def test_changed_source_identity_is_rejected(self):
        path = self.root / m.ENSEMBLE_PATH
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(m.EvidenceError):
            self.analyze()

    def test_finite_ensemble_prediction_must_equal_rounded_component_mean(self):
        with self.changed(m.ENSEMBLE_PATH, lambda rows: rows[1].__setitem__(2, "110.1")):
            with self.assertRaises(m.EvidenceError):
                self.analyze()

    def test_component_mean_is_compared_at_export_precision(self):
        for tag, value in zip(m.TAGS, ("110.1", "110.2", "110.2")):
            self.mutate(self.component(tag),
                        lambda rows, value=value: rows[1].__setitem__(2, value))
        self.mutate(m.ENSEMBLE_PATH, lambda rows: rows[1].__setitem__(2, "110.2"))
        report = self.analyze()
        actual = next(row for row in report["per_image_errors"] if row["imageid"] == "a.jpg")
        self.assertEqual(actual["price_pred"], 110.2)
        self.assertAlmostEqual(actual["residual"], 10.2)
        with self.changed(m.ENSEMBLE_PATH,
                          lambda rows: rows[1].__setitem__(2, "110.1666666666666667")):
            with self.assertRaises(m.EvidenceError):
                self.analyze()

    def test_analysis_and_new_report_preserve_every_input_byte(self):
        original = {relative: (self.root / relative).read_bytes() for relative in self.pins}
        report = self.analyze()
        destination = self.root / "docs/evidence/phase5_new_run"
        m.write_report(report, self.root, destination)
        self.assertTrue(destination.is_dir())
        self.assertTrue(any(destination.iterdir()))
        self.assertEqual(original, {
            relative: (self.root / relative).read_bytes() for relative in self.pins
        })

    def test_existing_output_directory_is_not_overwritten(self):
        destination = self.root / "docs/evidence/phase5_existing"
        destination.mkdir(parents=True)
        sentinel = destination / "existing_result.txt"
        sentinel.write_text("Preserve this result.\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            m.write_report(self.analyze(), self.root, destination)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "Preserve this result.\n")
        self.assertEqual(list(destination.iterdir()), [sentinel])

    def test_source_evidence_directory_cannot_be_reused_for_output(self):
        source = self.root / m.ENSEMBLE_PATH
        original = source.read_bytes()
        with self.assertRaises(m.EvidenceError):
            m.write_report(self.analyze(), self.root, source.parent)
        self.assertEqual(source.read_bytes(), original)

    def test_new_output_outside_evidence_directory_is_rejected(self):
        destination = self.root / "outputs/unexpected"
        with self.assertRaises(m.EvidenceError):
            m.write_report(self.analyze(), self.root, destination)
        self.assertFalse(destination.exists())

    @staticmethod
    def example_row(imageid, residual):
        return {"imageid": imageid, "price_true": 100.0,
                "price_pred": 100.0 + residual, "residual": residual,
                "abs_error": abs(residual), "squared_error": residual ** 2,
                "relative_error": residual / 100, "abs_relative_error": abs(residual) / 100,
                "price_bucket": "B01"}

    def test_examples_are_deterministic_directional_and_disjoint(self):
        values = (("aa.jpg", 0), ("ab.jpg", 0.5), ("ac.jpg", -0.5),
                  ("p1.jpg", 10), ("p0.jpg", 10), ("p2.jpg", 8), ("p3.jpg", 7),
                  ("n1.jpg", -10), ("n0.jpg", -10), ("n2.jpg", -8), ("n3.jpg", -7),
                  ("zz.jpg", 2))
        rows = [self.example_row(ident, residual) for ident, residual in values]
        chosen = m.select_examples(rows, per_category=3)
        self.assertEqual(chosen, m.select_examples(rows[::-1], per_category=3))
        self.assertEqual(len(chosen), 9)
        self.assertEqual(len({row["imageid"] for row in chosen}), 9)
        by_category = {}
        for row in chosen:
            by_category.setdefault(row["category"], []).append(row)
        self.assertEqual([row["imageid"] for row in by_category["best_absolute"]],
                         ["aa.jpg", "ab.jpg", "ac.jpg"])
        self.assertEqual([row["imageid"] for row in by_category["largest_overprediction"]],
                         ["p0.jpg", "p1.jpg", "p2.jpg"])
        self.assertEqual([row["imageid"] for row in by_category["largest_underprediction"]],
                         ["n0.jpg", "n1.jpg", "n2.jpg"])
        self.assertTrue(all(row["residual"] > 0 for row in by_category["largest_overprediction"]))
        self.assertTrue(all(row["residual"] < 0 for row in by_category["largest_underprediction"]))

    def test_small_example_sets_do_not_duplicate_images_to_fill_categories(self):
        rows = [self.example_row("a.jpg", 0), self.example_row("b.jpg", 2)]
        selected = m.select_examples(rows, per_category=3)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({row["imageid"] for row in selected}), 2)


if __name__ == "__main__":
    unittest.main()
