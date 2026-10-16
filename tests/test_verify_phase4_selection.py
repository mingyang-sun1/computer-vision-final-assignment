"""Synthetic tests only. No course data, images, checkpoints or ML packages."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "tools/verify_phase4_selection.py"
spec = importlib.util.spec_from_file_location("phase4_checker", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Phase4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.write(m.TRAIN, ["imageid", "price"], [
            ["tr1.jpg", 200], ["tr2.jpg", 400], ["tr3.jpg", 600],
            ["tr4.jpg", 1000], ["tr5.jpg", 2000]])
        self.write(m.VAL, ["imageid", "price"], [["a.jpg", 300], ["b.jpg", 700], ["c.jpg", 1400]])
        values = ((310, 680, 1430), (320, 680, 1430), (280, 740, 1360))
        for tag, ps in zip(m.TAGS, values):
            self.write(m.PRED+f"val_predictions_{tag}.csv", ["imageid", "price_true", "price_pred"],
                       [[i, y, p] for i, y, p in zip(("a.jpg", "b.jpg", "c.jpg"), (300,700,1400), ps)])
            self.write(m.PRED+f"submission_phase3_{tag}.csv", ["imageid", "price"],
                       [[i, p] for i,p in zip(("x.jpg","y.jpg","z.jpg"), ps)])
        self.write(m.PRED+"submission_phase2_ridge.csv", ["imageid", "price"],
                   [["x.jpg",400],["y.jpg",800],["z.jpg",1000]])
        ens = m.ensemble([[Decimal(x) for x in ps] for ps in values])
        self.write(m.PRED+"submission_ensemble.csv", ["imageid", "price"],
                   [[i,p] for i,p in zip(("x.jpg","y.jpg","z.jpg"),ens)])
        self.write(m.PRED+"v3_val_predictions.csv", ["imageid","price_true","price_pred_224","price_pred_320"],
                   [["a.jpg",300,600,550],["b.jpg",700,900,1100],["c.jpg",1400,1100,1200]])
        (self.root/"src").mkdir()
        (self.root/"src/ensemble.py").write_text("# Synthetic identity fixture, not a production model.\n")
        self.refresh()

    def write(self, rel, header, rows):
        path=self.root/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(header); w.writerows(rows)

    def mutate(self, rel, fn):
        path=self.root/rel
        rows=list(csv.reader(io.StringIO(path.read_text())))
        fn(rows)
        self.write(rel, rows[0], rows[1:])
        self.refresh()

    def refresh(self):
        self.pins={p:m.blob_sha((self.root/p).read_bytes()) for p in m.PINNED_BLOBS}

    def run_audit(self):
        return m.audit(self.root,pins=self.pins,counts=(5,3,3))

    def failed(self):
        report=self.run_audit()
        self.assertGreater(report["summary"]["FAIL"],0)
        self.assertIsNone(report["candidate"])
        return report

    def test_success_is_provisional(self):
        r=self.run_audit()
        self.assertEqual(r["summary"]["FAIL"],0)
        self.assertEqual(r["candidate"]["id"],"E5_mean_export_1dp")
        self.assertEqual(r["candidate"]["status"],"PROVISIONAL")
        self.assertEqual(r["joint_approval"],"PENDING")
        self.assertFalse(r["test_scores_used_for_selection"])

    def test_source_bytes_unchanged(self):
        before={p:(self.root/p).read_bytes() for p in self.pins}
        self.run_audit()
        self.assertEqual(before,{p:(self.root/p).read_bytes() for p in self.pins})

    def test_metrics_known(self):
        r=m.metrics([Decimal(1),Decimal(3)],[Decimal(2),Decimal(2)])
        self.assertEqual(r,{"mse":1.0,"mae":1.0,"r2":0.0})

    def test_reordered_validation_is_aligned_by_id(self):
        before=self.run_audit()["metrics"]
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r.__setitem__(slice(1,None),r[1:][::-1]))
        self.assertEqual(before,self.run_audit()["metrics"])

    def test_duplicate_validation_id(self):
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r[2].__setitem__(0,"a.jpg")); self.failed()

    def test_missing_validation_id(self):
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r.pop()); self.failed()

    def test_extra_validation_id(self):
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r.append(["extra.jpg",300,300])); self.failed()

    def test_incorrect_label(self):
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r[1].__setitem__(1,"301")); self.failed()

    def test_nan_prediction(self):
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r[1].__setitem__(2,"NaN")); self.failed()

    def test_infinite_prediction(self):
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r[1].__setitem__(2,"Infinity")); self.failed()

    def test_out_of_range(self):
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r[1].__setitem__(2,"3000")); self.failed()

    def test_non_grid_prediction(self):
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r[1].__setitem__(2,"300.01")); self.failed()

    def test_padded_identifier(self):
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r[1].__setitem__(0," a.jpg")); self.failed()

    def test_duplicate_header(self):
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r[0].__setitem__(2,"price_true")); self.failed()

    def test_wrong_train_count(self):
        self.mutate(m.TRAIN,lambda r:r.pop()); self.failed()

    def test_split_overlap(self):
        self.mutate(m.TRAIN,lambda r:r[1].__setitem__(0,"a.jpg")); self.failed()

    def test_zero_label(self):
        self.mutate(m.TRAIN,lambda r:r[1].__setitem__(1,"0")); self.failed()

    def test_constant_labels_rejected(self):
        with self.assertRaises(m.EvidenceError):
            m.metrics([Decimal(1),Decimal(1)],[Decimal(2),Decimal(3)])

    def test_v3_columns_not_ignored(self):
        self.assertIn("V3_stored_224",self.run_audit()["metrics"])
        self.assertIn("V3_stored_320",self.run_audit()["metrics"])

    def test_better_unidentified_v3_blocks_recommendation(self):
        self.mutate(m.PRED+"v3_val_predictions.csv",lambda r:[row.__setitem__(2,row[1]) for row in r[1:]])
        r=self.run_audit()
        self.assertEqual(r["summary"]["FAIL"],0)
        self.assertIsNone(r["candidate"])
        self.assertTrue(any(c["name"]=="selection" and c["status"]=="PENDING" for c in r["checks"]))

    def test_reordered_test_fails_existing_pipeline_contract(self):
        self.mutate(m.PRED+"submission_phase3_E3_320px.csv",lambda r:r.__setitem__(slice(1,None),r[1:][::-1])); self.failed()

    def test_wrong_ensemble_arithmetic(self):
        self.mutate(m.PRED+"submission_ensemble.csv",lambda r:r[1].__setitem__(1,"999.9")); self.failed()

    def test_submission_nonfinite(self):
        self.mutate(m.PRED+"submission_ensemble.csv",lambda r:r[1].__setitem__(1,"NaN")); self.failed()

    def test_missing_file_fails_before_selection(self):
        (self.root/m.VAL).unlink(); self.failed()

    def test_identity_drift_fails(self):
        path=self.root/m.VAL
        path.write_bytes(path.read_bytes()+b"\n")
        self.failed()

    def test_crlf_lf_same_identity(self):
        for rel in self.pins:
            p=self.root/rel
            p.write_bytes(p.read_bytes().replace(b"\r\n",b"\n").replace(b"\n",b"\r\n"))
        self.assertEqual(self.run_audit()["summary"]["FAIL"],0)

    def test_original_logs_only_inventory(self):
        p=self.root/"outputs/logs/phase3_E3_320px.json"
        p.parent.mkdir(parents=True); p.write_text('{"seed":42}')
        r=self.run_audit()
        row=next(x for x in r["artifacts"] if x["path"].endswith("phase3_E3_320px.json"))
        self.assertTrue(row["present"])
        self.assertFalse(row["loaded_or_provenance_verified"])
        self.assertGreater(r["summary"]["PENDING"],0)

    def test_output_folder_is_new(self):
        report=self.run_audit(); out=self.root/"docs/evidence/run1"
        m.write_report(report,self.root,out)
        with self.assertRaises(FileExistsError): m.write_report(report,self.root,out)
        self.assertTrue((out/"validation_mse.svg").is_file())
        self.assertEqual(json.loads((out/"selection_candidate.json").read_text())["status"],"PROVISIONAL")

    def test_output_outside_evidence_rejected(self):
        with self.assertRaises(m.EvidenceError):
            m.write_report(self.run_audit(),self.root,self.root/"outputs/new")

    def test_failed_audit_has_no_success_chart(self):
        (self.root/m.VAL).unlink(); r=self.failed(); out=self.root/"docs/evidence/failed"
        m.write_report(r,self.root,out)
        self.assertTrue((out/"audit.json").is_file())
        self.assertFalse((out/"validation_mse.svg").exists())
        self.assertFalse((out/"selection_candidate.json").exists())

    def test_three_component_mean_only(self):
        with self.assertRaises(m.EvidenceError): m.ensemble([[Decimal(1)],[Decimal(2)]])

    def test_rounding_policy(self):
        self.assertEqual(m.ensemble([[Decimal("300.1")],[Decimal("300.2")],[Decimal("300.2")]]),[Decimal("300.2")])


    # Additional regression tests for the 16 October revision.

    def test_supplied_manifest_matches(self):
        manifest = self.root / "course_test.csv"
        manifest.write_text("imageid\nx.jpg\ny.jpg\nz.jpg\n", encoding="utf-8")
        r = m.audit(self.root, pins=self.pins, counts=(5,3,3), test_manifest=manifest)
        self.assertEqual(r["summary"]["FAIL"], 0)
        self.assertEqual(next(x["status"] for x in r["checks"] if x["name"]=="test_manifest"), "PASS")
        self.assertFalse(r["external_test_manifest"]["labels_read"])

    def test_wrong_manifest_order_fails(self):
        manifest = self.root / "course_test.csv"
        manifest.write_text("imageid\nz.jpg\ny.jpg\nx.jpg\n", encoding="utf-8")
        r = m.audit(self.root, pins=self.pins, counts=(5,3,3), test_manifest=manifest)
        self.assertGreater(r["summary"]["FAIL"], 0)
        self.assertIsNone(r["candidate"])

    def test_manifest_with_labels_rejected(self):
        manifest = self.root / "course_test.csv"
        manifest.write_text("imageid,price\nx.jpg,500\ny.jpg,600\nz.jpg,700\n", encoding="utf-8")
        r = m.audit(self.root, pins=self.pins, counts=(5,3,3), test_manifest=manifest)
        self.assertGreater(r["summary"]["FAIL"], 0)
        self.assertFalse(r["test_scores_used_for_selection"])

    def test_missing_explicit_manifest_fails(self):
        r = m.audit(self.root, pins=self.pins, counts=(5,3,3), test_manifest=self.root/"absent.csv")
        self.assertGreater(r["summary"]["FAIL"], 0)

    def test_huge_numeric_value_fails_cleanly(self):
        self.mutate(m.PRED+"val_predictions_E3_320px.csv",lambda r:r[1].__setitem__(2,"1e999999"))
        self.failed()

    def test_nonfinite_metrics_rejected(self):
        with self.assertRaises(m.EvidenceError):
            m.metrics([Decimal("1"),Decimal("2")],[Decimal("NaN"),Decimal("3")])

    def test_pairwise_counts_cover_validation_population(self):
        report = self.run_audit()
        for comparison in report["paired_comparisons"].values():
            self.assertEqual(sum(comparison[k] for k in
                ("lower_absolute_error", "equal_absolute_error", "higher_absolute_error")), 3)

    def test_historical_references_not_marked_recomputed(self):
        report = self.run_audit()
        self.assertFalse(report["reported_references_recomputed"])
        self.assertEqual(report["reported_reference_mse"]["B1"], 114853.7)
        self.assertNotIn("B1", report["metrics"])

    def test_no_automatic_approval_or_date(self):
        candidate = self.run_audit()["candidate"]
        self.assertEqual(candidate["approved_by"], [])
        self.assertIsNone(candidate["review_date"])

    def test_existing_complete_outputs_are_not_overwritten(self):
        report = self.run_audit()
        out = self.root / "docs/evidence/once"
        m.write_report(report, self.root, out)
        before = {f.name:f.read_bytes() for f in out.iterdir()}
        with self.assertRaises(FileExistsError):
            m.write_report(report, self.root, out)
        self.assertEqual(before,{f.name:f.read_bytes() for f in out.iterdir()})

    def test_reference_numbers_not_used_as_test_labels(self):
        report = self.run_audit()
        self.assertFalse(report["kaggle_score_verified"])
        self.assertFalse(report["test_scores_used_for_selection"])
        self.assertEqual(report["checker_version"], "2.0.0")


if __name__ == "__main__":
    unittest.main()
