{
  "review_date": "2026-10-08",
  "repository": "mingyang-sun1/computer-vision-final-assignment",
  "commit": "8cb91dc12b29035eee70d3c3caf7690c7540399d",
  "review_scope": "Static source review plus CSV and B0 arithmetic. No visual training, no image reads, no Kaggle verification.",
  "acquisition": {
    "uploaded_csvs": [
      "train.csv",
      "test.csv",
      "sample_solution.csv"
    ],
    "split_method": "Reconstructed ONLY in an isolated audit workspace from uploaded train.csv with pandas.qcut(q=10, duplicates=drop) and train_test_split(test_size=0.125,random_state=42,stratify=bins); reset_index(drop=True); to_csv(index=False). Complete blob hashes then matched to GitHub. The distributed checker never reconstructs or writes splits.",
    "production_split_modified": false,
    "split_identity": "byte-for-byte match before any LF-normalised local comparison"
  },
  "source_blobs": {
    "config.py": "e5ccb24874258ed592a9fe6630911fef3ac22cf3",
    "src/baseline.py": "4a49350672a4cbff484f6f4210dd50fbf460cb5c",
    "src/features.py": "8654d75805a7fc1b8275e8ea2dbb7347f4e3cb78",
    "src/metrics.py": "0bc8341127b40ab2baa930d3705dcffdda81d668",
    "src/data.py": "8d8efb360fe831ced5facd1fae2d1a754ebbd38d",
    "docs/experiments.md": "c617874feb72e710fd83d3baf0cf3d408ade2f78",
    "docs/Phase1_Verification.md": "f021624466744daa1a862ff640a677b9f8b224cd",
    "data/splits/train_split.csv": "05851b5173130d45b78f1057ec7f0e78cf827a06",
    "data/splits/val_split.csv": "54c0018bb7fbb08c59dd26c511928836e5ea46ba",
    "outputs/predictions/submission_phase2_ridge.csv": "856a86573a39ee4c6e24b55dc046632420906f3d"
  },
  "current_output_tree": {
    "sha": "967351dfd8d5fe04f1bd514dfabfa0a227279ee6",
    "raw_phase2_log_present": false,
    "models_present": false,
    "ridge_submission_present": true,
    "ridge_submission_full_content_checked_in_this_workspace": false
  },
  "B0_float32_compatible_arithmetic": {
    "mean": 720.9805297851562,
    "mse": 154431.6921967651,
    "mae": 295.88285151672363,
    "r2": -4.5726209571661514e-05,
    "method": "Float32 labels and training mean, float64 metrics, matching the inspected expressions; not execution of src.baseline."
  },
  "historical_repository_record_not_new_experiments": {
    "B0_val_mse": 154431.7,
    "B1_val_mse": 114853.7,
    "B2_val_mse": 133378.6,
    "selected": "B1_ridge",
    "B1_kaggle_mse_reported": 114147.16,
    "B2_epochs_reported": 19,
    "ridge_alpha_original": null,
    "elapsed_training_time_original": null
  },
  "executed_checker_tests": {
    "tests": 17,
    "failures": 0,
    "errors": 0,
    "data": "synthetic fixtures only; dummy artifact files never loaded"
  },
  "environment_for_audit_only": {
    "python": "3.13.5",
    "numpy": "2.3.5",
    "pandas": "2.2.3"
  },
  "phase2_review_approval": "not supplied; do not infer from Phase 1 approval"
}
