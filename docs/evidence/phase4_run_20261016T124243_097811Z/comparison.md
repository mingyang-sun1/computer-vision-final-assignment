# Phase 4: executed CSV comparison

Reference snapshot: `90959c41384c7bd3c8c5a48147e8272246071bbe`.
Audit status: **PROVISIONAL_REVIEW_REQUIRED**. Joint approval: **PENDING**.

Scores below are recomputed from stored CSVs, not new training results. MSE is in squared
USD-1,000 units; MAE is in USD 1,000; R2 is unitless. No test score is used for selection.

| Candidate / reference | Validation MSE | MAE | R2 |
|---|---:|---:|---:|
| E2_finetune | 102,746.219475 | 235.650414 | 0.334651 |
| E3_320px | 99,051.197058 | 229.764228 | 0.358579 |
| E4_augment | 108,784.236328 | 246.318624 | 0.295551 |
| B0_mean | 154,431.691512 | 295.882842 | -0.000046 |
| E5_mean_unrounded | 95,850.620431 | 226.420787 | 0.379305 |
| E5_mean_export_1dp | 95,852.002676 | 226.423020 | 0.379296 |
| V3_stored_224 | 110,013.040666 | 247.096194 | 0.287594 |
| V3_stored_320 | 109,155.385126 | 241.960203 | 0.293148 |

The unrounded E5 row is a precision reference. Selection uses E5_mean_export_1dp,
which matches the existing submission mean-then-round policy. V3 columns are diagnostic
until their original model/run identity is confirmed. Historical B1/B2/E1 scores are in
`docs/tables/Phase4_Historical_Comparison.md`; they are not silently treated as recomputed.

## Candidate

```json
{
  "status": "PROVISIONAL",
  "id": "E5_mean_export_1dp",
  "selection_metric": "validation MSE from the pinned rounded CSV evidence",
  "validation_mse": 95852.002675622,
  "scope": [
    "E2_finetune",
    "E3_320px",
    "E4_augment",
    "E5_mean_export_1dp"
  ],
  "review_date": null,
  "approved_by": [],
  "weights": [
    "1/3",
    "1/3",
    "1/3"
  ],
  "source_tags": [
    "E2_finetune",
    "E3_320px",
    "E4_augment"
  ],
  "joint_approval": "PENDING",
  "checkpoint_provenance_verified": false
}
```

## Checks

- **PASS** `identity:data/splits/train_split.csv`: LF-normalized identity matches inspected snapshot
- **PASS** `identity:data/splits/val_split.csv`: LF-normalized identity matches inspected snapshot
- **PASS** `identity:outputs/predictions/val_predictions_E2_finetune.csv`: LF-normalized identity matches inspected snapshot
- **PASS** `identity:outputs/predictions/val_predictions_E3_320px.csv`: LF-normalized identity matches inspected snapshot
- **PASS** `identity:outputs/predictions/val_predictions_E4_augment.csv`: LF-normalized identity matches inspected snapshot
- **PASS** `identity:outputs/predictions/submission_phase2_ridge.csv`: LF-normalized identity matches inspected snapshot
- **PASS** `identity:outputs/predictions/submission_phase3_E2_finetune.csv`: LF-normalized identity matches inspected snapshot
- **PASS** `identity:outputs/predictions/submission_phase3_E3_320px.csv`: LF-normalized identity matches inspected snapshot
- **PASS** `identity:outputs/predictions/submission_phase3_E4_augment.csv`: LF-normalized identity matches inspected snapshot
- **PASS** `identity:outputs/predictions/submission_ensemble.csv`: LF-normalized identity matches inspected snapshot
- **PASS** `identity:outputs/predictions/v3_val_predictions.csv`: LF-normalized identity matches inspected snapshot
- **PASS** `identity:src/ensemble.py`: LF-normalized identity matches inspected snapshot
- **PASS** `fixed_split`: 7000 training / 1000 validation; disjoint IDs
- **PASS** `validation:E2_finetune`: Unique IDs aligned by imageid; labels, range and export grid checked
- **PASS** `validation:E3_320px`: Unique IDs aligned by imageid; labels, range and export grid checked
- **PASS** `validation:E4_augment`: Unique IDs aligned by imageid; labels, range and export grid checked
- **PASS** `v3_validation`: Both existing columns checked; original run identity remains unconfirmed
- **PASS** `submission_arithmetic`: 3000 rows match three-component mean rounded to 1 dp
- **PENDING** `test_manifest`: Optional course test.csv not supplied. The stored submissions agree internally; external manifest verification is deferred.
- **PASS** `selection`: Numerical candidate only: E5_mean_export_1dp; not final approval
- **PASS** `inventory:outputs/logs/phase3_E2_finetune.json`: Nonempty local file only; not executed or matched to predictions
- **PASS** `inventory:outputs/models/finetuned_E2_finetune.keras`: Nonempty local file only; not executed or matched to predictions
- **PASS** `inventory:outputs/logs/phase3_E3_320px.json`: Nonempty local file only; not executed or matched to predictions
- **PASS** `inventory:outputs/models/finetuned_E3_320px.keras`: Nonempty local file only; not executed or matched to predictions
- **PASS** `inventory:outputs/logs/phase3_E4_augment.json`: Nonempty local file only; not executed or matched to predictions
- **PASS** `inventory:outputs/models/finetuned_E4_augment.keras`: Nonempty local file only; not executed or matched to predictions
- **PENDING** `original_artifacts`: Phase 7 reproducibility follow-up. E2/E4 receipt is already recorded in Phase3_MemberB_Artifact_Update.md. Do not request those files again solely because they are absent from GitHub. Inventory is not checkpoint-to-prediction provenance.
- **PENDING** `v3_run_identity`: Supplementary V3 comparison requires its original run identity. Do not infer backbone or preprocessing from the column names alone.
- **PENDING** `joint_review`: Member A and Member B must approve the candidate and record actual review date
