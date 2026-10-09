# Phase 2 - Baseline Model Record

**Course:** COMP90086 Computer Vision, 2026 Semester 2  
**Project:** Predicting House Prices  
**Member A / document owner:** Mingyang Sun (1657392)  
**Member B / reviewer:** Yikai Qian (1722421)  
**Prepared:** 8 October 2026 | **Internal phase deadline:** 9 October 2026  
**Repository reviewed:** `main @ 8cb91dc12b29035eee70d3c3caf7690c7540399d`

**Status:** Baseline documentation and CSV-level verification prepared. The visual baseline already exists in the repository. Its original run artifacts and Member B's Phase 2 review remain to be checked. This is not a new model-training result or a claim that all Phase 2 acceptance criteria have passed.

## 1. Scope and responsibility

Member A's Phase 2 task is a regression baseline using frozen, general-purpose pretrained visual features, with its configuration and validation MSE recorded. Member B supports the training/evaluation workflow, model saving and prediction export. This supplement reviews the existing B0/B1/B2 implementation; it does not replace it, claim sole authorship of it, or change the completed Phase 1 files. Phase 3 augmentation and fine-tuning experiments are outside this handoff. [R1-R4]

Phase 1's review is already recorded in `docs/Phase1_Verification.md`. The current code protects existing split files and checks feature-cache image IDs in order. These changes are retained. [R2-R3]

## 2. Fixed experimental setup

| Item | Inspected implementation / configuration |
|---|---|
| Training / validation | 7,000 / 1,000 rows from the provided 8,000 labelled examples |
| Split policy | Existing committed split, price-decile stratification; validation fraction 0.125 |
| Seed | 42 |
| Target unit | USD 1,000; output is a scalar house price |
| Primary metric | Validation MSE; MAE and R2 are supporting metrics |
| Backbone | ImageNet-pretrained ResNet-50, `include_top=False`, global average pooling |
| Frozen features | Backbone has `trainable=False`; one 2,048-dimensional vector per image |
| Input processing | Decode JPEG as 3 channels; square resize to 224 x 224; `resnet50.preprocess_input` |
| Feature-extraction batch | 32 |
| Feature scaling | `StandardScaler` fitted on the training features, then applied unchanged to validation |
| Target scaling | Training mean and population standard deviation; reverse the transform before scoring |
| Augmentation | None in the Phase 2 baseline path |
| Post-processing | Clip B1/B2 predictions to training-label minimum and maximum, 195 to 2,000 |
| Selection | Choose B1 or B2 with lower validation MSE; the current code uses B1 on a tie |

These settings describe the inspected code, not a fresh execution of the visual models. Clipping is the implemented empirical policy; the observed training range is not being presented as independently established test ground truth. The training mean/std used in the model are not the rounded summary values in `split_meta.json`. [R1, R3-R5]

## 3. Baseline methods

### B0 - Training-mean reference

B0 predicts the mean price of the 7,000 training records for every validation record. It is an internal numerical reference, not the required visual model or the course's official performance baseline. Its mean must not be computed from validation prices or the `sample_solution.csv` placeholders. [R4]

`prediction = mean(training prices)`  
`MSE = mean((validation price - prediction)^2)`

### B1 - Frozen ResNet-50 features with Ridge

The existing feature extractor supplies frozen 2,048-dimensional image descriptors. After training-only feature and target scaling, `RidgeCV` fits the regression head. Its candidate alphas are the 25 values from `numpy.logspace(-2, 4, 25)`, from 0.01 to 10,000. The code selects the regularisation strength within the training partition. The selected alpha is written to the original run log; it is not available in the committed summary table and is not invented here. Predictions are returned to the original price unit and clipped before validation metrics are calculated. [R3-R4]

B1 tests the usefulness of a linear price predictor on the pretrained representation. The backbone itself is not fine-tuned. [R3-R4]

### B2 - Frozen ResNet-50 features with an MLP

B2 uses the same extracted features, data split and scaling as B1. Its head is `Dense(256, relu) -> Dropout(0.2) -> Dense(1)`. It uses Adam with learning rate 0.001, MSE loss, batch size 64, a cap of 200 epochs, and early stopping on `val_loss` with patience 15 and best-weight restoration. The reported run stopped after 19 epochs, according to the experiment document; this count has not been checked against its original JSON here. [R4, R6]

B1 versus B2 is a comparison of two regression-head methods, including their fitting procedures. Both use frozen features. No change to the existing learning algorithms is included in this package.

## 4. Results and their evidence status

| ID | Configuration | Validation MSE | Validation MAE | Validation R2 | Evidence status |
|---|---|---:|---:|---:|---|
| B0 | Training-mean reference | 154,431.7 | 295.9 | -0.000046 | Recomputed from CSVs in this task |
| B1 | Frozen features + Ridge | 114,853.7 | 249.0 | 0.256 | Historical repository record; not re-run here |
| B2 | Frozen features + MLP | 133,378.6 | 270.8 | 0.136 | Historical repository record; not re-run here |

The historical table in `docs/experiments.md` reports B0 R2 to three decimal places as `-0.000`. The extra digits above show the small negative value from this audit; no historical result was edited. [R6, E1]

The CSV checker calculated training mean **720.9804862857**, B0 validation MSE **154431.6915118010**, MAE **295.8828421749** and R2 **-0.0000457248** using Python-float arithmetic. The existing baseline casts labels and mean to float32 before computing float64 metrics. A separate arithmetic check following those casts gives MSE **154431.6921967651**. Both round to **154,431.7**. Neither calculation loads images or trains B1/B2. [E1-E2]

From the historical validation table, B1 has 25.6% lower MSE than B0 and 13.9% lower MSE than B2 (percentages calculated from the rounded recorded values). The repository therefore records B1 as the selected Phase 2 visual baseline. This supports the choice for the recorded comparison, not a claim that Ridge is always better than an MLP. [R6]

The repository reports B1 Kaggle MSE **114,147.16** and contains `outputs/predictions/submission_phase2_ridge.csv`. The score was not checked on Kaggle, and that prediction file's full content was not supplied to this audit workspace. Its repository presence is recorded separately from numerical validation. Test performance cannot be recomputed from `test.csv` or `sample_solution.csv`. [R7, E2]

### Interpretation boundary

The historical experiment document suggests that the MLP overfits and calls the validation split conservative based on the reported Kaggle comparison. Those are the document's interpretations. This supplement verifies neither explanation: the original training history and Kaggle record were not available. The directly supportable result is that the recorded B1 validation MSE is lower in this comparison. The final report should retain this distinction. [R6]

## 5. Artifacts and handoff

| Artifact | Purpose / current evidence |
|---|---|
| `src/features.py`, `src/baseline.py`, `src/metrics.py` | Existing model, preprocessing and metric implementation reviewed; unchanged |
| `outputs/logs/phase2_results.json` | Original B0/B1/B2 metrics, alpha, target statistics, MLP settings and selected model; request the matching original file |
| `outputs/models/x_scaler.joblib` | Saved feature scaler; original artifact still to be checked |
| `outputs/models/ridge.joblib` | Saved B1 model; original artifact still to be checked |
| `outputs/models/mlp.keras` | Saved B2 model; original artifact still to be checked |
| `outputs/models/target_stats.json` | Training mean, std and clipping bounds used for prediction; original artifact still to be checked |
| `outputs/predictions/submission_phase2_ridge.csv` | Present in the inspected repository; local ID/format check remains available via the helper |
| `docs/evidence/phase2_baseline_audit_2026-10-08.json` | Executed CSV/B0 checks; missing artifacts explicitly marked PENDING |

The original baseline saves these shared output filenames and does not preserve the full MLP epoch history in its JSON. Do not overwrite an earlier run just to fill a missing record. First obtain and preserve the matching originals from the person who ran the experiment. If a new run is necessary, archive the old artifacts first and record the new result separately. Small authentic logs may be copied to `docs/evidence/`; this package contains no invented `phase2_results.json` and changes no ignore rules. [R4, R7]

## 6. Report-ready baseline paragraph

The visual baseline used an ImageNet-pretrained ResNet-50 with its backbone frozen. Each image was resized to 224 x 224 and processed with the ResNet-50 input function. Global average pooling produced 2,048 features per image. The same fixed 7,000/1,000 training-validation split was used for a Ridge head and a small MLP head. Feature and target scaling were fitted on the training partition only. We also used the training mean as a numerical reference. The experiment record gives validation MSE values of 154,431.7 for the mean reference, 114,853.7 for Ridge and 133,378.6 for the MLP. Ridge was selected because it had the lowest validation MSE among the two visual baselines. These results describe the recorded run; they do not establish that the same ordering holds for every training setting. [R3-R6]

**Use note:** This paragraph is a draft based on the repository record, not a substitute for checking the original B1/B2 log before the final report.

## Source and evidence register

All repository references below are pinned to `8cb91dc12b29035eee70d3c3caf7690c7540399d`.

- **[R1]** `config.py` and `data/splits/`.
- **[R2]** `docs/Phase1_Verification.md`, including Member B's recorded approval; `src/data.py`.
- **[R3]** `src/features.py`.
- **[R4]** `src/baseline.py`.
- **[R5]** `src/metrics.py`.
- **[R6]** `docs/experiments.md`, Fixed setup and Phase 2 - baseline.
- **[R7]** GitHub output-tree inventory and `.gitignore`; not an independently checked Kaggle result.
- **[E1]** `docs/evidence/phase2_baseline_audit_2026-10-08.json`.
- **[E2]** `docs/evidence/phase2_source_review_2026-10-08.json`, including split acquisition and arithmetic provenance.
