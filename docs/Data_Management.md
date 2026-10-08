# Phase 1 - Data Management
**COMP90086 Computer Vision, 2026 Semester 2 | Predicting House Prices**

**Member A / document owner:** Mingyang Sun (1657392)

**Member B / reviewer:** Yikai Qian (1722421); **reviewed and approved 8 October 2026** — see section 6 of Phase1_Verification.md

**Revision date:** 8 October 2026 | **Repository baseline:** main @ 8f94c04

**Status:** The data-management prose and CSV-level audit are complete, and the two-person review required by the plan is done. All 32 executed metadata checks passed. Images and local training execution remain outside the verification. The original internal Phase 1 target was 7 October; this supplement is dated 8 October and does not claim earlier completion. Both code risks raised in the verification record have since been fixed (commit 8f94c04).

## 1. Dataset source
The project predicts a house sale price from an exterior photograph. The target is recorded in units of USD 1,000. The team uses the course-provided subset and permitted general-purpose pretrained features, as agreed in the project plan. The assignment prohibits additional images or metadata from the original House Prices and Images - SoCal dataset, models pretrained on this task or dataset, and pretrained MLLMs. Test ground-truth labels must not be sought or used. [S1, S2]

| Supplied file | Data rows | Columns | Role |
|---|---:|---|---|
| train.csv | 8,000 | imageid, price | Labelled source for training and validation |
| test.csv | 3,000 | imageid | Test identifiers for inference |
| sample_solution.csv | 3,000 | imageid, price | Submission-format example only |

The three uploaded CSVs were read in full and retained unchanged. Every sample price is 500; these are template values, not test ground truth and not predictions from the team's model. Its identifiers match test.csv in both membership and order. The train/ and test/ image folders were not supplied for inspection and were deliberately excluded from this audit. Row counts above refer to CSV records, not verified image files. [U1-U3, E1]

## 2. Folder structure
The existing code resolves the dataset through config.DATA_ROOT using CV_DATA_ROOT, with a fallback to ../comp-90086-2026/house_dataset relative to the repository. Personal paths should be configured through the environment rather than added to source code. The source_csv field in split_meta.json records a historical machine path; it is not a required path for other members. [R1]

```text
<dataset root>/                   # course files, outside the repository
  train.csv, test.csv, sample_solution.csv
  train/, test/                  # image folders; not inspected here
<repository>/
  config.py, src/                 # shared configuration and code
  data/splits/                   # existing committed split files
  docs/                          # data management and audit evidence
  outputs/predictions/            # CSV outputs permitted by .gitignore
  outputs/models/, features/, logs/  # generated outputs, ignored by default
```
The present .gitignore permits prediction CSVs but excludes image patterns and most other outputs. It is a safeguard, not proof that no restricted file can be committed. Team practice is to retain original data unchanged, keep working data separate from code, review staged files, and maintain a shared backup of essential run artefacts. The backup policy is specified here; the team's actual backup has not been verified. [R1]

## 3. Train / validation / test separation
The current project uses one fixed 7,000/1,000 split of the 8,000 labelled records. The validation fraction is 0.125: 87.5% training and 12.5% validation, not the earlier proposed 80%/20%. The configured seed is 42 and stratification uses price deciles. Existing split CSVs remain authoritative and are not replaced by this document package. [R1, E1]

| Partition | Records | Authoritative source |
|---|---:|---|
| Training | 7,000 | data/splits/train_split.csv |
| Validation | 1,000 | data/splits/val_split.csv |
| Test | 3,000 | test.csv; no ground-truth prices supplied |

| Labelled records | Mean | Median | Sample std. | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| All 8,000 | 720.65 | 639.90 | 395.86 | 195.00 | 2,000.00 |
| Training 7,000 | 720.98 | 639.90 | 396.27 | 195.00 | 2,000.00 |
| Validation 1,000 | 718.32 | 639.45 | 393.17 | 195.00 | 1,999.90 |

All statistics are calculated from the actual CSVs in USD 1,000; sample standard deviation uses n - 1. The full-source mean exceeds the median and the price range is broad. The existing price-decile design aims to retain representation across price ranges. It does not establish that this single split is representative in every visual or geographic respect. Summary statistics agree with split_meta.json after rounding. [E1]

The audit found no duplicate identifiers within any input or split file, no train/validation ID overlap, and no overlap between the labelled source and the test list. The two partitions cover all 8,000 labelled source IDs exactly once. Every partition price agrees numerically with the uploaded train.csv; no cleaning, label imputation or relabelling was needed for these metadata checks. Numeric formatting differences such as 750 and 750.0 are treated as equal. [E1]

**Preservation rule:** Do not recreate the split to improve validation scores. If a genuine source-data correction requires a new split, record the reason, version it, invalidate affected caches and rerun the comparisons. Current split creation and cache protections still need the follow-up described in the verification record; this package does not modify their code.

## 4. Leakage prevention
Training records are used to fit parameters; validation records are used to select settings and checkpoints; test records are reserved for producing predictions. The sample solution supplies only the output schema and identifiers. Neither its placeholder prices nor test results should be used as training labels or as a substitute validation set. These are the project's operating rules, not a claim that every historical run has been independently audited. [S1, R1]

In the inspected implementation, baseline feature standardisation is fitted on the training partition. Target mean, standard deviation and clipping limits are also calculated from training prices. The same transformations are then applied to validation and test data; predictions are returned to the original price units for evaluation and export. Keep the chosen backbone preprocessing and the run's recorded input resolution consistent. [R1: src/baseline.py, src/features.py, src/predict.py]

For E4, the augmentation object is passed only to the training data pipeline; validation is built without it. This is confirmed by code inspection, not by rerunning training. ID-level separation does not check whether different filenames contain duplicate images or views of the same property; no image-content or property-group audit was performed. [R1: src/exp_finetune.py; E1]

## 5. Reproducibility
The committed split files, rather than the seed alone, define the comparison dataset. The supplied audit report records source-file SHA-256 hashes and the Git blob identifiers of both split files. These identities allow a later result to be associated with the same source and row ordering. The existing repository is preserved at the inspected commit; this delivery contains no changed model, preprocessing or split files. [E1]

For each experiment, retain the code revision; source/split hashes; seed; Python and resolved package versions; backbone and weight source; input size and preprocessing; target statistics; augmentation parameters; learning rate and batch size; epoch cap and early-stopping settings; executed and best epochs; validation MSE; checkpoint; and corresponding prediction files. Record training duration when available. This is the recording standard for subsequent work, not a statement that every field is already archived.

| Artefact path in current code | Role / current handling |
|---|---|
| outputs/logs/phase2_results.json | Baseline metrics and selection; generated by baseline.py |
| outputs/logs/phase3_E2_finetune.json | E2 settings, metrics and per-epoch history |
| outputs/logs/phase3_E4_augment.json | E4 settings, metrics and per-epoch history |
| outputs/models/ | Saved models and baseline target/scaler artefacts |
| outputs/predictions/*.csv | Prediction files currently permitted in Git |

The current .gitignore excludes logs and model outputs by default. Essential small logs must therefore be deliberately archived or selectively included through an agreed change; they should not be assumed to be backed up by Git. Large checkpoints and feature caches may remain outside Git, but the two members must agree how necessary reproduction artefacts are shared and recovered. The repository reports limits to exact numerical reproduction; those limits and any rerun differences must be recorded rather than described as exact reproduction. No training rerun or checkpoint recovery was performed here. [R1]

A read-only verifier is included as tools/verify_phase1_metadata.py. It uses the Python standard library, reads existing CSVs, prints the audit result and optionally writes a new JSON file. It does not open images, train a model, download data or rewrite a split. Its fixed checks refer to this 8,000/3,000 dataset and the inspected repository snapshot; a changed source requires deliberate review.

## 6. Submission and data handling
The code submission must include model code, the team's Kaggle-format predictions, a README explaining execution and data locations, and any additional files needed to recreate the results. Course-provided train/test images must be excluded. The repository archive and the final course submission are distinct: ignoring a necessary artefact in Git does not remove the obligation to provide it or explain its reproduction. [S1, p. 3]

Predictions must use exactly imageid and price, with one finite numeric prediction per test ID, in USD 1,000. Preserve the sample/test identifier order, filename spelling and CSV header, and do not add a dataframe index. Check duplicate, missing or extra IDs before submission. Replace the template's constant 500 values with predictions from the team's own model; the unchanged template is not evidence of a model-generated submission. [U2-U3; S1]

This Phase 1 package contains only documentation, a metadata audit and its read-only helper. Raw uploaded CSVs and all images are deliberately excluded from the Git-ready package. Working and Library copies of the three source CSVs have been saved separately for this task; that action does not certify a backup on either member's computer.

**Handover:** Member A's supplemental document and metadata evidence are prepared on 8 October 2026. Member B should confirm the local data path, existing-split protection and evidence storage, then review this document. Image inspection remains explicitly excluded from this delivery. Teammate approval, local runtime success and submission status are not yet verified.

## Evidence and source register
**[U1-U3]** User-supplied train.csv, test.csv and sample_solution.csv, received 8 October 2026. Exact byte hashes are in [E1].

**[R1]** mingyang-sun1/computer-vision-final-assignment, commit 9a7ed84f3681d9e13d4d48a43ebdddbd5e94d591. Inspected sources: config.py; src/data.py; src/features.py; src/baseline.py; src/exp_finetune.py; src/predict.py; .gitignore; README.md; docs/Data_Management.md; and data/splits/.

**[S1]** COMP90086_2026_Project.pdf, supplied assignment specification, pp. 1-4. **[S2]** COMP90086_Project_Planning(2).pdf, pp. 1-3. **[S3]** User-supplied teacher-feedback screenshot identifying the missing Data Management section.

**[E1]** phase1_metadata_audit_2026-10-08.json, generated in this task. **Audit boundary:** CSV and static-code evidence only; no images, model execution, Kaggle verification or signed team approval.

Pinned repository source: [main snapshot 9a7ed84](https://github.com/mingyang-sun1/computer-vision-final-assignment/tree/9a7ed84f3681d9e13d4d48a43ebdddbd5e94d591).

Related record: [Phase 1 Verification](Phase1_Verification.md).
