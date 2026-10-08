# Phase 1 - Verification Record
**Member A:** Mingyang Sun (1657392) | **Prepared:** 8 October 2026  
**Repository inspected:** main @ 9a7ed84 | **Review by Member B:** pending

## 1. Checks actually completed
The supplied train.csv, test.csv and sample_solution.csv were read in full. Their originals were not edited. The audit compared identifiers and numeric labels against the existing repository split and checked the recorded summary statistics. All 32 executed checks passed; this is a metadata result, not an image or training-pipeline acceptance result. [E1]

| Check group | Observed result |
|---|---|
| Source schema and counts | train: 8,000 (imageid, price); test: 3,000 (imageid); sample: 3,000 (imageid, price) |
| IDs and prices | No duplicate/blank IDs; expected numeric .jpg filenames; all supplied prices finite and positive |
| Source/test isolation | 0 shared IDs between the 8,000 labelled IDs and 3,000 test IDs |
| Train/validation isolation | 0 shared IDs; 7,000 + 1,000 rows cover all source IDs exactly once |
| Partition-label agreement | 0 mismatched prices against uploaded train.csv |
| Test/sample consistency | Exact same 3,000 identifiers in the same order |
| Template prices | One unique value: 500; format placeholder only |
| Split identity and metadata | Both Git blob hashes match; settings and 2-decimal statistics match the repository record |

## 2. Evidence identity and method
Repository reference: 9a7ed84f3681d9e13d4d48a43ebdddbd5e94d591.

The split bytes used in the audit were reconstructed in an isolated working directory from the uploaded train.csv using the documented price-decile method. Their complete Git blob SHA-1 identifiers were then compared with those returned by the GitHub connector. Both matched exactly, identifying byte-for-byte identical contents, including order and formatting. No production split was regenerated, overwritten or committed. The JSON metadata was read from the pinned repository and compared by values, not asserted to be byte-identical. [R1, E1]

```text
train_split.csv Git blob:
05851b5173130d45b78f1057ec7f0e78cf827a06
val_split.csv Git blob:
54c0018bb7fbb08c59dd26c511928836e5ea46ba
```
Full input SHA-256 hashes, row counts, statistics, all individual checks and the acquisition method are retained in docs/phase1_metadata_audit_2026-10-08.json. The helper verifies the actual existing split files on a member's machine; it does not reconstruct them.

## 3. Safe local verification
Run from the project root after setting CV_DATA_ROOT to the actual dataset folder:

```powershell
python tools/verify_phase1_metadata.py --repo-root . `
  --out phase1_audit_local.json
```
The output filename must not already exist. A nonzero exit code indicates a failed check or an input error. This command does not import src.data. Avoid using python -m src.data merely as a verification command in the inspected version: its main() builds and saves a new split before checking it. [R1: src/data.py]

## 4. Assignment requirements checked
This supports Phase 1; it is not the submitted plan or final IEEE report. The supplied assignment and feedback were reviewed, but registration, submissions and later LMS updates were not checked. [S1-S3]

| Requirement | Phase 1 treatment / remaining action |
|---|---|
| Two-person allocation and estimated dates | A/B allocation is in the existing plan; deadline was 4 October. Preserve contribution records. |
| Data Management missing in feedback | Existing six-section Data_Management.md completed, using actual CSVs and repository settings. |
| Permitted data and model use | Restrictions recorded; no test ground truth or original-dataset additions used in this audit. |
| Code submission | Model code, own predictions, README and reproduction files; no course images. Final packaging is a later-stage task. |
| Report submission | Separate IEEE-format PDF: up to four A4 content pages; references may extend to page five. |
| Final deadline | Supplied specification: 25 October 2026, 23:59. Existing plan targets 24 October for normal submission. |

## 5. Open items and responsibility
| Item | Owner / status |
|---|---|
| Local dataset path and training environment | Member B, with A reviewing; not executed on either member's computer here. |
| Protect existing splits from accidental overwrite | Member B; current src.data main() rebuilds and saves. The verifier supplied here is read-only. No source-code fix applied. |
| Feature-cache identity | Member B; current cache check uses row count, so same-length reordered or changed splits require invalidation. No evidence that current cached results are wrong. |
| Essential run-log and model backup | Both members; current Git exclusions do not establish the presence of a shared backup. |
| Image existence, decoding and duplicate-image checks | Not inspected; explicitly excluded at the user's request. No pass/fail claim. |
| Final teammate review | Yikai Qian; pending. Approval date and review commit should be recorded after actual review. |

Member A's supplement, requirements review and metadata evidence are prepared. Record environment checks and teammate approval when completed; do not backdate them to 7 October.

## Evidence and source register
**[U1-U3]** User-supplied train.csv, test.csv and sample_solution.csv, received 8 October 2026. Exact byte hashes are in [E1].

**[R1]** mingyang-sun1/computer-vision-final-assignment, commit 9a7ed84f3681d9e13d4d48a43ebdddbd5e94d591. Inspected sources: config.py; src/data.py; src/features.py; src/baseline.py; src/exp_finetune.py; src/predict.py; .gitignore; README.md; docs/Data_Management.md; and data/splits/.

**[S1]** COMP90086_2026_Project.pdf, supplied assignment specification, pp. 1-4. **[S2]** COMP90086_Project_Planning(2).pdf, pp. 1-3. **[S3]** User-supplied teacher-feedback screenshot identifying the missing Data Management section.

**[E1]** phase1_metadata_audit_2026-10-08.json, generated in this task. **Audit boundary:** CSV and static-code evidence only; no images, model execution, Kaggle verification or signed team approval.

Pinned repository source: [main snapshot 9a7ed84](https://github.com/mingyang-sun1/computer-vision-final-assignment/tree/9a7ed84f3681d9e13d4d48a43ebdddbd5e94d591).
