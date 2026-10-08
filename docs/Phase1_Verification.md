# Phase 1 - Verification Record
**Member A:** Mingyang Sun (1657392) | **Prepared:** 8 October 2026  
**Repository inspected:** main @ 9a7ed84 | **Reviewed by Member B:** Yikai Qian, 8 October 2026

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
| Protect existing splits from accidental overwrite | **Resolved** (Member B, commit 8f94c04). `python -m src.data` now verifies read-only; regenerating requires an explicit `--rebuild`. Verified the file is byte-identical after a verify run. |
| Feature-cache identity | **Resolved** (Member B, commit 8f94c04). Caches now carry a SHA-1 fingerprint of the image ids in order and are reused only on a match. Existing caches have no sidecar and are re-extracted once, by design. |
| Essential run-log and model backup | Both members; current Git exclusions do not establish the presence of a shared backup. |
| Image existence, decoding and duplicate-image checks | Not inspected; explicitly excluded at the user's request. No pass/fail claim. |
| Final teammate review | **Complete.** See section 6. |

Member A's supplement, requirements review and metadata evidence are prepared, and Member B's review is recorded in section 6. The Phase 1 deadline of 7 October was not met; this work is dated 8 October and is not backdated.

## 6. Member B review

Reviewed by Yikai Qian on 8 October 2026, against main @ 8f94c04.

**Outcome: approved.** The audit tool was run against this repository rather
than read: `python tools/verify_phase1_metadata.py --repo-root .` reports 32
executed checks and 32 passes, and the split blob hashes it records match the
committed `data/splits/` files exactly. The document's scope is stated
accurately throughout — it claims CSV and static-code evidence only, and does
not claim to have inspected images, run models, or verified backups. The
distinction between "no evidence of a problem" and "verified correct" is
maintained where it matters.

One correction was made during review: the path to the audit JSON in section 2
read `docs/evidence/...`, but the file sits directly in `docs/`. Fixed in
commit a32b8b7.

Both code risks raised in section 5 were confirmed as real and have been
fixed; see commits 8f94c04 and the entries in the table above. Neither had
produced a wrong result — they were latent misuse risks, and the wording in
section 5 said so rather than overstating them.

One note for the record: `python -m src.data --rebuild` regenerates
`split_meta.json` with a fresh `created_utc`, so that one file is not
byte-stable across rebuilds. The split CSVs themselves are, and their blob
hashes are unchanged. Any check of the split's identity should use the CSVs.

## Evidence and source register
**[U1-U3]** User-supplied train.csv, test.csv and sample_solution.csv, received 8 October 2026. Exact byte hashes are in [E1].

**[R1]** mingyang-sun1/computer-vision-final-assignment, commit 9a7ed84f3681d9e13d4d48a43ebdddbd5e94d591. Inspected sources: config.py; src/data.py; src/features.py; src/baseline.py; src/exp_finetune.py; src/predict.py; .gitignore; README.md; docs/Data_Management.md; and data/splits/.

**[S1]** COMP90086_2026_Project.pdf, supplied assignment specification, pp. 1-4. **[S2]** COMP90086_Project_Planning(2).pdf, pp. 1-3. **[S3]** User-supplied teacher-feedback screenshot identifying the missing Data Management section.

**[E1]** phase1_metadata_audit_2026-10-08.json, generated in this task. **Audit boundary:** CSV and static-code evidence only; no images, model execution, Kaggle verification or signed team approval.

Pinned repository source: [main snapshot 9a7ed84](https://github.com/mingyang-sun1/computer-vision-final-assignment/tree/9a7ed84f3681d9e13d4d48a43ebdddbd5e94d591).
