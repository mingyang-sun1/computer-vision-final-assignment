# Phase 5 — Verification and Completion Guide

**Scope:** Member A numerical error analysis and example selection for E5.  
**Reference main:** `ae74826454627addbeabdec5905595a9599fc105`  
**Execution date:** 17 October 2026.

## 1. What was actually verified

| Check | Result | Evidence |
|---|---|---|
| Six source identities, accepting LF/CRLF differences | PASS | `phase5_run_20261017/summary.json` |
| Fixed 7,000/1,000 split; disjoint and unique IDs | PASS | Same summary |
| All E2/E3/E4/E5 IDs and true prices match the saved validation split | PASS | Same summary |
| All 1,000 E5 values equal the rounded three-component mean | PASS | Same summary |
| Numerical source and structural checks | 12 PASS | Same summary |
| New Phase 5 correctness tests | 19 PASS | `phase5_checker_tests_2026-10-17.txt` |
| Combined Phase 2/3/4/5 checker tests | 101 PASS | Same test log |
| Real E5 analysis of all validation rows | PASS | `phase5_run_20261017/analysis.md` and CSVs |
| Two numerical figures generated and visually inspected | PASS | `docs/figures/phase5_*.svg` / `.pdf` |
| Original example-photo rendering and visual interpretation | NOT COMPLETED | Photos require the local course `train/` folder |
| Retraining or original checkpoint execution | Not part of this run | Existing Phase 7 reproducibility follow-up |

Evidence paths in this table are relative to `docs/evidence/` unless shown
with `docs/`. The automated test suite uses synthetic fixtures to check
correctness and failure handling. The committed Phase 5 analysis is a
separate execution on real repository prediction CSVs; its numbers are not
synthetic test results.

## 2. Reproduce the numerical analysis

From the repository root, in PowerShell:

```powershell
py -3 -m unittest discover -s tests -p "test_analyze_phase5_errors.py" -v
py -3 tools/analyze_phase5_errors.py --repo-root .
```

Python 3.10+ and the standard library are sufficient. Expected output:

```text
Validated 1000 E5 rows against the fixed split and all three components.
MSE 95852.002676; MAE 226.423020; bias +6.912374 (predicted - true).
Written: .../docs/evidence/phase5_run_<new timestamp>
```

The default command creates a fresh timestamped evidence directory. It
does not overwrite the committed run or any source prediction. To choose
a different directory, use `--output-dir docs/evidence/<new name>`; the
directory must not already exist.

The five numerical output files are `summary.json`, `analysis.md`,
`per_image_errors.csv`, `price_buckets.csv` and `representative_examples.csv`.
The committed run also includes the provenance of its figures.

## 3. Use or regenerate the numerical figures

The committed SVG files render in GitHub. The corresponding PDFs provide
vector figures for the final paper. No original photos are needed for these
numerical figures.

```powershell
py -3 -m pip install matplotlib pillow
py -3 tools/plot_phase5_errors.py --analysis-dir docs/evidence/phase5_run_20261017 --output-dir outputs/phase5_figures
```

The new output directory receives SVG, PDF and PNG copies plus source-file
hashes. If it already exists, choose another `--output-dir`. To plot a new
analysis run, replace `--analysis-dir` with the path printed by that run.

## 4. Finish the photo examples locally

Use the existing course dataset on the local computer. `--image-root`
must point to its **train** directory, because validation images belong to
the provided training set. It does not point to the course test directory.

```powershell
$env:CV_DATA_ROOT = "C:\path\to\house_dataset"
py -3 tools/plot_phase5_errors.py --analysis-dir docs/evidence/phase5_run_20261017 --examples-only --image-root "$env:CV_DATA_ROOT\train"
```

The nine exact selected filenames are:

| Selection | Image filenames |
|---|---|
| Closest predictions | `751.jpg`, `1770.jpg`, `5703.jpg` |
| Largest overestimates | `3947.jpg`, `3709.jpg`, `1416.jpg` |
| Largest underestimates | `1977.jpg`, `2138.jpg`, `7737.jpg` |

The helper checks that every required image exists and can be opened before
creating a photo-output directory. It writes
`outputs/phase5_examples/phase5_selected_examples.png` (also SVG/PDF),
`visual_review.csv` and a short README. Aspect ratios are preserved.
An existing directory is protected; use `--image-output-dir` with a new path
for a subsequent run.

Open the panel, record visible features in the review CSV, and distinguish
observations from possible explanations. For example, a visible obstruction
may be an observation; claiming that it caused a particular error is an
interpretation that needs additional evidence. No visual-review field has
been marked as completed in advance.

Photo panels can be used in the written report. They stay under the ignored
`outputs/` directory and must not be added to the code submission with the
provided dataset images.

## 5. Handoff and Phase 6 integration

- Member A's numerical analysis, selected IDs, figures and English report
  section are ready for review.
- The original photos need local rendering and visual inspection to finish
  the example discussion.
- Member B can add interpretation of the original E2/E3/E4 training histories.
  E3's history must not be described as an ensemble training curve.
- Keep actual model-approval acknowledgements in `Phase4_Model_Selection.md`.
  Automated checks do not create a personal signature.
- Use `docs/report/Phase5_Error_Analysis_Section.md` when preparing the Phase 6
  report. It flags older claims in `build_docx.py` that also need reconciliation
  when the full report is assembled.

The analysis and suggestions are validation-based descriptions. No new
training result, test error analysis, measured inference speed or successful
photo inspection is claimed.
