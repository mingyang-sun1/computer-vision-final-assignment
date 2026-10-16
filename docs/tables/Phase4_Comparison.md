# Phase 4 — Experiment Comparison

Reference commit: `90959c41384c7bd3c8c5a48147e8272246071bbe`. Prepared 16 October 2026.

**Evidence level:** the following are historical values transcribed from the pinned experiment log, not scores newly recomputed from complete CSVs in this session. MSE is in squared USD-1,000 units; MAE is in USD 1,000; R2 is unitless. Lower MSE is better.

| ID | Configuration | Reported validation MSE | MAE | R2 | Role |
|---|---|---:|---:|---:|---|
| B0 | Training-mean predictor | 154,431.7 | 295.9 | -0.000 | Numerical reference |
| B1 | Frozen ResNet-50 + Ridge, 224px | 114,853.7 | 249.0 | 0.256 | Visual baseline |
| B2 | Frozen ResNet-50 + MLP, 224px | 133,378.6 | 270.8 | 0.136 | Alternative head |
| E1-R320 | Frozen ResNet-50 + Ridge, 320px | 111,822.5 | 246.2 | 0.276 | Resolution comparison |
| E1-R400 | Frozen ResNet-50 + Ridge, 400px | 112,219.7 | 242.0 | 0.273 | Resolution comparison |
| E1-M320 | Frozen ResNet-50 + MLP, 320px | 133,046.0 | 267.4 | 0.138 | Resolution comparison |
| E1-M400 | Frozen ResNet-50 + MLP, 400px | 126,795.1 | 265.8 | 0.179 | Resolution comparison |
| E2 | Partial fine-tuning, 224px | 102,745.7 | 235.7 | 0.335 | No-augmentation core run |
| E3 | Partial fine-tuning, 320px | 99,051.0 | 229.8 | 0.359 | Best reported single core run |
| E4 | Partial fine-tuning, 224px + augmentation | 108,784.5 | 246.3 | 0.296 | Augmentation comparison |
| E5 | Equal mean of E2 + E3 + E4 | 95,850.6 | Not recorded | 0.379 | Recommended existing ensemble |

**Reading the comparison.** E5 has the lowest reported validation MSE among the numerical rows above. Its reduction is 16.55% relative to B1 and 3.23% relative to E3. E4 is worse than E2 alone; this does not establish that excluding E4 improves an ensemble. These are descriptive observations from the saved runs, not tests of significance.

## Precision and evidence boundaries

E3 appears as 99,051.0 in the E3 run table and 99,051.2 in the E5 comparison/source summary. The Phase 4 historical table consistently uses 99,051.0 from the dedicated E3 row. Do not silently replace either historical record. The new checker calculates separate metrics from the rounded prediction exports and writes them to a new evidence directory.

E5 MAE was not recorded in the historical table and is therefore not invented here. The local checker calculates it from the stored predictions. B1/B2/E1 validation exports are not present in the inspected prediction directory, so their historical scores must not be labelled as newly recomputed.

## Additional existing experiments

| Existing item | Available evidence | Phase 4 treatment |
|---|---|---|
| V3, 224px and 320px columns | `outputs/predictions/v3_val_predictions.csv`, with true prices and two prediction columns | Recompute both columns in the local audit. Confirm original backbone/preprocessing/run identity before treating them as fully identified alternatives. A better stored V3 score blocks automatic recommendation. |
| ConvNeXt / aspect-preserving / cross-backbone work | README narrative and implementation/notebook files; no complete numerical result table in `docs/experiments.md` | Record the work, but do not invent missing numerical rows, training time, or execution evidence. Confirm any decision-relevant extra results with Member B. |

## Controlled-comparison limits

B1 versus E2 is a method comparison, not a pure unfreezing ablation: the regression head and optimisation method also differ. E3 versus E2 changes input resolution in the described setup. E4 versus E2 changes the augmentation policy; matching executed epoch counts do not prove that every legacy hyperparameter was recorded or that converged optima were compared. E5 combines existing predictions rather than training a new backbone.

## Sources

- [Pinned experiment log](https://github.com/mingyang-sun1/computer-vision-final-assignment/blob/90959c41384c7bd3c8c5a48147e8272246071bbe/docs/experiments.md).
- [Pinned README](https://github.com/mingyang-sun1/computer-vision-final-assignment/blob/90959c41384c7bd3c8c5a48147e8272246071bbe/README.md).
- [E2/E4 original-artifact supplement](https://github.com/mingyang-sun1/computer-vision-final-assignment/blob/90959c41384c7bd3c8c5a48147e8272246071bbe/docs/Phase3_MemberB_Artifact_Update.md).
- [Machine-readable historical table](../evidence/phase4_historical_metrics_2026-10-16.json).
