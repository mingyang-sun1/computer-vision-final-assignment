# Experiment log

Running record of every experiment in the project, in the order they were
run. The comparison tables in the final report should be built from this
file, so the numbers here must match `outputs/logs/*.json` exactly.

Add a row when an experiment finishes. Do not edit the numbers of an
experiment that has already been reported to Kaggle.

## Fixed setup

Every experiment uses the same conditions unless its row says otherwise.
Changing any of these invalidates the comparison.

| | |
|---|---|
| Train / validation | `data/splits/train_split.csv` (7000 rows), `val_split.csv` (1000 rows) |
| Seed | 42 (`config.SEED`) |
| Backbone | ResNet-50, ImageNet weights, global average pooling, 2048-d |
| Feature cache | `outputs/features/*.npy`, regenerated automatically if stale |
| Input | 224x224, `resnet50.preprocess_input` |
| Target | standardised using training mean/std |
| Post-processing | predictions clipped to the training price range [195, 2000] |
| Selection criterion | **validation MSE**. The test set is never used for tuning. |

## Phase 2 — baseline

| ID | Configuration | Val MSE | Val MAE | Val R² | Kaggle MSE |
|---|---|---|---|---|---|
| B0 | Predict the training mean | 154,431.7 | 295.9 | -0.000 | — |
| **B1** | **Frozen features + Ridge** — selected | **114,853.7** | **249.0** | **0.256** | **114,147.16** |
| B2 | Frozen features + MLP (256 hidden, dropout 0.2) | 133,378.6 | 270.8 | 0.136 | — |

Reproduce: `python -m src.baseline` then `python -m src.predict`.
Raw log: `outputs/logs/phase2_results.json`.

Observations:

- **B1 beats B2.** The MLP early-stopped after only 19 epochs. With 2048
  features and 7000 training samples the non-linear head overfits quickly
  while the linear probe does not, which suggests the price signal is
  already close to linearly available in the frozen representation.
- The clipping step moves validation MSE by about 2 points for B1 but about
  680 for B2, so it matters most for the head that extrapolates harder.
- Kaggle MSE (114,147.16) came back 706 points *better* than the validation
  estimate, so the validation split is honest and slightly conservative —
  no sign of validation overfitting.
- B0 on the test set is not directly measurable (no labels), but assuming a
  similar spread to validation, B1 is roughly 26% better than the mean
  predictor.

## Phase 3 — improvement experiments

Planned, deadline 12 Oct. Each row changes exactly one factor relative to
the baseline it is compared against.

| ID | Configuration | Compared against | Val MSE | Val MAE | Val R² | Kaggle MSE |
|---|---|---|---|---|---|---|
| E1 | Partial fine-tuning (unfreeze layer4 + head) | B1 | | | | |
| E2 | Frozen features + augmentation | B1 | | | | |
| E3 | Fine-tuning + augmentation | E1 | | | | |

_To be filled in._

## Phase 4 — final model selection

_To be filled in after 16 Oct. Record which configuration was chosen and the
evidence it was chosen on._
