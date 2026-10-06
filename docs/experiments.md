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

Each experiment changes exactly one factor relative to the row it is
compared against. Deadline 12 Oct.

### E1 — input resolution

The only change is the resolution fed to the frozen backbone. Split, seed,
backbone, pooling and both heads stay at their Phase 2 settings.

| Input | Ridge MSE | Ridge MAE | Ridge R² | MLP MSE | MLP MAE | MLP R² | MLP epochs |
|---|---|---|---|---|---|---|---|
| 224px (baseline) | 114,853.7 | 249.0 | 0.256 | 133,378.6 | 270.8 | 0.136 | 19 |
| **320px** | **111,822.5** | 246.2 | **0.276** | 133,046.0 | 267.4 | 0.138 | 19 |
| 400px | 112,219.7 | **242.0** | 0.273 | **126,795.1** | **265.8** | **0.179** | **47** |

Reproduce: `python -m src.exp_resolution`
Raw log: `outputs/logs/phase3_resolution.json`

Observations:

- **The gain is real but small.** Ridge improves 2.6% at 320px
  (114,853.7 → 111,822.5). Resolution alone does not come close to
  explaining the spread on the Kaggle leaderboard.
- **The two heads disagree about the optimum.** Ridge peaks at 320px and
  is slightly worse at 400px, while the MLP keeps improving to 400px. MAE
  at 400px is the best of any configuration, so 320px is not uniformly
  better — it just minimises squared error slightly more.
- **Higher resolution makes the non-linear head viable.** At 400px the MLP
  trains for 47 epochs instead of 19 and improves 4.9% over its own 224px
  result. The frozen features carry more usable signal at higher
  resolution; a linear readout cannot exploit it. This is the clearest
  indication so far that the **readout, not the input, is the bottleneck**.
- Natively the images are about 400px wide, so 400 is the ceiling — going
  higher would add no information.

### E2 — partial fine-tuning

Compared directly against B1. Resolution is held at 224px and the split and
seed are unchanged, so the only factor differing is whether the backbone
adapts. Layer4 (14.9M of the backbone's 23.5M parameters) plus a new
regression head are unfrozen; Adam at 1e-4; early stopping on validation
loss.

| Configuration | Val MSE | Val MAE | Val R² | Kaggle MSE | vs B1 | Best epoch |
|---|---|---|---|---|---|---|
| B1 frozen + Ridge | 114,853.7 | 249.0 | 0.256 | 114,147.16 | — | — |
| **E2 layer4 fine-tuned** | **102,745.7** | **235.7** | **0.335** | **103,129** | **−10.5%** | 12 of 15 |

Reproduce: `python -m src.exp_finetune`
Raw log: `outputs/logs/phase3_finetune.json` (includes the full per-epoch history)

Observations:

- **Fine-tuning helps substantially more than resolution did** (−10.5%
  against B1, versus −2.6% for the best E1 setting). Adapting the
  representation is worth roughly four times as much as feeding it more
  pixels.
- **The model is overfitting.** Training loss falls to 0.063 while the best
  validation loss is 0.654 — an order of magnitude gap. Validation loss was
  also noisy across epochs (0.689, 0.774, 0.698, 0.682, 0.666, ...), and the
  best epoch was 12 of a 15-epoch cap, so the run had not converged. A
  longer schedule with augmentation is the obvious next lever.
- 64% of parameters are trainable, which is a lot for 7000 images. Freezing
  more of the backbone is the other obvious lever, and the two trade off
  against each other.

### E3 — fine-tuning at 320px

E2 with the resolution changed and nothing else.

| Configuration | Val MSE | Val MAE | Val R² | Kaggle MSE | vs B1 | Best epoch |
|---|---|---|---|---|---|---|
| E2 fine-tune 224px | 102,745.7 | 235.7 | 0.335 | 103,129 | −10.5% | 12 of 15 |
| **E3 fine-tune 320px** | **99,051.0** | **229.8** | **0.359** | _pending_ | **−13.8%** | 10 of 15 |

Reproduce: `python -m src.exp_finetune --img-size 320 --tag E3_320px`

Observations:

- **Resolution and fine-tuning do compose**, adding a further 3.6% on top of
  E2. This is the opposite of what the interim per-epoch readings suggested
  (at epoch 4 E3 was still behind E2), which is a reminder not to read a
  training curve mid-run.
- E3 is the best configuration so far: 13.8% below the frozen-feature
  baseline.

### E4 — fine-tuning with augmentation

E2 with augmentation added and nothing else. Augmentation is mild and
targets the two directions the assignment says the data varies in --
translation and zoom for pose, brightness and contrast for lighting. The
horizontal flip is safe because house facades carry no meaningful left-right
orientation here. It is applied to the training partition only.

| Configuration | Val MSE | Val MAE | Val R² | vs B1 | Best epoch | Train loss @ epoch 10 |
|---|---|---|---|---|---|---|
| E2 fine-tune 224px, no augment | 102,745.7 | 235.7 | 0.335 | −10.5% | 12 of 15 | 0.080 |
| E4 fine-tune 224px, augmented | 108,784.5 | 246.3 | 0.296 | −5.3% | 14 of 15 | 0.197 |

Reproduce: `python -m src.exp_finetune --augment --tag E4_augment`

Observations:

- **At a fixed 15-epoch budget, augmentation hurts** — E4 lands 5.9% worse
  than E2. The mechanism is visible in the training loss: augmentation
  makes each epoch a harder task, so at epoch 10 E4 is at 0.197 against
  E2's 0.080. The model is learning more slowly, not learning less.
- **The comparison is confounded by the epoch budget**, and this should be
  stated rather than hidden. E4's best epoch was 14 of 15, so it was still
  improving when the budget ran out, while E2 had already peaked at epoch
  12. E4 has not been given a fair chance to converge.
- The honest conclusion is therefore limited: *augmentation slows
  convergence faster than 15 epochs can absorb*. Whether it helps at
  convergence is not yet tested, and testing it needs a longer run — see
  the pending experiment below.

### Phase 3 summary

| ID | Configuration | Val MSE | Val R² | Kaggle MSE | vs B1 |
|---|---|---|---|---|---|
| B1 | Frozen features + Ridge, 224px | 114,853.7 | 0.256 | 114,147.16 | — |
| E1 | Frozen features + Ridge, 320px | 111,822.5 | 0.276 | — | −2.6% |
| E2 | Fine-tuned, 224px | 102,745.7 | 0.335 | 103,129 | −10.5% |
| **E3** | **Fine-tuned, 320px** | **99,051.0** | **0.359** | _pending_ | **−13.8%** |
| E4 | Fine-tuned, 224px + augmentation | 108,784.5 | 0.296 | — | −5.3% |

### E5 — planned

| ID | Configuration | Compared against | Rationale |
|---|---|---|---|
| E5 | Fine-tuned + augmentation, longer schedule | E2 and E4 | Gives augmentation the epochs it needs, so the E2/E4 comparison is not decided by the budget |

_To be filled in._

## Phase 4 — final model selection

_To be filled in after 16 Oct. Record which configuration was chosen and the
evidence it was chosen on._
