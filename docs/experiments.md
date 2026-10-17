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

Experiments use the fixed split. E3 and E4 vary the described resolution
and augmentation policy relative to E2. B1 versus E2 also changes the
regression head and optimisation method, so it is a comparison of methods,
not a pure unfreezing ablation. These are saved runs, not repeated trials.
Original improvement-experiment deadline: 12 Oct.

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

Compared against B1 as a complete method. Both use 224px images and the
same documented split/seed, but B1 uses frozen features with Ridge while
E2 adapts the last CNN stage with a neural regression head and Adam.
The result does not isolate unfreezing alone. The recorded setup uses
learning rate 1e-4 and validation-loss checkpoint selection; legacy-log
limitations are recorded in [the Phase 3 artifact supplement](Phase3_MemberB_Artifact_Update.md).

| Configuration | Val MSE | Val MAE | Val R² | Kaggle MSE | vs B1 | Best epoch |
|---|---|---|---|---|---|---|
| B1 frozen + Ridge | 114,853.7 | 249.0 | 0.256 | 114,147.16 | — | — |
| **E2 layer4 fine-tuned** | **102,745.7** | **235.7** | **0.335** | **103,129** | **−10.5%** | 12 of 15 |

Reproduce: `python -m src.exp_finetune`
Original tagged log received: `outputs/logs/phase3_E2_finetune.json`.
The earlier name `phase3_finetune.json` is a historical documentation alias;
the legacy schema is described in the Phase 3 artifact supplement.

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
| **E3 fine-tune 320px** | **99,051.0** | **229.8** | **0.359** | **101,594.15** | **−13.8%** | 10 of 15 |

Reproduce: `python -m src.exp_finetune --img-size 320 --tag E3_320px`

Observations:

- **Resolution and fine-tuning do compose**, adding a further 3.6% on top of
  E2 *on validation*. This is the opposite of what the interim per-epoch
  readings suggested (at epoch 4 E3 was still behind E2), which is a
  reminder not to read a training curve mid-run.
- **On the test set the gain is smaller than validation claimed.** E3 scores
  101,594.15 against a validation estimate of 99,051.0 — the validation
  number is optimistic by 2.6%, and the real improvement over E2 is 1.5%,
  not 3.6%. See the note below on why.

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

**Phase 4 interpretation note:** the historical E4 observations above are
not proof of a learning mechanism or of performance after convergence.
A fixed training budget defines a useful comparison; augmented and
unchanged training-loss values are not directly comparable measurements.
The later [original-artifact supplement](Phase3_MemberB_Artifact_Update.md)
provides the more carefully qualified E2/E4 interpretation.

### Phase 3 summary

| ID | Configuration | Val MSE | Val R² | Kaggle MSE | vs B1 (val) | vs B1 (test) |
|---|---|---|---|---|---|---|
| B1 | Frozen features + Ridge, 224px | 114,853.7 | 0.256 | 114,147.16 | — | — |
| E1 | Frozen features + Ridge, 320px | 111,822.5 | 0.276 | — | −2.6% | — |
| E2 | Fine-tuned, 224px | 102,745.7 | 0.335 | 103,129 | −10.5% | −9.7% |
| **E3** | **Fine-tuned, 320px** | **99,051.0** | **0.359** | **101,594.15** | **−13.8%** | **−11.0%** |
| E4 | Fine-tuned, 224px + augmentation | 108,784.5 | 0.296 | — | −5.3% | — |

### Note: how far the validation score can be trusted

Three configurations have now been submitted to Kaggle, which lets the
validation split be checked against reality:

| Configuration | Val MSE | Kaggle MSE | Validation is off by |
|---|---|---|---|
| B1 frozen + Ridge | 114,853.7 | 114,147.16 | −0.6% (pessimistic) |
| E2 fine-tuned 224px | 102,745.7 | 103,129 | +0.4% (optimistic) |
| E3 fine-tuned 320px | 99,051.0 | 101,594.15 | **+2.6% (optimistic)** |

The first two agreed to within 1%, which was initially read as the 1000-image
validation split being a reliable estimator. **E3 breaks that pattern**, and
the cause is worth stating because it affects how the remaining experiments
should be read.

The bias is a consequence of selecting the best epoch on the validation set.
E3's per-epoch validation loss oscillated between roughly 0.63 and 0.77, and
the selected epoch 10 (0.6309) is partly a lucky draw from that noise. E2's
curve was flatter, so its best epoch was less lucky and its bias is smaller.
The more the validation curve fluctuates, the more optimistic the selected
minimum becomes.

Practical consequences:

- Validation MSE is an **optimistic** estimate whenever the epoch is chosen
  on it, and the optimism grows with the noise of the curve.
- Small differences between configurations (a few percent) should not be
  trusted on validation alone. E3's 3.6% advantage over E2 was really 1.5%.
- A held-out set used for selection is no longer a clean estimate of
  generalisation for the selected model.

### E5 — ensemble of the three fine-tuned runs

Averaging the validation predictions of E2, E3 and E4:

| Configuration | Val MSE | Val R² | Kaggle MSE | vs E3 on test |
|---|---|---|---|---|
| E3 alone (best single) | 99,051.2 | 0.359 | 101,594.15 | — |
| **E2+E3+E4 mean** | **95,850.6** | **0.379** | **96,958.98** | **−4.56%** |

Reproduce: `python -m src.ensemble` (writes `outputs/predictions/submission_ensemble.csv`)

Observations:

- **The ensemble is worth 3.2% on validation and 4.6% on test.** The three
  runs make different errors -- different input resolutions, and one trained
  with augmentation -- so averaging cancels part of the variance.
- **The absolute Kaggle error is not lower than validation error.** The
  reported 96,958.98 is **1,108.38 higher** than 95,850.6, or about **1.16%**
  higher. The relative improvement over E3 is larger on Kaggle, but that is
  a different comparison. These observations do not establish why the
  evaluation scores differ. The Kaggle result is historical and was not
  independently checked in this Phase 4 review.
- Not every subset helps: E2+E4 is 0.66% *worse* than E3 alone, because E4
  is the weakest of the three and its errors are not different enough to
  compensate. Adding E2 to E3+E4 still helps, so all three are used.
- **A fourth, decorrelated model was tested and rejected.** The v3 frozen
  ConvNeXt predictions differ in architecture as well as training, so they
  were the most promising candidate for adding error diversity. Averaging
  them in does not pay:

  | Ensemble | Val MSE | vs E2+E3+E4 |
  |---|---|---|
  | E2+E3+E4 | 95,850.6 | — |
  | + v3 @320 | 95,538.9 | +0.33% |
  | + v3 @224 | 96,065.8 | −0.22% |
  | + both v3 columns | 96,547.8 | −0.73% |

  The best case is +0.33%, which is inside noise, and adding both columns is
  clearly worse. The reason is that v3's validation MSE is 109,155 (frozen
  features, no fine-tuning) against 99,051 for E3, so it is not a peer of the
  models it would join: averaging in a substantially weaker model loses more
  through dilution than it gains through diversity. Diversity only helps
  between models of comparable accuracy. This does not change the selection,
  and it was the last ensemble variant tried. Reproduce with the
  `val_predictions_*.csv` and `v3_val_predictions.csv` files under
  `outputs/predictions/`.

### Phase 3 final state

| Configuration | Val MSE | Kaggle MSE | vs B1 on test |
|---|---|---|---|
| B1 frozen ResNet-50 + Ridge | 114,853.7 | 114,147.16 | — |
| E3 fine-tuned 320px | 99,051.0 | 101,594.15 | −11.0% |
| **E2+E3+E4 ensemble** | 95,850.6 | **96,958.98** | **−15.1%** |

## Phase 4 — model comparison and selection

**Reviewed:** 16 October 2026, against main `90959c4`. This section follows
the revised Phase 1–8 plan. It does not introduce another training experiment.

**Recommended configuration:** retain the existing equal-weight ensemble of
`E2_finetune`, `E3_320px`, and `E4_augment`. The historical validation MSE is
**95,850.6**, compared with **99,051.0** for the E3 run table and **114,853.7**
for B1. From those reported values, the reductions are **3.23%** and
**16.55%**, respectively. Selection is based on validation MSE, not Kaggle
scores. This is the best documented core candidate, not a claim about all
possible models or independently verified statistical superiority.

The [complete comparison](tables/Phase4_Comparison.md) includes all eleven
numerical baseline/resolution/core configurations recorded above. The
[selection record](Phase4_Model_Selection.md) explains the decision,
limitations, remaining checks, and Member A/B review. The
[historical figure](figures/phase4_historical_validation_mse.svg) is explicitly
labelled as reported evidence.

**Evidence status:** source review, historical comparison, and the Member A
selection write-up are prepared. Member B completed the real CSV audit on
16 October 2026: **25 PASS / 0 FAIL / 4 PENDING**, recorded in
`docs/evidence/phase4_run_20261016T124243_097811Z/`. Its pending items and
actual member review records are explained in `Phase4_Model_Selection.md`.
To reproduce the numerical checks:

```powershell
py -3 -m unittest discover -s tests -p "test_verify_phase4_selection.py" -v
py -3 tools/verify_phase4_selection.py --repo-root .
```

The checker aligns validation predictions by `imageid`, verifies the pinned
inputs and fixed split, recomputes MSE/MAE/R2, evaluates both V3 columns,
and checks the stored 3,000-row ensemble arithmetic. It writes a **new**
`docs/evidence/phase4_run_<timestamp>/` directory and never overwrites
training data, original predictions, checkpoints, or existing evidence.
A numerical candidate is not automatically an approved team decision.

**Handoff:** Member B's audit names the E5 `validation_ensemble.csv` in that
evidence directory for Phase 5. The new Member A analysis below uses that
exact file. E3's existing analysis remains a single-model reference, and
actual team approval stays in the Phase 4 selection record. Model loading
and full reproducibility remain
Phase 7 work; the E2/E4 artifact handoff already recorded in Phase 3 is not
reopened simply because large files are excluded from GitHub.

## Phase 5 — error analysis

### E5 ensemble — Member A, executed 17 October 2026

The current main snapshot `ae748264` contains the merged Phase 4 evidence.
The new CSV-only analysis uses its **one-decimal E5 validation export**,
checks all source identities and true labels, and confirms each prediction
against the equal mean of E2/E3/E4. No model was retrained or loaded.

| Metric | E5 | E3 reference, recomputed from its CSV |
|---|---:|---:|
| Validation MSE | 95,852.002676 | 99,051.197058 |
| MAE | 226.423020 | 229.764228 |
| Mean signed bias | +6.912374 | +11.429074 |
| Highest-decile MSE | 509,349.703033 | 488,255.361991 |
| Highest-decile share of total SSE | 53.14% | 49.29% |

E5 improves overall MSE by **3.23%**, while highest-decile MSE becomes
**4.32% worse**. Absolute error falls on 515 images and rises on 485.
The overall gain must therefore be reported together with the remaining
price-dependent errors. A near-zero overall bias does not remove those errors.

The lowest 30% of prices have mean bias +222.03 and contribute 21.19% of
total squared error. The middle 50% have bias +46.27 and contribute 16.81%.
The ninth decile contributes 8.86%; the highest decile contributes 53.14%
and has bias −648.48. All price/error values use USD 1,000; MSE uses its
squared unit. These figures use the existing export precision and a single
fixed split, not independent test labels or new full-precision predictions.

[The full Phase 5 report](Phase5_Error_Analysis.md) contains interpretation,
the deterministic nine-example selection, limitations and future-work
proposals. [Executed evidence](evidence/phase5_run_20261017/analysis.md)
contains exact numerical tables and source identities. The original photos
require local rendering and visual review; no image-specific cause is
asserted from numerical errors alone.

```powershell
py -3 -m unittest discover -s tests -p "test_analyze_phase5_errors.py" -v
py -3 tools/analyze_phase5_errors.py --repo-root .
```

### Historical E3 reference

The historical analysis below is for **E3, the best documented single core
model**, not the selected ensemble. It is retained as historical Phase 5
reference evidence; it must not be relabelled as E5 analysis. The legacy
entry point is `python -m src.error_analysis --tag E3_320px`; it reruns
inference and replaces that run's prediction CSV. Use a separate reproduction
copy when running it, so the pinned Phase 4/5 source CSVs remain available.
Historical raw output is in
`outputs/logs/error_analysis_E3_320px.json`; the per-image predictions are in
`outputs/predictions/val_predictions_E3_320px.csv`, sorted by absolute error.

### Historical E3 predictions across price groups

Predictions bucketed by the true price, 1000 validation images:

| True price | n | true mean | pred mean | MSE | **bias** |
|---|---|---|---|---|---|
| < 299 | 93 | 249.4 | 476.1 | 75,091 | **+226.8** |
| 299–397 | 107 | 344.9 | 559.8 | 66,758 | **+214.9** |
| 397–480 | 100 | 435.2 | 621.7 | 68,268 | +186.5 |
| 480–560 | 100 | 520.7 | 640.4 | 44,212 | +119.7 |
| 560–640 | 100 | 597.1 | 722.6 | 46,838 | +125.6 |
| 640–719 | 100 | 679.3 | 702.2 | 33,630 | +22.8 |
| 719–799 | 92 | 756.0 | 781.5 | 34,230 | +25.5 |
| 799–941 | 108 | 861.7 | 839.8 | 43,585 | −21.9 |
| 941–1300 | 100 | 1089.7 | 927.1 | 89,477 | **−162.6** |
| 1300+ | 100 | 1634.1 | 1015.7 | **488,256** | **−618.4** |

The cheapest groups are overestimated and the most expensive group is
underestimated. Only the three groups approximately 640–941 have small
signed biases in this historical table: the 480–560 and 560–640 groups
still have biases +119.7 and +125.6. The worst historical single prediction
is a house with true price 1995 and predicted price approximately 542.

The group means show a narrower range of predictions than true prices.
This is consistent with central-price predictions under uncertainty, but
the observed pattern alone cannot establish whether the cause is missing
visual information, representation limits, fitting choices or the objective.

### Almost half the error comes from the top decile of prices

| True price | share of total squared error |
|---|---|
| bottom 3 deciles (< 480) | 21.2% |
| middle 5 deciles (approximately 480–941) | 20.5% |
| top decile (1300+) | **49.3%** |
| top 2 deciles (941+) | **58.3%** |

The most expensive tenth of houses accounts for **just under half** of the
historical E3 squared error, and the top fifth for well over half. This
makes the expensive tail a useful target for further investigation, while
errors in the other groups also contribute to the required overall metric.

### Overfitting

From the recorded training history of E3 (best epoch 10 of 14):

| | |
|---|---|
| train loss at best epoch | 0.0684 |
| validation loss at best epoch | 0.6309 |
| ratio (train / validation) | **0.108** |

The recorded training loss is about one ninth of the validation loss. This
gap suggests a need to investigate generalisation and the comparability of
training/evaluation conditions; it does not alone prove memorisation. The
validation loss ranged from 0.6309 to 0.7666. Selecting the best epoch on
the same validation set can make the selected score optimistic, but the
size of that effect is not measured here. This is E3 history, not an E5
training curve.

Residuals overall: mean +11.4, standard deviation 314.5, range −1452.9 to
+768.6. Just under 40% of predictions are below the true price.

### What this implies for the report's future-work section

The historical analysis suggests hypotheses for controlled comparisons,
not a demonstrated preference for changing the objective over the architecture:

- compare predicting log price with the existing target, while evaluating
  both in original price units;
- compare other training objectives or price-band approaches, recognising
  that their optimisation goals may differ from the required MSE;
- evaluate a specified sampling or weighting policy and report its impact
  on every price group;
- compare a revised visual representation where image inspection suggests
  missing spatial information.

None of these alternatives was validated by this Phase 5 analysis, and none
is guaranteed to improve original-unit MSE.
