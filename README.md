# COMP90086 Final Project — Predicting House Prices

Computer Vision, 2026 Semester 2.
Group: Mingyang Sun (1657392) & Yikai Qian (1722421).

Predicting the sale price of a home (in units of $1000 USD) from a photograph
of its exterior, evaluated by mean squared error on the Kaggle test set.

## Result

| | Validation MSE | Kaggle MSE |
|---|---|---|
| Predicting the training mean | 154,431.7 | — |
| Frozen ResNet-50 features + Ridge (baseline) | 114,853.7 | 114,147.16 |
| **Final model: ensemble of three fine-tuned runs** | **95,850.6** | **96,958.98** |

The recorded Kaggle MSE is **15.1% lower than the baseline**. This is a
historical reported result, not a newly verified score. Model selection
uses validation evidence; see the Phase 4 record below.

## The final model

An average of the predictions of three partial fine-tunings of an
ImageNet-pretrained ResNet-50:

| Run | Configuration | Validation MSE |
|---|---|---|
| E2 | fine-tune layer4 + head, 224px | 102,745.7 |
| E3 | fine-tune layer4 + head, 320px | 99,051.0 |
| E4 | E2 plus augmentation | 108,784.5 |
| **E2+E3+E4** | simple mean of the three | **95,850.6** |

Only the last stage (layer4, 14.9M of the backbone's 23.5M parameters) plus a
new regression head is trained. The three runs differ in resolution and in
whether augmentation was used, so they make partly different errors on the
same images; averaging cancels the independent part.

The final submission is `outputs/predictions/submission_ensemble.csv`.

## Phase 4 — comparison and selection

This follows the **revised Phase 1–8 plan**. See the
[Phase 4 selection record](docs/Phase4_Model_Selection.md),
[complete comparison](docs/tables/Phase4_Comparison.md), and
[updated experiment log](docs/experiments.md#phase-4--model-comparison-and-selection).
The source review recommends retaining the existing E2/E3/E4 equal mean;
it does not claim a new training run or completed A/B approval.

```powershell
py -3 -m unittest discover -s tests -p "test_verify_phase4_selection.py" -v
py -3 tools/verify_phase4_selection.py --repo-root .
```

The checker uses Python 3.10+ and the standard library only. It needs the
committed split/prediction CSVs, not images, GPU packages, or checkpoints.
Outputs go into a **new** `docs/evidence/phase4_run_<timestamp>/` folder.
Historical scores and newly computed CSV scores remain separate. Member B
completed the full CSV audit on 16 October 2026; its committed evidence is in
`docs/evidence/phase4_run_20261016T124243_097811Z/`. The exported one-decimal
ensemble has validation MSE **95,852.00**; the unrounded mean of the stored
component exports has MSE **95,850.62**. These are precision variants of the
stored predictions, not new training runs. See the selection record for the
remaining review and reproducibility items. No existing submission is overwritten.

## Phase 5 — Member A error analysis

The ensemble analysis is complete for the **1,000 committed validation
predictions**. See [Phase 5 results and interpretation](docs/Phase5_Error_Analysis.md),
[executed numerical evidence](docs/evidence/phase5_run_20261017/analysis.md), and
[the concise report section](docs/report/Phase5_Error_Analysis_Section.md).

```powershell
py -3 -m unittest discover -s tests -p "test_analyze_phase5_errors.py" -v
py -3 tools/analyze_phase5_errors.py --repo-root .
```

Python 3.10+ and its standard library are sufficient for the numerical
analysis. The command uses the existing E5 CSV, validates its IDs and labels
against the saved split, verifies the E2/E3/E4 mean, and writes a new evidence
directory. It does not need TensorFlow, model files, a GPU or the raw images.

The package also contains two numerical figures. To regenerate them from
the committed Phase 5 evidence, using a new output directory:

```powershell
py -3 -m pip install matplotlib pillow
py -3 tools/plot_phase5_errors.py --analysis-dir docs/evidence/phase5_run_20261017 --output-dir outputs/phase5_figures
```

Nine example IDs are selected deterministically: three closest predictions,
three largest overestimates and three largest underestimates. The original
photos were not available in the preparation environment, so image-specific
observations are still to be reviewed. Generate the photo panel locally with:

```powershell
py -3 tools/plot_phase5_errors.py --analysis-dir docs/evidence/phase5_run_20261017 --examples-only --image-root "$env:CV_DATA_ROOT\train"
```

Set `CV_DATA_ROOT` to the course dataset folder first. Photo panels and
`visual_review.csv` go to the ignored `outputs/phase5_examples/` directory.
The panel is intended for the written report; course images stay out of the
code submission. An existing output directory is never silently replaced.

## Environment

Developed and tested with Python 3.11+, TensorFlow 2.20 / Keras 3.11, plus
numpy, pandas, scikit-learn, pillow and matplotlib (`requirements.txt`).

```
pip install -r requirements.txt
```

**No GPU was used.** Everything ran on CPU. This is a real constraint rather
than a preference: feature extraction takes about 3 minutes for 9000 images,
and fine-tuning runs at roughly 3 minutes per epoch. `notebooks/` contains
GPU-based runners for Kaggle for the experiments that were too slow to run
locally (ConvNeXt).

## Dataset

The course-provided images are **not** stored in this repository (see
`.gitignore`). The code finds them via `config.DATA_ROOT`, which reads the
`CV_DATA_ROOT` environment variable:

```bash
export CV_DATA_ROOT=/path/to/house_dataset
```

Without that variable it falls back to `../comp-90086-2026/house_dataset`
relative to this repository. Either way the folder must contain:

```
house_dataset/
├── train.csv        # imageid, price   (8000 rows)
├── test.csv         # imageid         (3000 rows)
├── train/           # 8000 .jpg
└── test/            # 3000 .jpg
```

## Reproducing the result

```bash
python config.py                                          # check resolved paths
python -m src.data                                        # verify the split (read-only)
python -m src.data --rebuild                              # only if you mean to replace it

python -m src.baseline                                    # B0/B1/B2: frozen features

python -m src.exp_finetune --img-size 224 --tag E2_finetune
python -m src.exp_finetune --img-size 320 --tag E3_320px
python -m src.exp_finetune --augment      --tag E4_augment

for t in E2_finetune E3_320px E4_augment; do
  python -m src.predict --model finetuned --tag $t
done

python -m src.ensemble                                    # final submission
```

### Known limitation: results are not bit-reproducible

TensorFlow on CPU is not deterministic across runs — thread scheduling
changes floating-point accumulation order. Re-training will produce a
similar but not identical model, and validation MSE will differ by a few
percent. The scripts, split, seed and hyper-parameters are all fixed, but
the numbers in `docs/experiments.md` cannot be recovered exactly by
re-running.

The committed `outputs/predictions/submission_ensemble.csv` is a fixed file
and is the submission the Kaggle score refers to. The trained model files
(~220 MB each) are **not** in this repository: they exceed GitHub's file
size limit, and they are regenerable from the commands above.

## Repository layout

```
config.py                    paths, seed, split and training settings
src/data.py                  data loading + the frozen train/validation split
src/features.py              frozen-backbone feature extraction, cached to .npy
src/baseline.py              B0 (mean), B1 (ridge), B2 (MLP) on frozen features
src/exp_resolution.py        E1 — input resolution sweep
src/exp_finetune.py          E2/E3/E4 — partial fine-tuning
src/train_v2.py              v2 — ConvNeXt backbone (did not help; see below)
src/exp_frozen_backbone.py   v3 — frozen ResNet-50 vs ConvNeXt comparison
src/ensemble.py              the final model
src/predict.py               Kaggle-format submission generation
src/error_analysis.py        historical single-run inference and training-gap analysis
tools/analyze_phase5_errors.py audited E5 validation errors, price buckets and examples
tools/plot_phase5_errors.py  numerical plots and optional local photo panels
src/metrics.py               MSE / MAE / R2
data/splits/                 the committed split — do not regenerate
docs/experiments.md          full experiment log with results and analysis
docs/Data_Management.md      data management record
notebooks/                   self-contained Kaggle GPU runners
outputs/predictions/         all submission csvs, including the final one
```

## Method summary

Full detail, including the reasoning behind each choice, is in
`docs/experiments.md`. In brief:

**Fixed setup.** One stratified split (7000 train / 1000 validation, seed
42, stratified by price decile because the price distribution is
right-skewed and a plain random split thins out the tails). Every experiment
uses the same split. Model selection is on validation MSE only; the test set
is used solely to produce the final predictions. The split is committed so
it cannot drift.

**Baseline.** Frozen ImageNet ResNet-50 features (2048-d, global average
pooled) with a Ridge probe, and separately a small MLP. Ridge wins: the
MLP early-stopped after 19 epochs, suggesting the price signal is close to
linearly available in the frozen representation.

**What helped.** Partial fine-tuning of layer4 (−13.8% against the frozen
baseline) and ensembling the fine-tuned runs (−3.2% on validation, −4.6% on
test).

**What did not help**, and is reported as such rather than hidden:

- *Higher input resolution alone*: −2.6%. The two heads disagree about the
  optimum, and 400px MAE is best of all — there is more signal at higher
  resolution, but a linear readout cannot exploit it.
- *Augmentation at a fixed 15-epoch budget*: 5.9% **worse**. The mechanism
  is visible in the training loss — augmentation makes each epoch a harder
  task, so the model learns more slowly, and E4 had not converged when the
  budget ran out. The comparison is confounded by the budget and is stated
  as such.
- *ConvNeXt backbone, fine-tuned*: 8.9% worse than fine-tuned ResNet-50,
  because 7000 images cannot support adapting a larger network.
- *Aspect-ratio correction*: the fix traded geometric distortion for a 31%
  loss of effective pixels, and the two roughly cancelled.
- *Cross-backbone ensembling*: +0.33%, within noise.

## Error analysis

The **E5 ensemble**, using its exported one-decimal predictions, has MSE
**95,852.00**, MAE **226.42** and mean signed error **+6.91**. Prices, MAE
and signed errors are in USD 1,000; MSE is in squared USD-1,000 units.

| True-price group | Images | Mean bias (pred − true) | Share of total squared error |
|---|---:|---:|---:|
| Lowest 30%, below 479.997 | 300 | +222.03 | 21.19% |
| Middle 50%, 479.997 to below 940.92 | 500 | +46.27 | 16.81% |
| Ninth decile, 940.92 to below 1299.91 | 100 | −179.81 | 8.86% |
| Highest 10%, 1299.91 and above | 100 | **−648.48** | **53.14%** |

The small overall bias hides opposite errors at the two ends of the price
range. The ensemble improves overall MSE by **3.23%** relative to E3's
recomputed CSV score, but **increases highest-decile MSE by 4.32%**. It
therefore does not improve every price group. The lowest deciles remain
overestimated and the highest decile remains substantially underestimated.

![E5 error analysis by true-price decile](docs/figures/phase5_price_buckets.svg)

These are descriptive results from one validation split. They do not isolate
the cause of the errors or establish that changing the objective is better
than changing the representation. The [full Phase 5 analysis](docs/Phase5_Error_Analysis.md)
includes exact bin boundaries, examples, limitations and testable future-work
suggestions. Historical E3 results remain labelled E3 in the experiment log.

Selecting checkpoints and model configurations on a validation set can
make the selected validation score optimistic; a difference between one
validation score and a Kaggle score alone does not prove the cause.
For the ensemble, the reported Kaggle MSE (96,958.98) is **higher** than the
reported validation MSE (95,850.6), by about **1.16%**. Its relative gain over
E3 is larger on Kaggle, but its absolute Kaggle error did not beat its
validation error. The Phase 4 review does not independently verify Kaggle
scores.

## Rules the project is bound by

- Only the course-provided data. No additional images or metadata from the
  original *House Prices and Images – SoCal* dataset.
- No pretrained models trained on this task/dataset, and no pretrained
  MLLMs. General-purpose pretrained features (ImageNet weights) are allowed
  and are what the project uses throughout.
- The test set has no labels here. It is used only to produce the final
  Kaggle predictions — never for training, tuning, or model selection.
- Course-provided images are not included in this repository or the code
  submission.
