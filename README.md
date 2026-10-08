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

The final model is **15.1% better than the baseline** on the test set.

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
src/error_analysis.py        error analysis by price bucket, overfitting measures
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

On the best single model (E3), bucketing validation predictions by true
price:

| True price | bias (pred − true) | share of total squared error |
|---|---|---|
| bottom 3 deciles (< 480) | +186 to +227 | 21.2% |
| middle 4 deciles (480–941) | −22 to +26 | 20.5% |
| top decile (1300+) | **−618** | **49.3%** |

The model over-predicts cheap houses and badly under-predicts expensive
ones — the behaviour MSE induces on a right-skewed target, where the optimal
response to uncertainty is to stay near the mean. **Just under half the
total error comes from the most expensive tenth of houses**, so performance
in the middle of the distribution is nearly irrelevant to the score. The
report's future-work section follows from this: the objective, not the
architecture, is the thing to change.

A methodological note that the report also covers: validation MSE is
*optimistic* whenever the epoch is chosen on it, and the optimism grows with
the noise of the per-epoch curve. E3's validation score was 2.6% optimistic
on test. Ensembling reverses this — its test result beat its validation
estimate — because averaging reduces the variance a single 1000-image split
measures badly.

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
