# COMP90086 Final Project — Predicting House Prices

Computer Vision, 2026 Semester 2.
Group: Mingyang Sun (1657392) & Yikai Qian (1722421).

Predicting the sale price of a home (in units of $1000 USD) from a photograph
of its exterior. Evaluated by mean squared error on the Kaggle test set.

## Environment

Developed and tested with:

- Python 3.11+
- TensorFlow 2.20 / Keras 3.11
- numpy, pandas, scikit-learn, pillow, matplotlib (`requirements.txt`)

```
pip install -r requirements.txt
```

No GPU is required for the frozen-feature experiments. See
`docs/Data_Management.md` for the compute plan.

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

## Quick start

```bash
# 1. Check the resolved paths and settings
python config.py

# 2. Build and freeze the train/validation split (writes data/splits/)
python -m src.data
```

Step 2 writes `train_split.csv` (7000 rows), `val_split.csv` (1000 rows), and
`split_meta.json`. These files are committed: **do not regenerate them.**
Every experiment loads the frozen split so results stay comparable.

## Repository layout

```
config.py                   paths, seed, split settings
src/data.py                 data loading + frozen split + verification
data/splits/                the committed split
docs/Data_Management.md     data management record
outputs/                    models, logs, predictions (not committed)
```

## Rules we are bound by

- Only the course-provided data. No additional images or metadata from the
  original *House Prices and Images – SoCal* dataset.
- No pretrained models trained on this task/dataset, and no pretrained MLLMs.
  General-purpose pretrained features (e.g. ImageNet weights) are allowed.
- The test set has no labels here. It is used only to produce the final Kaggle
  predictions — never for training, tuning, or model selection.
- Course-provided images must not be included in the submitted code package.
