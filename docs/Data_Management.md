# Data Management

COMP90086 Computer Vision 2026 Semester 2 — Final Project (Predicting House Prices)
Group: Mingyang Sun (1657392) & Yikai Qian (1722421)

> Status: **skeleton**. Member A owns the prose; Member B has completed the
> structural items (folder layout, split, path handling) — those are marked
> `[done]` and are backed by code in `src/data.py`. Fill in the `[TODO A]`
> items and review before the Phase 1 deadline (7 Oct).

## 1. Dataset source

`[TODO A]` State that only the course-provided data is used, and name the files:
`train.csv`, `test.csv`, `train/`, `test/` — 8000 labelled training images and
3000 unlabelled test images, prices in units of $1000 USD.

Explicitly record the two restrictions we are bound by:

- No additional images or metadata from the original *House Prices and Images –
  SoCal* dataset beyond what the course provided.
- No pretrained models that were trained on this task or this dataset, and no
  pretrained MLLMs.

## 2. Folder structure

`[done]` Paths are centralised in `config.py`; nothing in the code hard-codes a
personal absolute path. The course images live **outside** this repository
(see `.gitignore`).

```
computer-vision-final-assignment/
├── config.py                 # paths, seed, split settings
├── src/
│   └── data.py               # data loading + frozen split
├── data/
│   └── splits/               # committed: the frozen split (see §3)
├── docs/
│   └── Data_Management.md    # this file
└── outputs/                  # not committed: models, logs, predictions

<dataset root>/               # course-provided, NOT committed
├── train.csv, test.csv
└── train/, test/
```

The dataset location is resolved by `config.DATA_ROOT`, which reads the
`CV_DATA_ROOT` environment variable and otherwise falls back to
`../comp-90086-2026/house_dataset`.

## 3. Train / validation / test separation

`[done]` One split is created once and then frozen:

| Split | Source | Size | File |
|---|---|---|---|
| Train | `train.csv` | 7000 | `data/splits/train_split.csv` |
| Validation | `train.csv` | 1000 | `data/splits/val_split.csv` |
| Test | course test set | 3000 | no labels; predictions only |

`[TODO A]` Justify the split here. Points worth making:

- Stratified by **price decile**, not a plain random split, because the price
  distribution is right-skewed (median 640 < mean 721, range 195–2000). A plain
  random split can leave the validation tails thin, which would make the
  per-price-range error analysis unreliable.
- Fixed seed (`config.SEED = 42`); the exact row counts and price statistics of
  each partition are recorded in `data/splits/split_meta.json`.
- The split CSVs are committed, so every experiment loads the *same* file
  rather than re-deriving a split. Comparability across experiments depends on
  this.

## 4. Leakage prevention

`[TODO A]` State the rules; they are already enforced in code where possible.

- The test images have no labels in this repository and are used **only** to
  generate the final Kaggle predictions — never for training, hyper-parameter
  selection, or model selection.
- Data augmentation is applied to the **training partition only**.
- Any data-dependent preprocessing (normalisation statistics, etc.) is fitted
  on the training partition alone and then applied unchanged to validation and
  test.
- Model selection is decided on **validation MSE only**.
- `[done]` `src/data.py::verify()` asserts that train and validation share no
  image IDs and that together they cover every labelled training image.

## 5. Reproducibility

`[TODO A]` Record, per experiment: random seed, model configuration,
preprocessing, augmentation settings, learning rate, epochs, validation MSE,
and the saved checkpoint. Reference the log files under `outputs/logs/`.

`[done]` Global seed and split settings live in `config.py` and are printed by
`python config.py`.

## 6. Submission and data handling

`[TODO A]` Confirm the code submission contains model code, the Kaggle-format
prediction CSV, a README, and any files needed to reproduce the results — and
**not** the course-provided images.

`[done]` `.gitignore` excludes the dataset, all image files, and `outputs/`, so
the images cannot be committed by accident. The README explains where the code
expects to find them.
