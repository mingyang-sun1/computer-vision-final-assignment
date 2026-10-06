"""Generate notebooks/kaggle_v3_features.ipynb.

Question this answers: does a better backbone help once the overfitting is
taken out of the picture?

Every fine-tuned model so far has overfitted badly -- the training loss
falls to a fraction of the validation loss -- because 7000 images is small
for adapting 15-25M parameters. Freezing the backbone removes that failure
mode entirely: there is nothing to overfit, and the only capacity is a
ridge regression on top. v1 measured frozen ResNet-50 features at 114,853.7
validation MSE; this measures frozen ConvNeXt-Tiny features under the same
conditions, at two resolutions, using the aspect-correct preprocessing from
v2.

Run it on a Kaggle Notebook with a GPU accelerator enabled and the
competition attached as an input, then Run All. About 5 minutes.

    python notebooks/build_features_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "kaggle_v3_features.ipynb"

MARKDOWN = """# COMP90086 house prices — v3: frozen ConvNeXt features

Answers one question: **does a stronger backbone help when the overfitting
is removed?**

Every fine-tuned model so far overfitted badly — training loss fell to a
fraction of validation loss — because 7000 images is too few for adapting
15–25M parameters. Freezing the backbone removes that failure mode: there is
nothing to overfit, and the only capacity is a ridge regression on top.

* v1 reference: frozen **ResNet-50** features + ridge (squashed images) → **114,853.7**
* this notebook: frozen **ConvNeXt-Tiny** features + ridge, aspect-correct

Setup: new Notebook → import this file → **Add Input → Competition** →
**Settings → Accelerator → GPU** → Run All. Needs **Internet on** the first
time, to download the ConvNeXt weights.
"""

CELL_DATA = '''import os, glob, pandas as pd
hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
assert hits, "train.csv not found -- did you attach the competition as an input?"
DATA_ROOT = os.path.dirname(hits[0])
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")
train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
test_ids = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))["imageid"].astype(str).tolist()
print("data root :", DATA_ROOT)
print("train", len(train_df), " test", len(test_ids))
print("GPU:", [d.name for d in __import__("tensorflow").config.list_physical_devices("GPU")])
'''

CELL_EXTRACT = '''import json, time
import numpy as np, tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
BATCH = 64
RESOLUTIONS = [224, 320]

tf.random.set_seed(SEED); np.random.seed(SEED)

# same frozen split as every other experiment
bins = pd.qcut(train_df["price"], q=10, labels=False, duplicates="drop")
tr_df, va_df = train_test_split(train_df, test_size=0.125, random_state=SEED,
                                shuffle=True, stratify=bins)
tr_df, va_df = tr_df.reset_index(drop=True), va_df.reset_index(drop=True)
Y_MEAN, Y_STD = float(tr_df.price.mean()), float(tr_df.price.std())
Y_MIN, Y_MAX = float(tr_df.price.min()), float(tr_df.price.max())
print("split: train", len(tr_df), "val", len(va_df))

# frozen ConvNeXt-Tiny, global average pooled
extractor = keras.applications.ConvNeXtTiny(weights="imagenet",
                                            include_top=False, pooling="avg")
extractor.trainable = False

def prep(path, size):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize_with_pad(img, size, size)      # aspect preserved
    return keras.applications.convnext.preprocess_input(img)

def paths_ds(paths, size):
    return (tf.data.Dataset.from_tensor_slices(paths)
            .map(lambda p: prep(p, size), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH).prefetch(tf.data.AUTOTUNE))

def feats(ids, img_dir, size):
    paths = [os.path.join(img_dir, str(i)) for i in ids]
    return extractor.predict(paths_ds(paths, size), verbose=0)

tr_paths_ids = tr_df["imageid"].tolist()
va_paths_ids = va_df["imageid"].tolist()
y_tr = ((tr_df.price.to_numpy(np.float32) - Y_MEAN) / Y_STD)
y_va_true = va_df.price.to_numpy()

results, val_preds = {}, {}
for size in RESOLUTIONS:
    t = time.time()
    Xtr = feats(tr_paths_ids, TRAIN_IMG_DIR, size)
    Xva = feats(va_paths_ids, TRAIN_IMG_DIR, size)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    model = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xtr, y_tr)
    p = np.clip(model.predict(Xva) * Y_STD + Y_MEAN, Y_MIN, Y_MAX)
    mse = float(np.mean((y_va_true - p) ** 2))
    results[size] = {"mse": mse,
                     "mae": float(np.mean(np.abs(y_va_true - p))),
                     "r2": float(1 - mse / np.var(y_va_true)),
                     "alpha": float(model.alpha_)}
    val_preds[size] = p
    print(f"{size}px  MSE {mse:,.1f}  MAE {results[size]['mae']:,.1f}  "
          f"R2 {results[size]['r2']:.3f}   ({time.time()-t:.0f}s)")
print()
print("v1 reference (frozen ResNet-50, squashed): MSE 114,853.7  R2 0.256")
'''

CELL_PREDICT = '''BEST = min(results, key=lambda s: results[s]["mse"])
print(f"best resolution: {BEST}px  (MSE {results[BEST]['mse']:,.1f})")

# refit on train at the winning resolution, then predict test
Xtr = feats(tr_paths_ids, TRAIN_IMG_DIR, BEST)
sc = StandardScaler().fit(Xtr)
ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(Xtr), y_tr)

Xte = feats(test_ids, TEST_IMG_DIR, BEST)
prices = np.clip(ridge.predict(sc.transform(Xte)) * Y_STD + Y_MEAN, Y_MIN, Y_MAX)

sub = pd.DataFrame({"imageid": test_ids, "price": np.round(prices, 1)})
sub.to_csv("submission_v3.csv", index=False)
assert len(sub) == len(test_ids) and list(sub.columns) == ["imageid", "price"]
assert sub.price.notna().all()
print("wrote submission_v3.csv")
print(sub.head())
print("range %.1f - %.1f  mean %.1f" % (prices.min(), prices.max(), prices.mean()))
'''

CELL_SAVE = '''# Save the validation predictions so this run can be ensembled with the
# fine-tuned models, and the log so it can be cited without a re-run.
va_out = pd.DataFrame({"imageid": va_df["imageid"].values,
                       "price_true": y_va_true})
for size in RESOLUTIONS:
    va_out[f"price_pred_{size}"] = val_preds[size]
va_out.to_csv("v3_val_predictions.csv", index=False)

with open("v3_results.json", "w") as f:
    json.dump({"backbone": "ConvNeXtTiny (frozen, GAP)",
               "preprocessing": "resize_with_pad -- aspect preserved",
               "seed": SEED, "results": {str(k): v for k, v in results.items()},
               "best_resolution": int(BEST),
               "v1_reference_frozen_resnet50": {"mse": 114853.7, "r2": 0.256}},
              f, indent=2)
print("wrote v3_val_predictions.csv and v3_results.json")
'''

CELLS = [("markdown", MARKDOWN), ("code", CELL_DATA),
         ("code", CELL_EXTRACT), ("code", CELL_PREDICT), ("code", CELL_SAVE)]


def build() -> dict:
    return {
        "cells": [
            {"cell_type": k, "metadata": {}, "source": [ln + "\n" for ln in s.split("\n")],
             **({"outputs": [], "execution_count": None} if k == "code" else {})}
            for k, s in CELLS
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
