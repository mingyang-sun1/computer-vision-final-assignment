"""Generate notebooks/kaggle_v2.ipynb.

The notebook is self-contained so it can be uploaded to Kaggle and run
without cloning the (private) repository or handling any credentials. This
script is its source of truth, so edit here and regenerate rather than
editing the .ipynb by hand:

    python notebooks/build_kaggle_notebook.py

The pipeline it contains mirrors src/train_v2.py, which is the version that
lives in the repository and is the one to cite in the report.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "kaggle_v2.ipynb"

MARKDOWN_INTRO = """# COMP90086 house prices — v2 pipeline (ConvNeXt, aspect-correct)

Run this on a Kaggle Notebook with a **GPU accelerator** enabled.

Setup:
1. Create a new Notebook (Code).
2. **File → Import Notebook** and upload this `.ipynb`, or paste the cells.
3. On the right-hand panel: **Add Input → Competition → the COMP90086 competition**
   so the images are mounted under `/kaggle/input/`.
4. **Settings → Accelerator → GPU T4**.
5. **Run All**.

What v2 changes from v1, both motivated by where v1 failed:

* **Aspect ratio.** v1 resized every image to a square, which squashed
  400x277 by 1.44x vertically and 400x233 by 1.71x — the same building at
  inconsistent geometry. `resize_with_pad` scales by the longer side and
  pads, so nothing is distorted and nothing is cropped away.
* **Backbone.** ResNet-50 (2015) to ConvNeXt-Tiny (2022). About 3x slower
  per image on CPU, which is why this runs on a GPU.

Everything else is unchanged from v1 so the comparison stays clean: same
frozen split (rebuilt from the same seed and code), same target
standardisation, same clipping of predictions to the training price range.
"""

CELL_FIND_DATA = '''import os, glob, pandas as pd

# Locate the competition data wherever Kaggle mounted it.
hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
assert hits, "train.csv not found under /kaggle/input -- did you attach the competition as an input?"
TRAIN_CSV = hits[0]
DATA_ROOT = os.path.dirname(TRAIN_CSV)
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_ids = pd.read_csv(TEST_CSV)["imageid"].astype(str).tolist()
print("data root :", DATA_ROOT)
print("train rows:", len(train_df), " test rows:", len(test_ids))
print("train imgs:", len(os.listdir(TRAIN_IMG_DIR)), " test imgs:", len(os.listdir(TEST_IMG_DIR)))
'''

CELL_PIPELINE = '''import json, random
import numpy as np, tensorflow as tf
from tensorflow import keras

SEED = 42
BATCH = 32
EPOCHS = 20
PATIENCE = 6
BACKBONE = "convnext_tiny"
IMG_SIZE = 224
LAST_STAGE = {"convnext_tiny": "convnext_tiny_stage_3",
              "convnext_small": "convnext_small_stage_3",
              "convnext_base": "convnext_base_stage_3"}

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ---- the frozen split, rebuilt exactly as v1 --do not change --------------
from sklearn.model_selection import train_test_split
bins = pd.qcut(train_df["price"], q=10, labels=False, duplicates="drop")
tr_df, va_df = train_test_split(train_df, test_size=0.125, random_state=SEED,
                                shuffle=True, stratify=bins)
tr_df, va_df = tr_df.reset_index(drop=True), va_df.reset_index(drop=True)
print("split: train", len(tr_df), "val", len(va_df))

Y_MEAN, Y_STD = float(tr_df.price.mean()), float(tr_df.price.std())
Y_MIN, Y_MAX = float(tr_df.price.min()), float(tr_df.price.max())

# ---- preprocessing: scale by the longer side, pad -- never squash --------
def prep(path, augmenter=None):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
    if augmenter is not None:
        img = augmenter(img, training=True)
    return keras.applications.convnext.preprocess_input(img)

def augmenter():
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.15, value_range=(0.0, 255.0)),
        keras.layers.RandomContrast(0.15, value_range=(0.0, 255.0)),
    ])

def dataset(df, img_dir, aug=None, shuffle=False):
    paths = [os.path.join(img_dir, str(i)) for i in df["imageid"]]
    labels = ((df["price"].to_numpy(np.float32) - Y_MEAN) / Y_STD).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (prep(p, aug), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

# ---- model: ConvNeXt-Tiny, last stage trainable --------------------------
base = keras.applications.ConvNeXtTiny(weights="imagenet", include_top=False, pooling="avg")
base.trainable = False
matched = 0
for layer in base.layers:
    if layer.name.startswith(LAST_STAGE[BACKBONE]):
        layer.trainable = True; matched += 1
assert matched > 0, ("LAST_STAGE prefix matched no layers -- the backbone would stay "
                     "frozen and only the head would train")

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(256, activation="relu")(x)
out = keras.layers.Dense(1)(x)
model = keras.Model(inputs, out)
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse")

n_tr = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
n_tot = sum(int(np.prod(w.shape)) for w in model.weights)
print(f"trainable {n_tr:,} / {n_tot:,} ({100*n_tr/n_tot:.1f}%)")
print("GPU:", tf.config.list_physical_devices("GPU"))
'''

CELL_TRAIN = '''hist = model.fit(
    dataset(tr_df, TRAIN_IMG_DIR, augmenter(), shuffle=True),
    validation_data=dataset(va_df, TRAIN_IMG_DIR),
    epochs=EPOCHS,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                             restore_best_weights=True, verbose=1)],
    verbose=2,
)

pred = model.predict(dataset(va_df, TRAIN_IMG_DIR), verbose=0).ravel()
pred = np.clip(pred * Y_STD + Y_MEAN, Y_MIN, Y_MAX)
y_true = va_df["price"].to_numpy()
mse = float(np.mean((y_true - pred) ** 2))
print()
print(f"validation MSE {mse:,.1f}   MAE {np.mean(np.abs(y_true-pred)):,.1f}   "
      f"R2 {1 - mse/np.var(y_true):.3f}")
print(f"v1 baseline (frozen ResNet-50 + ridge) had validation MSE 114,853.7")
print(f"change vs v1 baseline: {100*(1 - mse/114853.7):+.1f}%")
'''

CELL_PREDICT = '''# ---- Kaggle submission ---------------------------------------------------
import pandas as pd

test_ds = (tf.data.Dataset.from_tensor_slices(
               [os.path.join(TEST_IMG_DIR, str(i)) for i in test_ids])
           .map(lambda p: prep(p), num_parallel_calls=tf.data.AUTOTUNE)
           .batch(BATCH).prefetch(tf.data.AUTOTUNE))

prices = np.clip(model.predict(test_ds, verbose=0).ravel() * Y_STD + Y_MEAN, Y_MIN, Y_MAX)
sub = pd.DataFrame({"imageid": test_ids, "price": np.round(prices, 1)})
sub.to_csv("submission.csv", index=False)

# fail loudly rather than silently scoring zero
assert len(sub) == len(test_ids)
assert list(sub.columns) == ["imageid", "price"]
assert sub["price"].notna().all()
print("wrote submission.csv")
print(sub.head())
print("range %.1f - %.1f  mean %.1f" % (prices.min(), prices.max(), prices.mean()))
'''

CELL_SAVE = '''# Persist the trained model and the run log as notebook output, so the
# result can be reproduced or cited without re-running.
model.save("v2_best.keras")
with open("v2_results.json", "w") as f:
    json.dump({"backbone": BACKBONE, "img_size": IMG_SIZE, "seed": SEED,
               "epochs_run": len(hist.history["loss"]),
               "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
               "val_mse": mse,
               "history": {k: [float(v) for v in vs] for k, vs in hist.history.items()}},
              f, indent=2)
print("wrote v2_best.keras and v2_results.json")
'''

CELLS = [
    ("markdown", MARKDOWN_INTRO),
    ("code", CELL_FIND_DATA),
    ("code", CELL_PIPELINE),
    ("code", CELL_TRAIN),
    ("code", CELL_PREDICT),
    ("code", CELL_SAVE),
]


def build() -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": [ln + "\n" for ln in src.split("\n")],
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
