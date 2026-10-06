"""Phase 3, experiment 2: partial fine-tuning.

    python -m src.exp_finetune

Compared against Phase 2's B1 (frozen ResNet-50 + Ridge). The resolution is
held at 224px and the split and seed are unchanged, so the only factor
differing is whether the backbone adapts to the data.

Motivation: E1 showed that raising the input resolution buys only 2.6%, but
that at 400px the non-linear head suddenly trains three times longer and
improves 4.9%. That points at the *representation* being the limit, not the
input: the frozen ImageNet features are not adapted to photographs of house
exteriors, which differ from ImageNet's object-centric images in lighting,
framing, and what actually matters in the scene.

Design choices:

- **Unfreeze only the last stage (layer4) plus a new regression head.** The
  early layers of a CNN encode generic edges and textures that transfer
  well; the later layers encode task-specific structure that does not.
  Layer4 is 14.9M of the backbone's 23.5M parameters, so this still leaves
  most of the network trainable and overfitting is a real risk -- which is
  what the early stopping and the low learning rate are there to contain.
  Freezing more would train faster but adapt less; this is the middle
  setting, and it is the one worth reporting.
- **Low learning rate (1e-4).** The pretrained weights are a good starting
  point; a large step would destroy them before the randomly-initialised
  head has settled.
- **MSE on the standardised target**, matching the competition metric.
- Early stopping on validation loss, restoring the best weights.

The test set is not touched by this module.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    BATCH_SIZE,
    IMAGE_COL,
    IMG_SIZE,
    LOG_DIR,
    MODEL_DIR,
    PRICE_COL,
    SEED,
    TRAIN_IMG_DIR,
)
from src.data import load_split
from src.features import _preprocess, set_seeds
from src.metrics import format_summary, summary

RESULTS_PATH = LOG_DIR / "phase3_finetune.json"
MODEL_PATH = MODEL_DIR / "finetuned_layer4.keras"

LEARNING_RATE = 1e-4
EPOCHS = 15
PATIENCE = 4
HEAD_UNITS = 256
HEAD_DROPOUT = 0.2


def build_model() -> keras.Model:
    """ResNet-50 with only the last stage trainable, plus a regression head."""
    base = keras.applications.ResNet50(
        weights="imagenet", include_top=False, pooling="avg"
    )
    base.trainable = False
    for layer in base.layers:
        if layer.name.startswith("conv5"):  # ResNet-50's last stage is conv5/layer4
            layer.trainable = True

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = keras.layers.Dropout(HEAD_DROPOUT)(x)
    x = keras.layers.Dense(HEAD_UNITS, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name="resnet50_layer4_finetune")
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    return model


def make_dataset(df, img_dir, y_mean, y_std, batch_size=BATCH_SIZE, shuffle=False):
    """(image, standardised price) pipeline. No augmentation in this experiment."""
    paths = [str(Path(img_dir) / str(i)) for i in df[IMAGE_COL]]
    labels = ((df[PRICE_COL].to_numpy(np.float32) - y_mean) / y_std).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: (_preprocess(p, IMG_SIZE), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def count_trainable(model: keras.Model) -> tuple[int, int]:
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(sum(np.prod(w.shape) for w in model.weights))
    return trainable, total


def main() -> None:
    set_seeds(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_split()
    y_tr_raw = train_df[PRICE_COL].to_numpy(np.float32)
    y_va_raw = val_df[PRICE_COL].to_numpy(np.float32)
    y_mean, y_std = float(y_tr_raw.mean()), float(y_tr_raw.std())
    y_min, y_max = float(y_tr_raw.min()), float(y_tr_raw.max())

    model = build_model()
    trainable, total = count_trainable(model)
    print(f"Fixed: split 7000/1000, seed {SEED}, resolution {IMG_SIZE}px")
    print(f"Varying: backbone frozen vs layer4+head fine-tuned")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print()

    train_ds = make_dataset(train_df, TRAIN_IMG_DIR, y_mean, y_std, shuffle=True)
    val_ds = make_dataset(val_df, TRAIN_IMG_DIR, y_mean, y_std)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE,
                restore_best_weights=True, verbose=1,
            )
        ],
        verbose=2,
    )

    pred = model.predict(val_ds, verbose=0).ravel() * y_std + y_mean
    pred = np.clip(pred, y_min, y_max)
    res = summary(y_va_raw, pred)
    epochs_run = len(history.history["loss"])

    print()
    print(format_summary("E2  layer4 fine-tuned", res))
    print(f"    {epochs_run} epochs, best val loss {min(history.history['val_loss']):.4f}")
    print()
    print(format_summary("B1  frozen + ridge (ref)", {"mse": 114853.7, "mae": 249.0, "r2": 0.256}))
    print(f"    change vs B1: {100*(1 - res['mse']/114853.7):+.1f}%")

    model.save(MODEL_PATH)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "experiment": "E2_partial_finetuning",
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seed": SEED,
                "img_size": IMG_SIZE,
                "unfrozen": "layer4 (conv5) + regression head",
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs_run": epochs_run,
                "trainable_params": trainable,
                "total_params": total,
                "val": res,
                "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {MODEL_PATH.name} and {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
