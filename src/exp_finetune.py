"""Phase 3, experiments E2/E3/E4: partial fine-tuning.

    python -m src.exp_finetune                        # E2: 224px, no augment
    python -m src.exp_finetune --img-size 320         # E3: resolution
    python -m src.exp_finetune --augment              # E4: augmentation

All three run the same code with different settings, so each differs from
E2 in exactly one factor:

    E2  baseline for this family -- fine-tune at 224px, no augmentation
    E3  + input resolution 320px
    E4  + augmentation

Compared against Phase 2's B1 (frozen ResNet-50 + Ridge, val MSE 114,853.7).
The split and seed are unchanged throughout, so results are comparable.

Design choices:

- **Unfreeze only the last stage (layer4) plus a new regression head.** The
  early layers of a CNN encode generic edges and textures that transfer
  well; the later layers encode task-specific structure that does not.
  Layer4 is 14.9M of the backbone's 23.5M parameters, so this still leaves
  most of the network trainable and overfitting is a real risk -- which is
  what the early stopping, the low learning rate, and E4's augmentation are
  there to contain.
- **Low learning rate (1e-4).** The pretrained weights are a good starting
  point; a large step would destroy them before the randomly-initialised
  head has settled.
- **MSE on the standardised target**, matching the competition metric.
- Early stopping on validation loss, restoring the best weights.

Augmentation (E4) targets the two directions the assignment says the data
actually varies in -- "pose, lighting, quality" -- and deliberately stays
mild, because the task is to read appearance and aggressive colour or crop
augmentation could destroy the very signal being predicted.

The test set is not touched by this module.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    BATCH_SIZE,
    IMAGE_COL,
    LOG_DIR,
    MODEL_DIR,
    PRICE_COL,
    SEED,
    TRAIN_IMG_DIR,
)
from src.data import load_split
from src.features import set_seeds
from src.metrics import format_summary, summary

B1_REFERENCE = {"mse": 114853.7, "mae": 249.0, "r2": 0.256}

LEARNING_RATE = 1e-4
EPOCHS = 15
PATIENCE = 4
HEAD_UNITS = 256
HEAD_DROPOUT = 0.2


def build_augmenter() -> keras.Sequential:
    """Mild, geometry-and-lighting augmentation on 0-255 images.

    Translation and zoom stand in for the pose and framing variation in the
    dataset; brightness and contrast stand in for the lighting variation.
    Horizontal flip is safe here because house facades have no meaningful
    left-right orientation in this task.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomBrightness(0.15, value_range=(0.0, 255.0)),
            keras.layers.RandomContrast(0.15, value_range=(0.0, 255.0)),
        ],
        name="augment",
    )


def _preprocess(path: tf.Tensor, img_size: int, augmenter=None) -> tf.Tensor:
    """Decode, resize, optionally augment, then apply the backbone's norm."""
    raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, (img_size, img_size))
    if augmenter is not None:
        image = augmenter(image, training=True)
    return keras.applications.resnet50.preprocess_input(image)


def build_model(img_size: int) -> keras.Model:
    """ResNet-50 with only the last stage trainable, plus a regression head."""
    base = keras.applications.ResNet50(
        weights="imagenet", include_top=False, pooling="avg"
    )
    base.trainable = False
    for layer in base.layers:
        if layer.name.startswith("conv5"):  # ResNet-50's last stage is conv5/layer4
            layer.trainable = True

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = keras.layers.Dropout(HEAD_DROPOUT)(x)
    x = keras.layers.Dense(HEAD_UNITS, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name="resnet50_layer4_finetune")
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    return model


def make_dataset(
    df, img_dir, y_mean, y_std, img_size, augmenter=None, batch_size=BATCH_SIZE, shuffle=False
) -> tf.data.Dataset:
    """(image, standardised price) pipeline.

    Augmentation is applied here and this function is only ever called with
    augmenter=None for validation, so the training partition is the only one
    that sees augmented images.
    """
    paths = [str(Path(img_dir) / str(i)) for i in df[IMAGE_COL]]
    labels = ((df[PRICE_COL].to_numpy(np.float32) - y_mean) / y_std).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: (_preprocess(p, img_size, augmenter), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def count_trainable(model: keras.Model) -> tuple[int, int]:
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(sum(np.prod(w.shape) for w in model.weights))
    return trainable, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--tag", default=None, help="experiment id (default derived from flags)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    tag = args.tag or ("E4_augment" if args.augment else ("E3_320px" if args.img_size != 224 else "E2_finetune"))

    set_seeds(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_split()
    y_tr_raw = train_df[PRICE_COL].to_numpy(np.float32)
    y_va_raw = val_df[PRICE_COL].to_numpy(np.float32)
    y_mean, y_std = float(y_tr_raw.mean()), float(y_tr_raw.std())
    y_min, y_max = float(y_tr_raw.min()), float(y_tr_raw.max())

    augmenter = build_augmenter() if args.augment else None
    model = build_model(args.img_size)
    trainable, total = count_trainable(model)

    print(f"[{tag}] resolution {args.img_size}px, augmentation {bool(augmenter)}")
    print(f"[{tag}] split 7000/1000, seed {SEED}, epochs {args.epochs}, patience {args.patience}")
    print(f"[{tag}] trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print()

    train_ds = make_dataset(
        train_df, TRAIN_IMG_DIR, y_mean, y_std, args.img_size, augmenter, shuffle=True
    )
    val_ds = make_dataset(val_df, TRAIN_IMG_DIR, y_mean, y_std, args.img_size)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience,
                restore_best_weights=True, verbose=1,
            )
        ],
        verbose=2,
    )

    pred = np.clip(model.predict(val_ds, verbose=0).ravel() * y_std + y_mean, y_min, y_max)
    res = summary(y_va_raw, pred)
    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    print()
    print(format_summary(f"{tag}  {args.img_size}px aug={bool(augmenter)}", res))
    print(f"    {epochs_run} epochs run, best at epoch {best_epoch}")
    print(format_summary("B1  frozen + ridge (ref)", B1_REFERENCE))
    print(f"    change vs B1: {100*(1 - res['mse']/B1_REFERENCE['mse']):+.1f}%")

    model_path = MODEL_DIR / f"finetuned_{tag}.keras"
    results_path = LOG_DIR / f"phase3_{tag}.json"
    model.save(model_path)
    results_path.write_text(
        json.dumps(
            {
                "experiment": tag,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seed": SEED,
                "img_size": args.img_size,
                "augmentation": bool(augmenter),
                "unfrozen": "layer4 (conv5) + regression head",
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs_cap": args.epochs,
                "patience": args.patience,
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "trainable_params": trainable,
                "total_params": total,
                "val": res,
                "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {model_path.name} and {results_path.name}")


if __name__ == "__main__":
    main()
