"""v2 pipeline: aspect-correct preprocessing + a modern backbone.

    python -m src.train_v2 --epochs 20                 # local (slow)
    python -m src.train_v2 --epochs 20 --backbone convnext_tiny

Two things changed from the v1 pipeline, and both were motivated by looking
at where v1 failed rather than by trying things at random.

**Aspect ratio.** v1 resized every image to a square. The originals are
about 400x280, so that squashed them horizontally by a different factor for
every source size -- 400x277 was stretched 1.44x vertically relative to
horizontal, 400x233 by 1.71x. The task partly requires judging the
proportions and size of a house, and v1 was feeding the model the same
building at inconsistent geometry. `resize_with_pad` scales by the longer
side and pads the remainder, so no shape is distorted and nothing is
cropped away.

**Backbone.** v1 used ResNet-50, a 2015 architecture. ConvNeXt is a 2022
one and its ImageNet features are substantially stronger. It is also about
three times slower per image on CPU, which is why this pipeline is meant to
run on a GPU (see notebooks/kaggle_v2.ipynb).

Everything else is deliberately unchanged from v1 so the comparison stays
clean: same frozen split, same seed, same target standardisation, same
clipping of predictions to the training price range.
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

BACKBONES = {
    "convnext_tiny": keras.applications.ConvNeXtTiny,
    "convnext_small": keras.applications.ConvNeXtSmall,
    "convnext_base": keras.applications.ConvNeXtBase,
    "resnet50": keras.applications.ResNet50,
}
# Each family expects its own input scaling.
PREPROCESS = {
    "convnext_tiny": keras.applications.convnext.preprocess_input,
    "convnext_small": keras.applications.convnext.preprocess_input,
    "convnext_base": keras.applications.convnext.preprocess_input,
    "resnet50": keras.applications.resnet50.preprocess_input,
}
# Prefix of the last trainable stage. These are the *full* layer-name
# prefixes: Keras names ConvNeXt layers "convnext_tiny_stage_3_...", so a
# bare "stage_3" matches nothing and silently leaves the whole backbone
# frozen. build_model asserts that the prefix actually matched.
LAST_STAGE = {
    "convnext_tiny": "convnext_tiny_stage_3",
    "convnext_small": "convnext_small_stage_3",
    "convnext_base": "convnext_base_stage_3",
    "resnet50": "conv5",
}


def build_model(backbone: str, img_size: int, trainable_from_last_stage: bool = True) -> keras.Model:
    """Backbone with only its last stage trainable, plus a regression head."""
    base = BACKBONES[backbone](weights="imagenet", include_top=False, pooling="avg")
    base.trainable = False
    if trainable_from_last_stage:
        stage = LAST_STAGE[backbone]
        matched = 0
        for layer in base.layers:
            if layer.name.startswith(stage):
                layer.trainable = True
                matched += 1
        if matched == 0:
            raise RuntimeError(
                f"LAST_STAGE[{backbone!r}] = {stage!r} matched no layers, so the "
                f"backbone would stay frozen and only the head would train. "
                f"Layer names start with e.g. {base.layers[1].name!r}."
            )

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name=f"{backbone}_finetune")
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse")
    return model


def build_augmenter() -> keras.Sequential:
    """Mild pose and lighting augmentation. Orientation of a facade is not
    meaningful, so a horizontal flip is safe."""
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


def _preprocess(path: tf.Tensor, img_size: int, backbone: str, augmenter=None) -> tf.Tensor:
    """Decode, scale by the longer side and pad -- never squash.

    ``resize_with_pad`` is the difference from v1, where every image was
    forced into a square and distorted by a source-dependent amount.
    """
    raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize_with_pad(image, img_size, img_size)
    if augmenter is not None:
        image = augmenter(image, training=True)
    return PREPROCESS[backbone](image)


def make_dataset(
    df_or_ids, img_dir, img_size, backbone, y_mean=None, y_std=None,
    augmenter=None, batch_size=BATCH_SIZE, shuffle=False, with_labels=True,
) -> tf.data.Dataset:
    """tf.data pipeline over images, optionally paired with a target."""
    if with_labels:
        paths = [str(Path(img_dir) / str(i)) for i in df_or_ids[IMAGE_COL]]
        labels = ((df_or_ids[PRICE_COL].to_numpy(np.float32) - y_mean) / y_std).astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda p, y: (_preprocess(p, img_size, backbone, augmenter), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        paths = [str(Path(img_dir) / str(i)) for i in df_or_ids]
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(
            lambda p: _preprocess(p, img_size, backbone, augmenter),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train(args) -> dict:
    set_seeds(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_split()
    y_tr = train_df[PRICE_COL].to_numpy(np.float32)
    y_va = val_df[PRICE_COL].to_numpy(np.float32)
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    y_min, y_max = float(y_tr.min()), float(y_tr.max())

    model = build_model(args.backbone, args.img_size)
    n_train = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    n_total = int(sum(np.prod(w.shape) for w in model.weights))
    print(f"backbone {args.backbone}  resolution {args.img_size}px  augmentation {args.augment}")
    print(f"trainable {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")

    augmenter = build_augmenter() if args.augment else None
    train_ds = make_dataset(train_df, TRAIN_IMG_DIR, args.img_size, args.backbone,
                            y_mean, y_std, augmenter, shuffle=True)
    val_ds = make_dataset(val_df, TRAIN_IMG_DIR, args.img_size, args.backbone,
                          y_mean, y_std)

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
    res = summary(y_va, pred)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    print()
    print(format_summary("v2", res))
    print(f"    {len(history.history['loss'])} epochs, best at {best_epoch}")
    print(format_summary("B1 (v1 baseline)", B1_REFERENCE))
    print(f"    change vs B1: {100*(1 - res['mse']/B1_REFERENCE['mse']):+.1f}%")

    model.save(MODEL_DIR / "v2_best.keras")
    out = {
        "pipeline": "v2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backbone": args.backbone,
        "img_size": args.img_size,
        "augmentation": bool(augmenter),
        "resize": "resize_with_pad (aspect preserved; v1 used a square squash)",
        "seed": SEED,
        "epochs_cap": args.epochs,
        "epochs_run": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "trainable_params": n_train,
        "total_params": n_total,
        "val": res,
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
    }
    (LOG_DIR / "v2_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote v2_best.keras and v2_results.json")
    return out


def predict(args) -> None:
    """Write a Kaggle submission from the trained v2 model."""
    import pandas as pd

    from config import PRED_DIR, TEST_CSV, TEST_IMG_DIR

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    test_ids = pd.read_csv(TEST_CSV)[IMAGE_COL].astype(str).tolist()

    train_df, _ = load_split()
    y_tr = train_df[PRICE_COL].to_numpy(np.float32)
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    y_min, y_max = float(y_tr.min()), float(y_tr.max())

    model = keras.models.load_model(MODEL_DIR / "v2_best.keras")
    ds = make_dataset(test_ids, TEST_IMG_DIR, args.img_size, args.backbone, with_labels=False)
    raw = model.predict(ds, verbose=0).ravel()

    prices = np.clip(raw * y_std + y_mean, y_min, y_max)
    out_path = PRED_DIR / "submission_v2.csv"
    pd.DataFrame({IMAGE_COL: test_ids, PRICE_COL: np.round(prices, 1)}).to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(f"  rows {len(test_ids)}  range {prices.min():.1f}-{prices.max():.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="convnext_tiny", choices=sorted(BACKBONES))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--predict-only", action="store_true", help="skip training, just write the submission")
    args = p.parse_args()

    if args.predict_only:
        predict(args)
    else:
        train(args)
        predict(args)


if __name__ == "__main__":
    main()
