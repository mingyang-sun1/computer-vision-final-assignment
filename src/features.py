"""Frozen ResNet-50 feature extraction.

Phase 2 baseline. An ImageNet-pretrained ResNet-50 is used as a *fixed*
feature extractor: the global-average-pooled output of the last
convolutional block gives a 2048-d descriptor per image, and a small
regression head is trained on top. Nothing in the backbone is updated --
that is the Phase 3 experiment.

Why the features are cached
---------------------------
This machine has no GPU (Intel Arc is not usable from TensorFlow), so
extraction is the only slow step: roughly 40 minutes for all 11,000 images.
Extracting once and caching to `.npy` means the head itself trains in
seconds, which is what makes the Phase 3 sweep of controlled experiments
practical at all. The cache is keyed by split name and backbone, and is
regenerated automatically if a cached file does not match the split size.

Why global average pooling
--------------------------
GAP collapses the spatial feature map into one vector per image, which
makes the descriptor invariant to where in the frame a feature appears.
That is a reasonable prior for this task (a garage is a garage whether it
is left or right of frame) but it discards layout information such as
relative size of house to lot -- worth revisiting in the error analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import BACKBONE, BATCH_SIZE, FEATURE_DIR, IMG_SIZE, TEST_IMG_DIR, TRAIN_IMG_DIR


def set_seeds(seed: int) -> None:
    """Make a run reproducible. Called by every entry point."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_extractor() -> keras.Model:
    """ImageNet ResNet-50, GAP-pooled, frozen. Weights download on first use."""
    base = keras.applications.ResNet50(
        weights="imagenet", include_top=False, pooling="avg"
    )
    base.trainable = False
    return base


def _preprocess(path: tf.Tensor) -> tf.Tensor:
    """Decode, resize to 224x224, and apply the backbone's own normalisation.

    The images are only ~400x280, so resizing to 224 loses little detail.
    preprocess_input must match the weights: ResNet-50 expects the Caffe
    convention (RGB->BGR and per-channel mean subtraction over 0-255 inputs).
    """
    raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return keras.applications.resnet50.preprocess_input(image)


def make_dataset(image_ids, img_dir, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Batched pipeline over image files, in the order of ``image_ids``."""
    paths = [str(Path(img_dir) / str(i)) for i in image_ids]
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def extract(image_ids, img_dir, extractor: keras.Model, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Run the frozen backbone over every image. Row order matches image_ids."""
    ds = make_dataset(image_ids, img_dir, batch_size)
    feats = extractor.predict(ds, verbose=1)
    return np.asarray(feats, dtype=np.float32)


def cached_features(name: str, image_ids, img_dir, extractor: keras.Model | None = None) -> np.ndarray:
    """Return features for a split, extracting them only if not already cached.

    ``name`` is one of "train", "val", "test". The cache is invalidated
    automatically when its row count no longer matches the split, which is
    what happens if the split is ever rebuilt.
    """
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURE_DIR / f"{name}_{BACKBONE}.npy"

    if path.exists():
        feats = np.load(path)
        if len(feats) == len(image_ids):
            print(f"  {name:<5} cached   {path.name} {feats.shape}")
            return feats
        print(f"  {name:<5} cache stale ({len(feats)} rows vs {len(image_ids)}), re-extracting")

    extractor = extractor or build_extractor()
    print(f"  {name:<5} extracting {len(image_ids)} images ...")
    feats = extract(image_ids, img_dir, extractor)
    np.save(path, feats)
    print(f"  {name:<5} saved    {path.name} {feats.shape}")
    return feats


def main() -> None:
    """Extract and cache features for train / val / test."""
    from src.data import load_split

    train_df, val_df = load_split()
    test_ids = _read_test_ids()

    print("Extracting frozen features")
    extractor = build_extractor()
    cached_features("train", train_df["imageid"].tolist(), TRAIN_IMG_DIR, extractor)
    cached_features("val", val_df["imageid"].tolist(), TRAIN_IMG_DIR, extractor)
    cached_features("test", test_ids, TEST_IMG_DIR, extractor)


def _read_test_ids() -> list[str]:
    """Test image IDs in the order given by the course csv."""
    import pandas as pd

    from config import TEST_CSV

    return pd.read_csv(TEST_CSV)["imageid"].astype(str).tolist()


if __name__ == "__main__":
    main()
