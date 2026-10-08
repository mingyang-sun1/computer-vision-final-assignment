"""Frozen ResNet-50 feature extraction.

Phase 2 baseline plus the Phase 3 resolution experiments. An
ImageNet-pretrained ResNet-50 is used as a *fixed* feature extractor: the
global-average-pooled output of the last convolutional block gives a 2048-d
descriptor per image, and a regression head is trained on top.

Why the features are cached
---------------------------
This machine has no GPU, so extraction is the slow step. Extracting once and
caching to `.npy` means the head itself trains in seconds, which is what
makes the Phase 3 sweep of controlled experiments practical.

**The cache key includes the input resolution.** Features extracted at 224
and at 400 are different tensors and must never be mixed up; keying only by
split name silently reused the 224 features when the resolution changed.

Why global average pooling
--------------------------
GAP collapses the spatial feature map into one vector per image, making the
descriptor invariant to where a feature appears. That is a reasonable prior
(a garage is a garage whether it is left or right of frame) but it discards
layout information such as the size of the house relative to its lot, and it
averages away small high-frequency details -- including the MLS watermarks
in the bottom corner of many of these photos, which are a location cue.
Both costs are what the Phase 3 experiments probe.
"""

from __future__ import annotations

import hashlib
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


def _preprocess(path: tf.Tensor, img_size: int) -> tf.Tensor:
    """Decode, resize to a square, and apply the backbone's own normalisation.

    preprocess_input must match the weights: ResNet-50 expects the Caffe
    convention (RGB->BGR and per-channel mean subtraction over 0-255 inputs).

    The native images are about 400x280, so a square resize distorts the
    aspect ratio. That is held fixed across the resolution sweep so that
    resolution is the only factor changing.
    """
    raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, (img_size, img_size))
    return keras.applications.resnet50.preprocess_input(image)


def make_dataset(
    image_ids, img_dir, img_size: int = IMG_SIZE, batch_size: int = BATCH_SIZE
) -> tf.data.Dataset:
    """Batched pipeline over image files, in the order of ``image_ids``."""
    paths = [str(Path(img_dir) / str(i)) for i in image_ids]
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda p: _preprocess(p, img_size), num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def extract(
    image_ids,
    img_dir,
    extractor: keras.Model,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Run the frozen backbone over every image. Row order matches image_ids."""
    ds = make_dataset(image_ids, img_dir, img_size, batch_size)
    feats = extractor.predict(ds, verbose=0)
    return np.asarray(feats, dtype=np.float32)


def feature_path(name: str, img_size: int = IMG_SIZE) -> Path:
    """Cache location. The resolution is part of the key, not decoration."""
    return FEATURE_DIR / f"{name}_{BACKBONE}_{img_size}.npy"


def _ids_fingerprint(image_ids) -> str:
    """SHA-1 over the image ids in order, identifying the exact row set."""
    h = hashlib.sha1()
    for image_id in image_ids:
        h.update(str(image_id).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _fingerprint_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".ids")


def _read_fingerprint(path: Path) -> str | None:
    sidecar = _fingerprint_path(path)
    return sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else None


def _write_fingerprint(path: Path, fingerprint: str) -> None:
    _fingerprint_path(path).write_text(fingerprint + "\n", encoding="utf-8")


def cached_features(
    name: str,
    image_ids,
    img_dir,
    extractor: keras.Model | None = None,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """Return features for a split, extracting them only if not already cached.

    ``name`` is one of "train", "val", "test". A cached file is reused only if
    it was built from *the same images in the same order*: row count alone is
    not sufficient, because a reordered or swapped split of the same length
    would silently produce features that no longer line up with the labels.
    The fingerprint is stored in a sidecar file next to the cache.
    """
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    path = feature_path(name, img_size)
    fingerprint = _ids_fingerprint(image_ids)

    if path.exists():
        feats = np.load(path)
        if len(feats) == len(image_ids) and _read_fingerprint(path) == fingerprint:
            print(f"  {name:<5} @{img_size}px  cached  {path.name} {feats.shape}")
            return feats
        reason = (
            f"{len(feats)} rows vs {len(image_ids)}"
            if len(feats) != len(image_ids)
            else "same row count but different image ids"
        )
        print(f"  {name:<5} @{img_size}px  cache stale ({reason}), re-extracting")

    extractor = extractor or build_extractor()
    print(f"  {name:<5} @{img_size}px  extracting {len(image_ids)} images ...")
    feats = extract(image_ids, img_dir, extractor, img_size)
    np.save(path, feats)
    _write_fingerprint(path, fingerprint)
    print(f"  {name:<5} @{img_size}px  saved   {path.name} {feats.shape}")
    return feats


def _read_test_ids() -> list[str]:
    """Test image IDs in the order given by the course csv."""
    import pandas as pd

    from config import TEST_CSV

    return pd.read_csv(TEST_CSV)["imageid"].astype(str).tolist()


def main() -> None:
    """Extract and cache features for train / val / test at the config size."""
    from src.data import load_split

    train_df, val_df = load_split()
    test_ids = _read_test_ids()

    print(f"Extracting frozen features at {IMG_SIZE}px")
    extractor = build_extractor()
    cached_features("train", train_df["imageid"].tolist(), TRAIN_IMG_DIR, extractor)
    cached_features("val", val_df["imageid"].tolist(), TRAIN_IMG_DIR, extractor)
    cached_features("test", test_ids, TEST_IMG_DIR, extractor)


if __name__ == "__main__":
    main()
