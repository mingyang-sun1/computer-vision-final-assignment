"""Does a stronger backbone help once the overfitting is removed?

    python -m src.exp_frozen_backbone

Every fine-tuned model in this project overfits badly: the training loss
falls to a fraction of the validation loss, because 7000 images is too few
for adapting 15-25M parameters. Freezing the backbone removes that failure
mode entirely -- there is nothing to overfit, and the only capacity is a
ridge regression on top.

That makes this the cleanest way to ask whether the backbone itself matters.
Both backbones are run through the *same* aspect-correct preprocessing so the
only thing differing between rows is the network, and each is run at two
resolutions.

Reference: v1 measured frozen ResNet-50 features (squashed preprocessing,
224px) at 114,853.7 validation MSE, R2 0.256.

Features are cached per (backbone, resolution) -- see
features.feature_path, which keys the cache by resolution precisely so that
changing it cannot silently reuse the wrong tensors.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from config import FEATURE_DIR, IMAGE_COL, LOG_DIR, PRICE_COL, SEED, TRAIN_IMG_DIR
from src.data import load_split
from src.features import set_seeds
from src.metrics import format_summary, summary

B1_REFERENCE = {"mse": 114853.7, "mae": 249.0, "r2": 0.256}

BACKBONES = {
    "resnet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
    "convnext_tiny": (keras.applications.ConvNeXtTiny, keras.applications.convnext.preprocess_input),
}
RESOLUTIONS = [224, 320]


def build_extractor(backbone: str) -> keras.Model:
    ctor, _ = BACKBONES[backbone]
    model = ctor(weights="imagenet", include_top=False, pooling="avg")
    model.trainable = False
    return model


def _preprocess(path, size, preprocess_input):
    """Aspect-preserving: scale by the longer side, pad. Matches v2."""
    raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize_with_pad(image, size, size)
    return preprocess_input(image)


def extract(image_ids, img_dir, backbone, size, batch_size=64) -> np.ndarray:
    _, preprocess_input = BACKBONES[backbone]
    cache = FEATURE_DIR / f"frozen_{backbone}_{size}.npy"
    if cache.exists():
        feats = np.load(cache)
        if len(feats) == len(image_ids):
            print(f"    cached {cache.name} {feats.shape}")
            return feats
        print(f"    cache stale ({len(feats)} vs {len(image_ids)}), re-extracting")

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [str(Path(img_dir) / str(i)) for i in image_ids]
    ds = (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(lambda p: _preprocess(p, size, preprocess_input),
             num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    print(f"    extracting {len(image_ids)} images ...")
    feats = np.asarray(build_extractor(backbone).predict(ds, verbose=0), dtype=np.float32)
    np.save(cache, feats)
    print(f"    saved {cache.name} {feats.shape}")
    return feats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    p.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS)
    args = p.parse_args()

    set_seeds(SEED)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_split()
    y_tr_raw = train_df[PRICE_COL].to_numpy(np.float32)
    y_va_raw = val_df[PRICE_COL].to_numpy(np.float32)
    y_mean, y_std = float(y_tr_raw.mean()), float(y_tr_raw.std())
    y_min, y_max = float(y_tr_raw.min()), float(y_tr_raw.max())
    y_tr = ((y_tr_raw - y_mean) / y_std).astype(np.float32)

    print("Frozen features + ridge. Same split, seed and preprocessing for every row.")
    print(f"v1 reference (ResNet-50, squashed 224px): MSE {B1_REFERENCE['mse']:,.1f}\n")

    rows = []
    for backbone in args.backbones:
        for size in args.resolutions:
            print(f"  {backbone} @ {size}px")
            Xtr = extract(train_df[IMAGE_COL].tolist(), TRAIN_IMG_DIR, backbone, size)
            Xva = extract(val_df[IMAGE_COL].tolist(), TRAIN_IMG_DIR, backbone, size)

            scaler = StandardScaler().fit(Xtr)
            model = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(scaler.transform(Xtr), y_tr)
            pred = np.clip(model.predict(scaler.transform(Xva)) * y_std + y_mean, y_min, y_max)
            res = summary(y_va_raw, pred)
            res.update({"backbone": backbone, "img_size": size,
                        "alpha": float(model.alpha_), "feature_dim": int(Xtr.shape[1])})
            rows.append(res)
            print("    " + format_summary("", res).strip())

    print()
    print(f"{'backbone':<16}{'size':>6}{'MSE':>12}{'MAE':>9}{'R2':>8}{'vs B1':>10}")
    for r in rows:
        print(f"{r['backbone']:<16}{r['img_size']:>6}{r['mse']:>12,.1f}{r['mae']:>9.1f}"
              f"{r['r2']:>8.3f}{100*(1-r['mse']/B1_REFERENCE['mse']):>+9.1f}%")

    best = min(rows, key=lambda r: r["mse"])
    print(f"\nbest: {best['backbone']} @ {best['img_size']}px  MSE {best['mse']:,.1f}  "
          f"R2 {best['r2']:.3f}")

    out = {
        "experiment": "frozen_backbone_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "preprocessing": "resize_with_pad -- aspect preserved",
        "head": "ridge, alpha by cross-validation on the training partition",
        "rows": rows,
        "v1_reference": B1_REFERENCE,
    }
    (LOG_DIR / "frozen_backbone.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote frozen_backbone.json")


if __name__ == "__main__":
    main()
