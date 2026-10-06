"""Phase 3, experiment 1: input resolution.

    python -m src.exp_resolution

Only the input resolution changes across rows. The backbone stays frozen at
ImageNet weights, the pooling stays global average, the split and seed stay
fixed, and both heads use their Phase 2 settings. Any difference in
validation MSE is therefore attributable to resolution alone.

Motivation: the images are natively about 400px wide, but Phase 2 resized
them to 224. That downsampling destroys two things the task cares about --
fine detail that indicates condition and build quality, and the small MLS
watermarks in the bottom corner of many photos, which are a cue to *where*
the house is. Location is the dominant driver of house price, and the
assignment explicitly suggests predicting location-related features.

The images are never upscaled beyond their native width (400), since that
would add no information.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from config import IMAGE_COL, LOG_DIR, MODEL_DIR, PRICE_COL, SEED, TRAIN_IMG_DIR
from src.data import load_split
from src.features import build_extractor, cached_features, set_seeds
from src.metrics import format_summary, summary

RESOLUTIONS = [224, 320, 400]
RESULTS_PATH = LOG_DIR / "phase3_resolution.json"

MLP_HIDDEN = 256
MLP_DROPOUT = 0.2
MLP_LR = 1e-3
MLP_EPOCHS = 200
MLP_BATCH = 64
MLP_PATIENCE = 15


def fit_ridge(X_tr, y_tr, X_va, y_va, y_mean, y_std, y_min, y_max):
    """Linear probe, identical to Phase 2's B1."""
    model = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_va) * y_std + y_mean, y_min, y_max)
    return model, pred


def fit_mlp(X_tr, y_tr, X_va, y_va, y_mean, y_std, y_min, y_max):
    """Small MLP head, identical to Phase 2's B2."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_tr.shape[1],)),
            keras.layers.Dense(MLP_HIDDEN, activation="relu"),
            keras.layers.Dropout(MLP_DROPOUT),
            keras.layers.Dense(1),
        ],
        name="mlp_head",
    )
    model.compile(optimizer=keras.optimizers.Adam(MLP_LR), loss="mse")
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=MLP_PATIENCE,
                restore_best_weights=True, verbose=0,
            )
        ],
        verbose=0,
    )
    pred = np.clip(model.predict(X_va, verbose=0).ravel() * y_std + y_mean, y_min, y_max)
    return model, pred, len(history.history["loss"])


def main() -> None:
    set_seeds(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_split()
    y_tr_raw = train_df[PRICE_COL].to_numpy(np.float32)
    y_va_raw = val_df[PRICE_COL].to_numpy(np.float32)

    # Target statistics are resolution-independent: same split, same prices.
    y_mean, y_std = float(y_tr_raw.mean()), float(y_tr_raw.std())
    y_min, y_max = float(y_tr_raw.min()), float(y_tr_raw.max())

    print(f"Fixed: split 7000/1000, seed {SEED}, frozen ResNet-50, GAP pooling")
    print(f"Varying: input resolution over {RESOLUTIONS}\n")

    extractor = build_extractor()
    results = {
        "experiment": "E1_input_resolution",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "backbone": "resnet50 (ImageNet, frozen, GAP)",
        "head": "ridge + mlp, Phase 2 settings",
        "split": "data/splits/train_split.csv + val_split.csv",
        "resolutions": RESOLUTIONS,
        "rows": [],
    }

    for size in RESOLUTIONS:
        print(f"--- {size}px ---")
        X_tr_raw = cached_features("train", train_df[IMAGE_COL].tolist(), TRAIN_IMG_DIR, extractor, size)
        X_va_raw = cached_features("val", val_df[IMAGE_COL].tolist(), TRAIN_IMG_DIR, extractor, size)

        scaler = StandardScaler().fit(X_tr_raw)
        X_tr = scaler.transform(X_tr_raw).astype(np.float32)
        X_va = scaler.transform(X_va_raw).astype(np.float32)
        y_tr = ((y_tr_raw - y_mean) / y_std).astype(np.float32)
        y_va = ((y_va_raw - y_mean) / y_std).astype(np.float32)

        _, pred_ridge = fit_ridge(X_tr, y_tr, X_va, y_va, y_mean, y_std, y_min, y_max)
        res_ridge = summary(y_va_raw, pred_ridge)

        _, pred_mlp, epochs = fit_mlp(X_tr, y_tr, X_va, y_va, y_mean, y_std, y_min, y_max)
        res_mlp = summary(y_va_raw, pred_mlp)

        print(format_summary(f"  {size}px ridge", res_ridge))
        print(format_summary(f"  {size}px mlp  ", res_mlp) + f"   ({epochs} epochs)")

        results["rows"].append(
            {"img_size": size, "ridge": res_ridge, "mlp": res_mlp, "mlp_epochs": epochs}
        )

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS_PATH.relative_to(RESULTS_PATH.parents[1])}")


if __name__ == "__main__":
    main()
