"""Phase 2 baseline: frozen features + a regression head.

Entry point for Phase 2:

    python -m src.baseline

Two heads are trained on the same cached features, so the comparison
between them is controlled -- same split, same features, same seed, one
factor changed (the head):

  * Ridge regression -- a linear probe on the frozen features. It answers
    whether the price signal is already linearly available in the
    representation. Regularisation strength is chosen by cross-validation
    on the training partition only.
  * A small MLP -- tests whether a non-linearity in the head helps. Early
    stopping monitors validation loss.

Model selection is decided on validation MSE only. The test set is never
read by this module.

Preprocessing is fitted on the training partition alone (feature
standardisation and target standardisation) and then applied unchanged to
validation and test -- applying it earlier would leak validation
information into training.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from config import (
    IMAGE_COL,
    LOG_DIR,
    MODEL_DIR,
    PRICE_COL,
    SEED,
    TRAIN_IMG_DIR,
)
from src.data import load_split
from src.features import build_extractor, cached_features, set_seeds
from src.metrics import format_summary, mse, summary

X_SCALER_PATH = MODEL_DIR / "x_scaler.joblib"
TARGET_STATS_PATH = MODEL_DIR / "target_stats.json"
RIDGE_PATH = MODEL_DIR / "ridge.joblib"
MLP_PATH = MODEL_DIR / "mlp.keras"
RESULTS_PATH = LOG_DIR / "phase2_results.json"

MLP_HIDDEN = 256
MLP_DROPOUT = 0.2
MLP_LR = 1e-3
MLP_EPOCHS = 200
MLP_BATCH = 64
MLP_PATIENCE = 15


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------
def build_mlp(input_dim: int) -> keras.Model:
    """A deliberately small head: the backbone already supplies the features."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(MLP_HIDDEN, activation="relu"),
            keras.layers.Dropout(MLP_DROPOUT),
            keras.layers.Dense(1),
        ],
        name="mlp_head",
    )
    model.compile(optimizer=keras.optimizers.Adam(MLP_LR), loss="mse")
    return model


def fit_mlp(X_tr, y_tr, X_va, y_va):
    """Fit the MLP, stopping when validation loss stops improving."""
    model = build_mlp(X_tr.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=MLP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH,
        callbacks=callbacks,
        verbose=2,
    )
    return model, history


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def main() -> None:
    set_seeds(SEED)
    for d in (MODEL_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_split()
    y_tr_raw = train_df[PRICE_COL].to_numpy(np.float32)
    y_va_raw = val_df[PRICE_COL].to_numpy(np.float32)

    print(f"Frozen split: train {len(train_df)}, val {len(val_df)}")
    print("\nFeatures")
    extractor = build_extractor()
    X_tr_raw = cached_features("train", train_df[IMAGE_COL].tolist(), TRAIN_IMG_DIR, extractor)
    X_va_raw = cached_features("val", val_df[IMAGE_COL].tolist(), TRAIN_IMG_DIR, extractor)

    # Preprocessing is fitted on train only, then applied to val.
    x_scaler = StandardScaler().fit(X_tr_raw)
    X_tr = x_scaler.transform(X_tr_raw).astype(np.float32)
    X_va = x_scaler.transform(X_va_raw).astype(np.float32)

    y_mean, y_std = float(y_tr_raw.mean()), float(y_tr_raw.std())
    y_tr = ((y_tr_raw - y_mean) / y_std).astype(np.float32)
    y_va = ((y_va_raw - y_mean) / y_std).astype(np.float32)

    # A linear head extrapolates: nothing stops Ridge predicting a price the
    # training data never contained. The target has a hard empirical range
    # (195-2000 in $1000), so predictions outside it are known to be wrong and
    # are clipped back to the range observed in training. The bounds come from
    # the training partition only -- using validation bounds would leak.
    y_min, y_max = float(y_tr_raw.min()), float(y_tr_raw.max())

    results: dict = {
        "phase": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "feature_dim": int(X_tr.shape[1]),
        "feature_cache": "outputs/features/*.npy (frozen ResNet-50, cached)",
        "target_stats": {"mean": y_mean, "std": y_std, "min": y_min, "max": y_max},
        "postprocessing": f"predictions clipped to the training price range [{y_min:.0f}, {y_max:.0f}]",
    }

    # Reference: always predicting the training mean.
    b0 = summary(y_va_raw, np.full_like(y_va_raw, y_mean))
    results["B0_mean"] = b0
    print("\n" + format_summary("B0  (predict mean)", b0))

    # Ridge -- linear probe.
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(X_tr, y_tr)
    pred_ridge_raw = ridge.predict(X_va) * y_std + y_mean
    pred_ridge = np.clip(pred_ridge_raw, y_min, y_max)
    res_ridge = summary(y_va_raw, pred_ridge)
    n_clip_ridge = int(((pred_ridge_raw < y_min) | (pred_ridge_raw > y_max)).sum())
    results["B1_ridge"] = {
        **res_ridge,
        "alpha": float(ridge.alpha_),
        "unclipped_mse": mse(y_va_raw, pred_ridge_raw),
        "n_clipped": n_clip_ridge,
    }
    print(format_summary("B1  frozen + ridge", res_ridge))
    print(f"    alpha = {ridge.alpha_:.3g}, clipped {n_clip_ridge} val rows "
          f"(unclipped MSE {results['B1_ridge']['unclipped_mse']:.1f})")

    # MLP head.
    mlp, history = fit_mlp(X_tr, y_tr, X_va, y_va)
    pred_mlp_raw = mlp.predict(X_va, verbose=0).ravel() * y_std + y_mean
    pred_mlp = np.clip(pred_mlp_raw, y_min, y_max)
    res_mlp = summary(y_va_raw, pred_mlp)
    n_clip_mlp = int(((pred_mlp_raw < y_min) | (pred_mlp_raw > y_max)).sum())
    results["B2_mlp"] = {
        **res_mlp,
        "epochs_run": len(history.history["loss"]),
        "hidden": MLP_HIDDEN,
        "dropout": MLP_DROPOUT,
        "learning_rate": MLP_LR,
        "unclipped_mse": mse(y_va_raw, pred_mlp_raw),
        "n_clipped": n_clip_mlp,
    }
    print(format_summary("B2  frozen + mlp", res_mlp))
    print(f"    epochs run = {len(history.history['loss'])}, clipped {n_clip_mlp} val rows "
          f"(unclipped MSE {results['B2_mlp']['unclipped_mse']:.1f})")

    # Select on validation MSE.
    selected = "B2_mlp" if res_mlp["mse"] < res_ridge["mse"] else "B1_ridge"
    results["selected"] = selected

    print("\nSaving")
    joblib.dump(x_scaler, X_SCALER_PATH)
    joblib.dump(ridge, RIDGE_PATH)
    mlp.save(MLP_PATH)
    TARGET_STATS_PATH.write_text(
        json.dumps({"mean": y_mean, "std": y_std, "min": y_min, "max": y_max}, indent=2),
        encoding="utf-8",
    )
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for p in (X_SCALER_PATH, RIDGE_PATH, MLP_PATH, TARGET_STATS_PATH, RESULTS_PATH):
        print(f"  {p.relative_to(p.parents[1])}")

    print(f"\nSelected on validation MSE: {selected}")
    print(f"Improvement over B0: {100 * (1 - results[selected]['mse'] / b0['mse']):.1f}%")


if __name__ == "__main__":
    main()
