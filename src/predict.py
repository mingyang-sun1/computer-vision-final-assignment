"""Generate the Kaggle submission CSV from a trained Phase 2 model.

    python -m src.predict                 # the head selected on validation
    python -m src.predict --model ridge   # force a specific head

Output format follows sample_solution.csv: two columns, `imageid` and
`price`, with price in units of $1000 USD, one row per test image in the
order given by the course test.csv.

This is the only module that reads the test set, and it uses it purely to
produce predictions -- never for training, tuning, or model selection.
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from config import (
    IMAGE_COL,
    PRED_DIR,
    PRICE_COL,
    SEED,
    TEST_CSV,
    TEST_IMG_DIR,
)
from src.baseline import MLP_PATH, RESULTS_PATH, RIDGE_PATH, TARGET_STATS_PATH, X_SCALER_PATH
from src.features import build_extractor, cached_features, set_seeds


def load_model(name: str):
    """Return (model, kind) for a head name, or the Phase 2 selection."""
    if name == "selected":
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        name = "mlp" if results["selected"] == "B2_mlp" else "ridge"
    if name == "mlp":
        from tensorflow import keras

        return keras.models.load_model(MLP_PATH), "mlp"
    return joblib.load(RIDGE_PATH), "ridge"


def predict(model, kind: str, X) -> np.ndarray:
    """Predict in the original price units, clipped to the training range.

    Both heads were trained against a standardised target, so the inverse
    transform is applied here to get back to $1000 USD. The result is then
    clipped to the price range seen during training: a linear head
    extrapolates freely, but a price outside the range the data contains is
    known to be wrong, so there is nothing to lose by bounding it.

    The bounds are read from the training partition, never from test.
    """
    stats = json.loads(TARGET_STATS_PATH.read_text(encoding="utf-8"))
    if "min" not in stats or "max" not in stats:
        raise KeyError(
            f"{TARGET_STATS_PATH.name} has no clipping bounds. "
            "Re-run `python -m src.baseline` to regenerate it."
        )

    if kind == "mlp":
        raw = model.predict(X, verbose=0).ravel()
    else:
        raw = model.predict(X)

    prices = raw * stats["std"] + stats["mean"]
    lo, hi = stats["min"], stats["max"]
    n_clipped = int(((prices < lo) | (prices > hi)).sum())
    if n_clipped:
        print(f"  clipped {n_clipped} predictions into the training range [{lo:.0f}, {hi:.0f}]")
    return np.clip(prices, lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="selected", choices=["selected", "ridge", "mlp"],
        help="which head to use (default: the one selected in Phase 2)",
    )
    parser.add_argument("--out", default=None, help="output csv path")
    args = parser.parse_args()

    set_seeds(SEED)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(TEST_CSV)
    test_ids = test_df[IMAGE_COL].astype(str).tolist()
    print(f"test images: {len(test_ids)}")

    print("\nFeatures")
    extractor = build_extractor()
    X_test_raw = cached_features("test", test_ids, TEST_IMG_DIR, extractor)

    x_scaler = joblib.load(X_SCALER_PATH)
    X_test = x_scaler.transform(X_test_raw).astype(np.float32)

    model, kind = load_model(args.model)
    print(f"\nPredicting with: {kind}")
    prices = predict(model, kind, X_test)

    out_path = PRED_DIR / (args.out or f"submission_phase2_{kind}.csv")
    submission = pd.DataFrame({IMAGE_COL: test_ids, PRICE_COL: np.round(prices, 1)})
    submission.to_csv(out_path, index=False)

    # Sanity checks -- a malformed csv scores zero on Kaggle regardless of
    # how good the model is, so this is worth failing loudly on.
    assert len(submission) == len(test_ids), "row count does not match the test set"
    assert submission[PRICE_COL].notna().all(), "predictions contain NaN"
    assert list(submission.columns) == [IMAGE_COL, PRICE_COL], "unexpected columns"

    print(f"\nwrote {out_path}")
    print(f"  rows          {len(submission)}")
    print(f"  columns       {list(submission.columns)}")
    print(f"  price range   {submission[PRICE_COL].min():.1f} - {submission[PRICE_COL].max():.1f}")
    print(f"  price mean    {submission[PRICE_COL].mean():.1f}")
    print("\nSubmit this file at the Kaggle competition page.")


if __name__ == "__main__":
    main()
