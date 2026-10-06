"""Generate the Kaggle submission CSV from a trained model.

    python -m src.predict                              # head selected in Phase 2
    python -m src.predict --model finetuned            # best Phase 3 run (E3_320px)
    python -m src.predict --model finetuned --tag E2_finetune
    python -m src.predict --model ridge                # force a specific head

Output format follows sample_solution.csv: two columns, `imageid` and
`price`, with price in units of $1000 USD, one row per test image in the
order given by the course test.csv.

Two kinds of model are supported and they consume the test set differently:

- The Phase 2 heads (ridge, mlp) run on features that were extracted once
  and cached, so prediction is a matrix multiply.
- The Phase 3 fine-tuned model takes images directly, because its backbone
  is trainable and therefore has no fixed feature representation to cache.

This is the only module that reads the test set, and it uses it purely to
produce predictions -- never for training, tuning, or model selection.
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from config import IMAGE_COL, IMG_SIZE, PRED_DIR, PRICE_COL, SEED, TEST_CSV, TEST_IMG_DIR
from src.baseline import MLP_PATH, RESULTS_PATH, RIDGE_PATH, TARGET_STATS_PATH, X_SCALER_PATH
from src.features import build_extractor, cached_features, make_dataset, set_seeds

def _finetuned_path(tag: str):
    """Model file and the resolution it expects, read from its result log.

    The resolution has to come from the run itself: a model fine-tuned at
    320px must be fed 320px images, and guessing would silently degrade it.
    """
    from config import LOG_DIR, MODEL_DIR

    results_path = LOG_DIR / f"phase3_{tag}.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Available tags: "
            + ", ".join(sorted(p.stem.replace('phase3_', '') for p in LOG_DIR.glob("phase3_*.json")))
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    model_path = MODEL_DIR / f"finetuned_{tag}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run `python -m src.exp_finetune` first.")
    return model_path, int(results["img_size"])


def load_head_model(name: str):
    """Return (model, kind) for a Phase 2 head name, or the Phase 2 selection."""
    if name == "selected":
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        name = "mlp" if results["selected"] == "B2_mlp" else "ridge"
    if name == "mlp":
        from tensorflow import keras

        return keras.models.load_model(MLP_PATH), "mlp"
    return joblib.load(RIDGE_PATH), "ridge"


def _target_stats() -> dict:
    stats = json.loads(TARGET_STATS_PATH.read_text(encoding="utf-8"))
    if "min" not in stats or "max" not in stats:
        raise KeyError(
            f"{TARGET_STATS_PATH.name} has no clipping bounds. "
            "Re-run `python -m src.baseline` to regenerate it."
        )
    return stats


def clip_and_report(prices: np.ndarray, stats: dict) -> np.ndarray:
    """Bound predictions to the training price range and say how many moved.

    A regression head extrapolates freely, but a price outside the range the
    training data contains is known to be wrong, so there is nothing to lose
    by bounding it. The bounds come from the training partition, never test.
    """
    lo, hi = stats["min"], stats["max"]
    n = int(((prices < lo) | (prices > hi)).sum())
    if n:
        print(f"  clipped {n} predictions into the training range [{lo:.0f}, {hi:.0f}]")
    return np.clip(prices, lo, hi)


def predict_head(test_ids, kind: str) -> np.ndarray:
    """Phase 2 path: cached features -> standardise -> head."""
    extractor = build_extractor()
    X_raw = cached_features("test", test_ids, TEST_IMG_DIR, extractor, IMG_SIZE)

    x_scaler = joblib.load(X_SCALER_PATH)
    X = x_scaler.transform(X_raw).astype(np.float32)

    model, kind = load_head_model(kind)
    print(f"predicting with: {kind}")
    raw = model.predict(X, verbose=0).ravel() if kind == "mlp" else model.predict(X)

    stats = _target_stats()
    return clip_and_report(raw * stats["std"] + stats["mean"], stats)


def predict_finetuned(test_ids, tag: str) -> np.ndarray:
    """Phase 3 path: the model consumes images, not cached features."""
    from tensorflow import keras

    path, img_size = _finetuned_path(tag)
    model = keras.models.load_model(path)
    print(f"predicting with: fine-tuned ResNet-50 [{tag}] at {img_size}px")

    ds = make_dataset(test_ids, TEST_IMG_DIR, img_size)
    raw = model.predict(ds, verbose=0).ravel()

    stats = _target_stats()
    return clip_and_report(raw * stats["std"] + stats["mean"], stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="selected",
        choices=["selected", "ridge", "mlp", "finetuned"],
        help="which model to use (default: the Phase 2 selection)",
    )
    parser.add_argument("--out", default=None, help="output csv path")
    parser.add_argument(
        "--tag", default="E3_320px",
        help="which fine-tuned run to use, e.g. E2_finetune, E3_320px, E4_augment",
    )
    args = parser.parse_args()

    set_seeds(SEED)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(TEST_CSV)
    test_ids = test_df[IMAGE_COL].astype(str).tolist()
    print(f"test images: {len(test_ids)}")

    if args.model == "finetuned":
        prices = predict_finetuned(test_ids, args.tag)
        default_name = f"submission_phase3_{args.tag}.csv"
    else:
        prices = predict_head(test_ids, args.model)
        default_name = f"submission_phase2_{args.model}.csv"

    out_path = PRED_DIR / (args.out or default_name)
    submission = pd.DataFrame({IMAGE_COL: test_ids, PRICE_COL: np.round(prices, 1)})
    submission.to_csv(out_path, index=False)

    # A malformed csv scores zero regardless of model quality, so fail loudly.
    assert len(submission) == len(test_ids), "row count does not match the test set"
    assert submission[PRICE_COL].notna().all(), "predictions contain NaN"
    assert list(submission.columns) == [IMAGE_COL, PRICE_COL], "unexpected columns"
    assert list(submission[IMAGE_COL]) == test_ids, "image id order changed"

    print(f"\nwrote {out_path}")
    print(f"  rows          {len(submission)}")
    print(f"  columns       {list(submission.columns)}")
    print(f"  price range   {submission[PRICE_COL].min():.1f} - {submission[PRICE_COL].max():.1f}")
    print(f"  price mean    {submission[PRICE_COL].mean():.1f}")
    print("\nSubmit this file at the Kaggle competition page.")


if __name__ == "__main__":
    main()
