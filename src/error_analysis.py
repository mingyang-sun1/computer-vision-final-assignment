"""Phase 5: error analysis.

    python -m src.error_analysis                      # the selected model (E3)
    python -m src.error_analysis --tag E2_finetune

Runs a trained model over the validation split and records where it does
well and where it fails. The report's error analysis section is built from
the files this writes.

Three things are measured, which answer different questions:

1. **How much of the error is overfitting.** The training history recorded
   by each fine-tuning run already contains the train and validation loss
   per epoch, so the gap is read from there rather than recomputed. A large
   and widening gap between the two curves is the signature of a model
   memorising the training set.

2. **Where the errors are, as a function of price.** Predictions are
   bucketed by the true price and the error summarised per bucket. A model
   that predicts the mean of its training data will look fine in the middle
   of the distribution and bad at both ends, so this is where that shows up.

3. **Whether the errors are biased in a direction.** For each bucket the
   mean signed residual is reported: negative means the model underpredicts
   that price range, positive means it overpredicts.

It also writes the per-image predictions plus the best and worst cases, so
the figures and example images in the report can be produced from real rows
rather than retyped numbers.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    IMAGE_COL,
    IMG_SIZE,
    LOG_DIR,
    PRED_DIR,
    PRICE_COL,
    SEED,
    TRAIN_IMG_DIR,
)
from src.data import load_split
from src.features import make_dataset, set_seeds
from src.metrics import format_summary, summary

N_BUCKETS = 10


def load_run(tag: str):
    """Model file, resolution, and recorded history for a fine-tuned run."""
    from tensorflow import keras

    from src.predict import _finetuned_path, _target_stats

    path, img_size = _finetuned_path(tag)
    results_path = LOG_DIR / f"phase3_{tag}.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return keras.models.load_model(path), img_size, results, _target_stats()


def overfitting_report(results: dict) -> dict:
    """Read the train/validation gap out of the recorded training history."""
    hist = results.get("history", {})
    loss, val = hist.get("loss", []), hist.get("val_loss", [])
    if not loss or not val:
        return {"available": False}

    best_epoch = int(np.argmin(val))
    return {
        "available": True,
        "epochs_run": len(loss),
        "best_epoch": best_epoch + 1,
        "train_loss_at_best": float(loss[best_epoch]),
        "val_loss_at_best": float(val[best_epoch]),
        "gap_ratio_at_best": float(loss[best_epoch] / val[best_epoch]) if val[best_epoch] else None,
        "train_loss_final": float(loss[-1]),
        "val_loss_final": float(val[-1]),
        "val_loss_std": float(np.std(val)),
        "val_loss_min": float(np.min(val)),
        "val_loss_max": float(np.max(val)),
    }


def bucket_table(y_true: np.ndarray, y_pred: np.ndarray, n: int = N_BUCKETS) -> list[dict]:
    """Error and bias per price bucket, split on the true price quantiles."""
    edges = np.quantile(y_true, np.linspace(0, 1, n + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    # quantiles can tie; collapse duplicate edges so buckets stay non-empty
    edges = np.unique(edges)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_true >= lo) & (y_true < hi)
        if not m.any():
            continue
        t, p = y_true[m], y_pred[m]
        rows.append(
            {
                "price_lo": None if np.isneginf(lo) else round(float(lo), 1),
                "price_hi": None if np.isposinf(hi) else round(float(hi), 1),
                "n": int(m.sum()),
                "true_mean": round(float(t.mean()), 1),
                "pred_mean": round(float(p.mean()), 1),
                "mse": round(float(np.mean((t - p) ** 2)), 1),
                "mae": round(float(np.mean(np.abs(t - p))), 1),
                "bias": round(float(np.mean(p - t)), 1),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="E3_320px", help="which fine-tuned run to analyse")
    parser.add_argument("--buckets", type=int, default=N_BUCKETS)
    args = parser.parse_args()

    set_seeds(SEED)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    _, val_df = load_split()
    y_true = val_df[PRICE_COL].to_numpy(np.float32)

    print(f"Loading {args.tag}")
    model, img_size, results, stats = load_run(args.tag)
    print(f"  model resolution {img_size}px")

    ds = make_dataset(val_df[IMAGE_COL].tolist(), TRAIN_IMG_DIR, img_size)
    raw = model.predict(ds, verbose=0).ravel()
    y_pred = np.clip(raw * stats["std"] + stats["mean"], stats["min"], stats["max"])

    overall = summary(y_true, y_pred)
    resid = y_pred - y_true
    print()
    print(format_summary(f"{args.tag} validation", overall))

    # --- overfitting -------------------------------------------------------
    of = overfitting_report(results)
    print("\nOverfitting (from the recorded training history)")
    if of["available"]:
        print(f"  epochs run          {of['epochs_run']}, best at {of['best_epoch']}")
        print(f"  train loss @ best   {of['train_loss_at_best']:.4f}")
        print(f"  val   loss @ best   {of['val_loss_at_best']:.4f}")
        print(f"  ratio (train/val)   {of['gap_ratio_at_best']:.3f}   <- far below 1 means memorising")
        print(f"  val loss min / max  {of['val_loss_min']:.4f} / {of['val_loss_max']:.4f}")

    # --- error by price ----------------------------------------------------
    buckets = bucket_table(y_true, y_pred, args.buckets)
    print("\nError by true price bucket")
    print(f"  {'range':>18} {'n':>5} {'true mu':>9} {'pred mu':>9} {'MAE':>8} {'MSE':>10} {'bias':>8}")
    for b in buckets:
        lo = "-inf" if b["price_lo"] is None else f"{b['price_lo']:.0f}"
        hi = "+inf" if b["price_hi"] is None else f"{b['price_hi']:.0f}"
        print(f"  {lo:>8}-{hi:<8} {b['n']:>5} {b['true_mean']:>9.1f} {b['pred_mean']:>9.1f} "
              f"{b['mae']:>8.1f} {b['mse']:>10.1f} {b['bias']:>+8.1f}")

    # --- residuals ---------------------------------------------------------
    print("\nResiduals (predicted - true)")
    print(f"  mean {resid.mean():+.1f}   std {resid.std():.1f}   "
          f"min {resid.min():+.1f}   max {resid.max():+.1f}")
    print(f"  underpredictions (resid < 0): {100*(resid < 0).mean():.1f}%")

    # --- per-image table, sorted, so figures can be built from real rows ----
    per_image = pd.DataFrame(
        {
            IMAGE_COL: val_df[IMAGE_COL].tolist(),
            "price_true": y_true,
            "price_pred": np.round(y_pred, 1),
            "residual": np.round(resid, 1),
            "abs_error": np.round(np.abs(resid), 1),
        }
    ).sort_values("abs_error")

    pred_path = PRED_DIR / f"val_predictions_{args.tag}.csv"
    per_image.to_csv(pred_path, index=False)

    examples = {
        "best": per_image.head(8).to_dict("records"),
        "worst": per_image.tail(8).iloc[::-1].to_dict("records"),
    }

    out = {
        "phase": 5,
        "tag": args.tag,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "img_size": img_size,
        "n_val": int(len(y_true)),
        "overall": overall,
        "overfitting": of,
        "buckets": buckets,
        "residuals": {
            "mean": round(float(resid.mean()), 2),
            "std": round(float(resid.std()), 2),
            "min": round(float(resid.min()), 2),
            "max": round(float(resid.max()), 2),
            "frac_underpredicted": round(float((resid < 0).mean()), 4),
        },
        "examples": examples,
    }
    out_path = LOG_DIR / f"error_analysis_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nwrote {out_path.name}")
    print(f"wrote {pred_path.name}  ({len(per_image)} rows, sorted by absolute error)")
    print("\nWorst 5:")
    for r in examples["worst"][:5]:
        print(f"  {r['imageid']:>10}  true {r['price_true']:>7.1f}  pred {r['price_pred']:>7.1f}  resid {r['residual']:>+8.1f}")


if __name__ == "__main__":
    main()
