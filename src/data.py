"""Data loading and the frozen train/validation split.

Phase 1 deliverable (Member B): a deterministic, documented split that every
experiment reuses, plus the path handling that makes the code reproducible on
another machine.

Design notes
------------
* The split is **stratified by price decile**. Prices in this dataset are
  right-skewed (median 640, mean 721, range 195-2000), so a plain random split
  can leave the validation set thin at the tails -- exactly the price ranges
  the error analysis needs to say something about. Stratifying keeps the
  validation price distribution representative of the training one.
* The split is saved to CSV and committed. It is frozen by construction: every
  experiment loads the same files instead of re-deriving a split, so results
  stay comparable.
* Only the course-provided *training* labels are read here. The test set ships
  without labels in this repository, and this module never loads it.

Usage
-----
    python -m src.data              # verify the existing split (read-only)
    python -m src.data --rebuild    # regenerate and overwrite it

Verifying is the default and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    IMAGE_COL,
    N_PRICE_BINS,
    PRICE_COL,
    SEED,
    SPLIT_DIR,
    TRAIN_CSV,
    TRAIN_IMG_DIR,
    VAL_FRACTION,
)

TRAIN_SPLIT_CSV = SPLIT_DIR / "train_split.csv"
VAL_SPLIT_CSV = SPLIT_DIR / "val_split.csv"
SPLIT_META_JSON = SPLIT_DIR / "split_meta.json"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_train_labels(csv_path=TRAIN_CSV) -> pd.DataFrame:
    """Read the course training labels and validate their basic shape."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training labels not found at {csv_path}.\n"
            "Set the CV_DATA_ROOT environment variable to the folder holding "
            "train.csv / test.csv (see README)."
        )

    df = pd.read_csv(csv_path)

    missing = {IMAGE_COL, PRICE_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    if df[IMAGE_COL].duplicated().any():
        raise ValueError(f"{csv_path} contains duplicate {IMAGE_COL} values")
    if df[PRICE_COL].isna().any():
        raise ValueError(f"{csv_path} contains missing {PRICE_COL} values")

    return df


def image_path(image_id: str, split: str = "train", data_root=None):
    """Absolute path to an image, e.g. image_path('1.jpg', 'train')."""
    if data_root is None:
        from config import TRAIN_IMG_DIR, TEST_IMG_DIR

        data_root = TRAIN_IMG_DIR if split == "train" else TEST_IMG_DIR
    return data_root / image_id


# --------------------------------------------------------------------------
# Split construction
# --------------------------------------------------------------------------
def build_split(df: pd.DataFrame, val_fraction: float = VAL_FRACTION, seed: int = SEED):
    """Return (train_df, val_df) sharing the same row schema as ``df``.

    Stratification uses price deciles so the validation set covers the whole
    price range in the same proportion as the training set.
    """
    # qcut can drop bins if prices are heavily tied; duplicates='drop' handles
    # that, and the number of bins actually used is recorded in split_meta.json.
    bins = pd.qcut(df[PRICE_COL], q=N_PRICE_BINS, labels=False, duplicates="drop")

    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=bins,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def save_split(train_df: pd.DataFrame, val_df: pd.DataFrame, extra: dict | None = None) -> None:
    """Persist the split plus a metadata record describing how it was made."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(VAL_SPLIT_CSV, index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "n_price_bins": N_PRICE_BINS,
        "stratified_by": f"{PRICE_COL} decile",
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "train_price_stats": _stats(train_df[PRICE_COL]),
        "val_price_stats": _stats(val_df[PRICE_COL]),
        "source_csv": str(TRAIN_CSV),
    }
    if extra:
        meta.update(extra)

    SPLIT_META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote {TRAIN_SPLIT_CSV.relative_to(SPLIT_DIR.parent.parent)}  ({len(train_df)} rows)")
    print(f"wrote {VAL_SPLIT_CSV.relative_to(SPLIT_DIR.parent.parent)}  ({len(val_df)} rows)")
    print(f"wrote {SPLIT_META_JSON.relative_to(SPLIT_DIR.parent.parent)}")


def _stats(series: pd.Series) -> dict:
    return {
        "mean": round(float(series.mean()), 2),
        "std": round(float(series.std()), 2),
        "min": round(float(series.min()), 2),
        "median": round(float(series.median()), 2),
        "max": round(float(series.max()), 2),
    }


def load_split():
    """Load the frozen split. Every experiment must go through this."""
    if not TRAIN_SPLIT_CSV.exists() or not VAL_SPLIT_CSV.exists():
        raise FileNotFoundError(
            "Split files not found. Run `python -m src.data --rebuild` to create them."
        )
    return pd.read_csv(TRAIN_SPLIT_CSV), pd.read_csv(VAL_SPLIT_CSV)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def verify(df: pd.DataFrame) -> bool:
    """Check the frozen split is well formed and leak-free."""
    train_df, val_df = load_split()
    problems: list[str] = []

    train_ids = set(train_df[IMAGE_COL])
    val_ids = set(val_df[IMAGE_COL])

    overlap = train_ids & val_ids
    if overlap:
        problems.append(f"{len(overlap)} image IDs appear in both splits")

    covered = train_ids | val_ids
    expected = set(df[IMAGE_COL])
    if covered != expected:
        problems.append(
            f"split covers {len(covered)} of {len(expected)} training images"
        )

    if len(train_df) + len(val_df) != len(df):
        problems.append("train + val row count does not match the source labels")

    if problems:
        print("SPLIT VERIFICATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        return False

    print("Split verification passed:")
    print(f"  train {len(train_df)}  val {len(val_df)}  total {len(train_df) + len(val_df)}")
    print(f"  no overlap, full coverage of {len(expected)} training images")
    print(f"  train price mean {train_df[PRICE_COL].mean():.1f} | "
          f"val price mean {val_df[PRICE_COL].mean():.1f}")
    return True


def main() -> None:
    """Verify the frozen split, or rebuild it with --rebuild.

    Verifying is the default and does not write anything. This matters
    because the split is a shared commitment: every experiment in the
    project, and both team members' results, are only comparable while the
    same file is in use. A command that silently rewrote it whenever someone
    asked it to "check" the data would be a trap.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="regenerate and overwrite the committed split (default: verify only)",
    )
    args = parser.parse_args()

    df = load_train_labels()
    print(f"loaded {len(df)} training labels from {TRAIN_CSV}")

    if args.rebuild:
        print("--rebuild given: regenerating the split")
        train_df, val_df = build_split(df)
        save_split(train_df, val_df)
    elif not (TRAIN_SPLIT_CSV.exists() and VAL_SPLIT_CSV.exists()):
        sys.exit(
            "No split found. Run `python -m src.data --rebuild` to create it.\n"
            "(Note this is a different decision from a new teammate's first "
            "checkout: the split is committed, so a plain clone already has it.)"
        )
    else:
        print("verifying the existing split; pass --rebuild to regenerate it")

    sys.exit(0 if verify(df) else 1)


if __name__ == "__main__":
    main()
