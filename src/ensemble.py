"""Average the test predictions of several trained runs.

    python -m src.ensemble                                     # E2 + E3 + E4
    python -m src.ensemble --tags E3_320px E4_augment

Why averaging works here
------------------------
The three fine-tuned runs differ in input resolution (224 vs 320) and in
whether augmentation was used, so they make partly different errors on the
same images. Averaging cancels the part of the error that is independent
between them, and keeps the part they share. On validation:

    E2 fine-tune 224px     102,746.2   R2 0.335
    E3 fine-tune 320px      99,051.2   R2 0.359
    E4 augmented           108,784.2   R2 0.296
    E2+E3+E4 mean           95,850.6   R2 0.379   (+3.23% over the best single)

On the Kaggle test set the ensemble scored 96,958.98 against E3 alone at
101,594.15 -- a 4.56% gain, larger than validation predicted. That is the
expected direction: averaging reduces variance, and a single 1000-image
validation split is precisely what is bad at measuring variance.

Not every subset helps. E2+E4 is 0.66% worse than E3 alone, because E4 is
the weakest run and its errors are not different enough to compensate. All
three is the best combination measured.

This module only averages already-produced prediction files; it does not
train anything or touch the test images.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import IMAGE_COL, PRED_DIR, PRICE_COL

DEFAULT_TAGS = ["E2_finetune", "E3_320px", "E4_augment"]
OUT_NAME = "submission_ensemble.csv"


def submission_path(tag: str):
    return PRED_DIR / f"submission_phase3_{tag}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    parser.add_argument("--out", default=OUT_NAME)
    args = parser.parse_args()

    frames = {}
    for tag in args.tags:
        path = submission_path(tag)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Generate it with "
                f"`python -m src.predict --model finetuned --tag {tag}`."
            )
        frames[tag] = pd.read_csv(path)

    # Every file must describe the same test images, in the same order.
    reference_ids = frames[args.tags[0]][IMAGE_COL].astype(str).tolist()
    for tag, df in frames.items():
        if df[IMAGE_COL].astype(str).tolist() != reference_ids:
            raise ValueError(f"{tag} has a different image id order to {args.tags[0]}")
        if df[PRICE_COL].isna().any():
            raise ValueError(f"{tag} contains NaN predictions")

    prices = np.mean([df[PRICE_COL].to_numpy() for df in frames.values()], axis=0)
    out = pd.DataFrame({IMAGE_COL: reference_ids, PRICE_COL: np.round(prices, 1)})

    out_path = PRED_DIR / args.out
    out.to_csv(out_path, index=False)

    print(f"averaged {len(frames)} runs: {', '.join(args.tags)}")
    for tag, df in frames.items():
        print(f"  {tag:<14} mean {df[PRICE_COL].mean():7.1f}")
    print(f"\nwrote {out_path}")
    print(f"  rows        {len(out)}")
    print(f"  price range {prices.min():.1f} - {prices.max():.1f}")
    print(f"  price mean  {prices.mean():.1f}")


if __name__ == "__main__":
    main()
