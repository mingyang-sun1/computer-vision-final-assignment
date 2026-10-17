# Phase 5 — E5 Validation Error Analysis

Generated: 2026-10-17T05:46:46+00:00  
Source: `docs/evidence/phase4_run_20261016T124243_097811Z/validation_ensemble.csv`  
Model: `E5_mean_export_1dp`; equal weights for E2, E3 and E4.  
Rows: 1,000 validation; 7,000 disjoint training IDs.

## Scope and precision

Descriptive analysis of the existing E5 candidate on the fixed validation split.

Decimal(50) arithmetic on exact CSV label fields and one-decimal prediction exports; E5 equals the three-component mean rounded with ROUND_HALF_EVEN to 0.1. JSON/CSV results are finite floats; original model precision is not recovered.

Prices, RMSE, MAE and signed bias use USD 1,000. MSE and SSE use squared USD 1,000 units. A positive residual means overprediction; a negative residual means underprediction.

## Overall errors

| Statistic | Value |
|---|---:|
| MSE | 95,852.002676 |
| RMSE | 309.599746 |
| MAE | 226.423020 |
| Mean signed bias | 6.912374 |
| Median absolute error | 173.600000 |
| 90th percentile absolute error | 460.280000 |
| R2 | 0.379296 |
| Overprediction | 59.40% |
| Underprediction | 40.60% |
| Exact prediction | 0.00% |

## Errors by true-price interval

Linear quantiles at (n - 1) * q; duplicate edges collapsed; empty intervals omitted.

Lower inclusive, upper exclusive, except the final upper edge is inclusive.

Data-derived diagnostic bins based on validation true prices. Do not use these results to tune a model and then claim an independent validation estimate.

| Bin | True-price interval | Count | MAE | MSE | Bias | Share of total SSE |
|---|---|---:|---:|---:|---:|---:|
| B01 | [195, 299) | 93 | 252.989 | 78,867.548 | +250.759 | 7.65% |
| B02 | [299, 396.6) | 107 | 235.193 | 68,943.279 | +234.298 | 7.70% |
| B03 | [396.6, 479.997) | 100 | 196.653 | 55,987.686 | +182.177 | 5.84% |
| B04 | [479.997, 559.6) | 100 | 146.302 | 36,149.512 | +129.250 | 3.77% |
| B05 | [559.6, 639.45) | 100 | 141.527 | 35,598.048 | +112.551 | 3.71% |
| B06 | [639.45, 718.9328) | 100 | 115.565 | 22,308.471 | +15.589 | 2.33% |
| B07 | [718.9328, 799) | 92 | 122.951 | 24,138.351 | +21.275 | 2.32% |
| B08 | [799, 940.92) | 108 | 166.230 | 41,520.884 | -42.256 | 4.68% |
| B09 | [940.92, 1299.91) | 100 | 234.393 | 84,960.642 | -179.809 | 8.86% |
| B10 | [1299.91, 1999.9] | 100 | 650.212 | 509,349.703 | -648.476 | 53.14% |

All interval statistics, means, RMSE, error-direction fractions and SSE are in `price_buckets.csv`. Unequal interval counts can result from tied prices.

## Paired E3 comparison

Descriptive paired comparison on the same split; not a significance test.

E5 has lower absolute error on **515** rows, equal error on **0**, and higher error on **485**. E5 minus E3 MSE is **-3,199.194382**.

| Bin | E3 MSE | E5 MSE | E5 minus E3 MSE | E3 bias | E5 bias |
|---|---:|---:|---:|---:|---:|
| B01 | 75,090.706 | 78,867.548 | +3,776.842 | +226.755 | +250.759 |
| B02 | 66,757.399 | 68,943.279 | +2,185.880 | +214.950 | +234.298 |
| B03 | 68,267.764 | 55,987.686 | -12,280.079 | +186.527 | +182.177 |
| B04 | 44,213.016 | 36,149.512 | -8,063.504 | +119.714 | +129.250 |
| B05 | 46,838.590 | 35,598.048 | -11,240.542 | +125.576 | +112.551 |
| B06 | 33,628.121 | 22,308.471 | -11,319.651 | +22.847 | +15.589 |
| B07 | 34,233.143 | 24,138.351 | -10,094.792 | +25.467 | +21.275 |
| B08 | 43,584.513 | 41,520.884 | -2,063.630 | -21.878 | -42.256 |
| B09 | 89,478.578 | 84,960.642 | -4,517.936 | -162.635 | -179.809 |
| B10 | 488,255.362 | 509,349.703 | +21,094.341 | -618.418 | -648.476 |

In the highest true-price interval, E5 minus E3 MSE is **+21,094.341**. Aggregate performance should therefore be assessed alongside the interval results, not treated as a guarantee for every price range.

## Error thresholds

Counts with absolute error <= 50 or <= 100 (USD 1,000); diagnostic summaries, not course acceptance criteria.

- Absolute error <= 50: 166 rows (16.6%).
- Absolute error <= 100: 304 rows (30.4%).

## Examples for subsequent visual inspection

Up to three per category, in order: smallest absolute error, largest positive residual, largest negative residual. No repeated imageid; ties use lexical imageid order. These are extremes, not typical cases.

| Category | Rank | Image ID | True | Predicted | Residual | Absolute error |
|---|---:|---|---:|---:|---:|---:|
| best_absolute | 1 | `751.jpg` | 759.000 | 758.200 | -0.800 | 0.800 |
| best_absolute | 2 | `1770.jpg` | 599.000 | 600.800 | +1.800 | 1.800 |
| best_absolute | 3 | `5703.jpg` | 650.000 | 648.200 | -1.800 | 1.800 |
| largest_overprediction | 1 | `3947.jpg` | 420.000 | 1,026.700 | +606.700 | 606.700 |
| largest_overprediction | 2 | `3709.jpg` | 629.900 | 1,235.800 | +605.900 | 605.900 |
| largest_overprediction | 3 | `1416.jpg` | 485.000 | 1,054.500 | +569.500 | 569.500 |
| largest_underprediction | 1 | `1977.jpg` | 1,995.000 | 545.900 | -1,449.100 | 1,449.100 |
| largest_underprediction | 2 | `2138.jpg` | 1,995.000 | 577.200 | -1,417.800 | 1,417.800 |
| largest_underprediction | 3 | `7737.jpg` | 1,999.000 | 700.200 | -1,298.800 | 1,298.800 |

No visual explanation is asserted by this numerical report. `representative_examples.csv` also records signed and absolute relative errors as fractions.

## Limits

- One fixed split and previously selected models; no repeated-seed uncertainty or independent test estimate.
- Aggregate improvement does not imply improvement for every price range or image.
- Images were not opened by this numerical analysis; image-content explanations require visual inspection.
- Training/validation learning curves were not re-evaluated; these residuals alone do not establish overfitting.

## Source identities

Both hashes below normalize Windows CRLF to LF before hashing. Source bytes and input identity checks are recorded in `summary.json`.

| Input path | LF-normalized SHA-256 |
|---|---|
| `data/splits/train_split.csv` | `821c4be4f30087f70aa16b2c7b5735908da31c3d0745230315b9e4d5f0a59774` |
| `data/splits/val_split.csv` | `3a8236908586bdf8ceedd002130883425a8cf8295e3dc6d43c9b9db9e8ac3402` |
| `outputs/predictions/val_predictions_E2_finetune.csv` | `d123bf1db4b1d0b2737691ac19fb9e154d30a83ae875cf87845af84f7fe61738` |
| `outputs/predictions/val_predictions_E3_320px.csv` | `965a5ce2a1f17b8c44b037b6f48dbc42b7d7dedc0c4cf315aaf6df5c19f5cdf5` |
| `outputs/predictions/val_predictions_E4_augment.csv` | `64e02604747ee642f56e57b7c221b866d2beb7fec8ed4b3125ec8fa813fde838` |
| `docs/evidence/phase4_run_20261016T124243_097811Z/validation_ensemble.csv` | `4fcbb7ae209145b85ca055cbef12237739994f3d4c577a7c74118573233ae110` |

Reference repository commit: `ae74826454627addbeabdec5905595a9599fc105`.  
Phase 4 reference commit: `90959c41384c7bd3c8c5a48147e8272246071bbe`.  
Current local HEAD: `ae74826454627addbeabdec5905595a9599fc105`.
