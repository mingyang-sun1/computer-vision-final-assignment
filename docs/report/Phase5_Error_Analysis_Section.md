# Phase 5 — Error Analysis Text for the Final Report

This is a concise Member A section for the Phase 6 IEEE draft. Figures and
references must be numbered consistently when integrating the final paper.
All numerical claims below use the executed E5 analysis, not the historical
E3 table. The full traceable source is [Phase5_Error_Analysis.md](../Phase5_Error_Analysis.md).

## Error Analysis

We analysed the equal-weight E2/E3/E4 ensemble on the fixed 1,000-image
validation split. Predictions were matched by image identifier and retained
at the stored one-decimal export precision. The ensemble achieved MSE
95,852.00, RMSE 309.60 and MAE 226.42, where prices and absolute errors are
measured in USD 1,000. MSE was 3.23% lower than the best single model, E3,
but absolute error improved on only 515 of the 1,000 images.

Grouping images by true-price deciles reveals systematic errors hidden by
the small overall signed bias of +6.91. The lowest-price 30% were overestimated
by 222.03 on average. The highest-price 10% were underestimated by 648.48
and contributed 53.14% of total squared error. Their MSE was 509,349.70,
4.32% higher than E3 in the same group. Thus, averaging improved overall
performance while making the expensive tail less accurate. The lowest
group MAE, 115.56, occurred between true prices of 639.45 and 718.9328.

We selected three examples each with the smallest absolute errors, largest
overestimates and largest underestimates. For example, 751.jpg was predicted
at 758.2 against a true price of 759.0. In contrast, 3947.jpg was overestimated
by 606.7 and 1977.jpg was underestimated by 1449.1. These deliberately selected
examples illustrate extremes and are not representative estimates of average
performance.

The group means show predictions concentrated near central prices. This
may reflect limited information in exterior photographs, representation
limits or fitting choices; the analysis does not establish a unique cause.
Potential improvements include controlled comparisons of target transformations,
sampling policies and spatial representations, evaluated using original-unit
MSE. These findings are descriptive results from one validation split already
used for model development, rather than independent evidence of generalisation.

## Compact table available for integration

| True-price group | n | Mean signed error | Share of squared error |
|---|---:|---:|---:|
| Lowest 30% | 300 | +222.03 | 21.19% |
| Middle 50% | 500 | +46.27 | 16.81% |
| Ninth decile | 100 | −179.81 | 8.86% |
| Highest 10% | 100 | −648.48 | 53.14% |

*Prices and signed errors are in USD 1,000. Groups use true-price quantiles
with tied prices kept together; exact boundaries are in the analysis evidence.*

## Figure captions and integration notes

**Numerical figure:** Validation errors of the E5 ensemble by true-price
decile. Mean absolute error, signed bias, contribution to squared error and
mean true/predicted prices show that the largest errors occur in the expensive
tail despite a small overall bias.

Use [phase5_price_buckets.pdf](../figures/phase5_price_buckets.pdf) for vector
placement, or the matching SVG. The supplementary
[prediction diagnostics](../figures/phase5_prediction_diagnostics.pdf) show
all 1,000 predictions and the residual distribution. Select the space needed
within the final four-page content limit.

**Photo figure, after local generation:** Validation examples selected by
numerical error: three closest predictions, three largest overestimates and
three largest underestimates. Labels give image identifiers, true prices,
predicted prices and signed residuals, all in USD 1,000. These are selected
extreme cases, not a random sample.

The original photos still require local rendering and inspection. Do not add
visual explanations until the actual images have been reviewed. Keep the
generated photo panel in the written report, outside the code submission.

When integrating Phase 6, also review the current `build_docx.py` abstract,
introduction, design justification and conclusion. They contain historical
E3 wording, unsupported claims that the objective uniquely causes the errors,
and a claim that the ensemble's absolute test MSE is below its validation MSE.
Phase 4 already corrected that score comparison. The final paper should
use the qualified E5 findings above throughout; replacing only the Error
Analysis placeholder will not correct those other sections.
