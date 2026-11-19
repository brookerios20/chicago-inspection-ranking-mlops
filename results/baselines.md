# Baselines

Facility scope: restaurant. Scored on 6,099 canvass inspections in the validation window (2024-10-30 to 2025-10-29), with a Fail base rate of 0.228.

The headline, weekly_precision_at_k, ranks each week's canvass inspections and scores the top 9. k is 10% of the median 93 canvass inspections per week in the training window, where the median for all inspection types is 191 per week. se_headline is its standard error, and the precision_at columns rank the whole window at once.

Rate baselines use category Fail rates from 30,444 canvass inspections in the training window, and the logistic regressions train on all 62,070 training inspections.

| baseline | weekly_precision_at_k | se_headline | lift_per_100 | precision_at_5 | precision_at_10 | precision_at_20 | roc_auc | pr_auc |
|---|---|---|---|---|---|---|---|---|
| logistic_regression_v2 | 0.449 | 0.023 | 22.090 | 0.480 | 0.450 | 0.402 | 0.680 | 0.373 |
| logistic_regression_v1 | 0.394 | 0.022 | 16.639 | 0.414 | 0.396 | 0.360 | 0.646 | 0.336 |
| rate:prior_fails_capped | 0.335 | 0.022 | 10.769 | 0.322 | 0.304 | 0.290 | 0.539 | 0.253 |
| rate:zip | 0.287 | 0.021 | 5.947 | 0.296 | 0.315 | 0.260 | 0.546 | 0.256 |
| rate:prev_result | 0.281 | 0.021 | 5.318 | 0.316 | 0.323 | 0.307 | 0.545 | 0.249 |
| rate:risk | 0.237 | 0.019 | 0.916 | 0.247 | 0.235 | 0.246 | 0.508 | 0.231 |
| random | 0.197 | 0.018 | -3.068 | 0.197 | 0.199 | 0.219 | 0.486 | 0.221 |
