# Baselines

Facility scope: all. Scored on 8,556 canvass inspections in the validation window (2024-10-30 to 2025-10-29), with a Fail base rate of 0.223.

The headline, weekly_precision_at_k, ranks each week's canvass inspections and scores the top 14. k is 10% of the median 136 canvass inspections per week in the training window, where the median for all inspection types is 280 per week. se_headline is its standard error, and the precision_at columns rank the whole window at once.

Rate baselines use category Fail rates from 44,596 canvass inspections in the training window, and the logistic regressions train on all 91,344 training inspections.

| baseline | weekly_precision_at_k | se_headline | lift_per_100 | precision_at_5 | precision_at_10 | precision_at_20 | roc_auc | pr_auc |
|---|---|---|---|---|---|---|---|---|
| logistic_regression_v2 | 0.430 | 0.018 | 20.703 | 0.471 | 0.425 | 0.385 | 0.672 | 0.359 |
| logistic_regression_v1 | 0.394 | 0.018 | 17.065 | 0.410 | 0.378 | 0.341 | 0.638 | 0.320 |
| rate:prev_result | 0.303 | 0.017 | 8.035 | 0.258 | 0.281 | 0.295 | 0.542 | 0.242 |
| rate:prior_fails_capped | 0.303 | 0.017 | 8.035 | 0.304 | 0.292 | 0.286 | 0.546 | 0.251 |
| rate:zip | 0.298 | 0.017 | 7.496 | 0.290 | 0.318 | 0.267 | 0.542 | 0.252 |
| rate:facility_group | 0.237 | 0.016 | 1.431 | 0.227 | 0.233 | 0.230 | 0.515 | 0.229 |
| random | 0.228 | 0.015 | 0.488 | 0.218 | 0.232 | 0.225 | 0.498 | 0.223 |
| rate:risk | 0.212 | 0.015 | -1.129 | 0.225 | 0.244 | 0.230 | 0.504 | 0.224 |
