# Baselines

Scored on 8,556 canvass inspections in the validation window (2024-10-30 to 2025-10-29), with a Fail base rate of 0.223. Rate baselines use category Fail rates from 44,596 canvass inspections in the training window, and the logistic regressions train on all 91,344 training inspections. The headline metric is precision_at_10, and se_at_headline is its standard error.

| baseline | precision_at_5 | precision_at_10 | precision_at_20 | se_at_headline | lift_per_100 | roc_auc | pr_auc |
|---|---|---|---|---|---|---|---|
| logistic_regression_v2 | 0.471 | 0.425 | 0.385 | 0.017 | 20.168 | 0.672 | 0.359 |
| logistic_regression_v1 | 0.410 | 0.378 | 0.341 | 0.017 | 15.489 | 0.638 | 0.320 |
| rate:zip | 0.290 | 0.318 | 0.267 | 0.016 | 9.524 | 0.542 | 0.252 |
| rate:prior_fails_capped | 0.304 | 0.292 | 0.286 | 0.016 | 6.951 | 0.546 | 0.251 |
| rate:prev_result | 0.258 | 0.281 | 0.295 | 0.015 | 5.782 | 0.542 | 0.242 |
| rate:risk | 0.225 | 0.244 | 0.230 | 0.015 | 2.156 | 0.504 | 0.224 |
| rate:facility_group | 0.227 | 0.233 | 0.230 | 0.014 | 0.986 | 0.515 | 0.229 |
| random | 0.218 | 0.232 | 0.225 | 0.014 | 0.869 | 0.498 | 0.223 |
