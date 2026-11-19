# Model comparison

Facility scope: restaurant. AutoML selected each configuration on 5,910 canvass inspections in the inner holdout (training rows from 2023-10-30), shown as inner_weekly_precision_at_k. The other columns score each model once on 6,099 canvass inspections in the validation window, with k = 9 visits per week and a base rate of 0.228. The feature version with the higher inner score is the challenger, and the gate reads its validation score.

| model | inner_weekly_precision_at_k | weekly_precision_at_k | se_headline | lift_per_100 | roc_auc |
|---|---|---|---|---|---|
| best baseline (logistic_regression_v2) |  | 0.449 | 0.023 | 22.090 | 0.680 |
| AutoML v2 (lgbm) | 0.476 | 0.447 | 0.023 | 21.880 | 0.672 |
| AutoML v1 (xgboost) | 0.447 | 0.405 | 0.022 | 17.687 | 0.646 |

Gate: challenger 0.447 does not beat baseline logistic_regression_v2 (0.449 + 1 SE = 0.471). No production model existed, so the best baseline (logistic_regression v2) was registered as version 1 (1.0.0) and is now production. Later challengers must beat it.
