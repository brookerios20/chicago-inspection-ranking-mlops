# Model card: inspection-ranker 1.0.0

Following the model card structure of Mitchell et al. (2019), covered in Session 6.

## Model details

- **Name and version:** `inspection-ranker`, MLflow registry version 1, semantic version `1.0.0`, alias `production`
- **Type:** logistic regression (scikit-learn, `max_iter=1000`) inside one scikit-learn pipeline with its preprocessing: one-hot encoding of 4 categorical features, median imputation with missing-value flags and scaling of 6 numeric features (21 model inputs in total)
- **Features (version v2):** inspection group, facility group, risk category, previous result, earlier fails, earlier inspections, days since the previous inspection, violations at the previous inspection, the ZIP code's fail rate over the last 365 days and its complaint inspections over the last 90 days, all computed from earlier days only
- **Data version:** `data-v2` (restaurants only), City of Chicago Food Inspections (dataset `4ijn-s7e5`), snapshot ending 2026-10-29
- **How it was chosen:** a FLAML AutoML search over six algorithms did not beat this baseline by one standard error, so the training pipeline registered the baseline (promotion tag `bootstrap_baseline`)

## Intended use

- **Primary use:** ranking each week's routine (canvass) restaurant inspections in Chicago by their predicted probability of failing, so that inspectors visit the highest-risk restaurants first. It supports the scheduling decision; inspectors still carry out every inspection and decide its result.
- **Out of scope:** complaint, license and re-inspection visits (triggered by events, not scheduled), facility types other than restaurants, and any use that penalizes a restaurant based on its score.

## Factors

The evaluation looks at results by region (North vs South Side, split at Roosevelt Road), by the city's risk category and by inspection history (first inspection vs repeat), the slices in `src/metrics.py`.

## Metrics

- **Headline:** weekly precision@k: each week the model ranks that week's canvass inspections, the top k = 9 (10% of the median 93 canvass inspections per week) count as visits, and the metric is the share of those visits that find a failing restaurant
- **Also reported:** standard error of the headline, lift per 100 visits over random selection, precision at 5%, 10% and 20% of the window, ROC-AUC, PR-AUC

## Training and evaluation data

- **Training:** 62,070 restaurant inspections from 2018-07-01 to 2024-10-29; the registered model was refit on the training and validation rows
- **Validation:** 6,099 canvass inspections from 2024-10-30 to 2025-10-29 (base fail rate 0.228)
- **Test:** inspections from 2025-10-30 to 2026-10-29, held out for production validation through the API (Part D)

## Quantitative analysis (validation window)

| | Weekly precision@k | Notes |
|---|---|---|
| This model | 0.449 (standard error 0.023) | about 22 more failures per 100 routine visits than random selection |
| Random ranking | 0.197 | |
| City risk category alone | 0.237 | |

| Region | Canvass inspections | Fail rate | Selection rate | Precision of picks | Share of failures found |
|---|---|---|---|---|---|
| North Side | 4,337 | 0.242 | 0.070 | 0.455 | 0.131 |
| South Side | 1,756 | 0.192 | 0.099 | 0.437 | 0.225 |

Demographic parity difference (selection rate gap): 0.029.

## Ethical considerations

- **Uneven attention across the city:** South Side restaurants are selected more often than North Side restaurants while failing less often in the validation period, likely through the ZIP code features. The city should decide whether this gap is acceptable; the monitoring stage re-checks it.
- **Feedback loop:** if the city follows the ranking, the restaurants it rates as risky are inspected more often, so future training data over-represents them. Retraining should keep a share of randomly chosen inspections to counter this.
- **Privacy:** business names and addresses are not model inputs, and the API's requests carry only a client ID and the ten features.

## Caveats and recommendations

- The July 2018 food code change renumbered the violation codes; the model trains only on inspections from that date on, and the monitoring stage watches for similar changes in the source data.
- Scores are probabilities from one city and one period; they should be read as a ranking within a week, not as a fixed risk level.
- The model should be re-checked (performance, region gap and drift) whenever new inspection data is added.
