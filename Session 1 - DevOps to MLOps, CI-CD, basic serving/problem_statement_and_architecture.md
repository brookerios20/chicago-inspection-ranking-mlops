# Session 1: DevOps to MLOps, CI/CD, basic serving

#### 1.1 Problem statement

The City of Chicago inspects thousands of restaurants every year, but it has a limited number of inspectors. Since not every restaurant can be visited at once, the order of visits matters: if the restaurants most likely to have problems are visited first, more health risks are caught earlier with the same staff.

InspectRank predicts the probability that a restaurant will fail an inspection and uses these probabilities to rank the week's routine inspections from highest to lowest risk. Inspectors would then work down the list.

#### 1.2 Prediction target

Each row of the Food Inspections dataset is one inspection. The target `fail` is 1 when the result is "Fail" and 0 when it is "Pass" or "Pass w/ Conditions". Results that record no inspection outcome ("Out of Business", "No Entry", "Not Ready", "Business Not Located") are removed.

Every feature is built only from inspections on earlier days, so the model never sees information from the visit it is predicting. "Fail" alone was chosen over "Fail or Pass w/ Conditions" because, as the team's EDA found, the combined rate more than doubles after the July 2018 food code change, which would give the model years of change that has nothing to do with how restaurants behave.

#### 1.3 Evaluation metric: weekly precision@k

The city can only reorder its routine ("canvass") inspections; complaint, license and re-inspection visits are triggered by events. So each week the model ranks that week's canvass inspections, the top k are counted as visited, and the metric is the share of those visits that find a failing restaurant, pooled over all weeks of the evaluation window.

k stands in for inspector capacity: it is 10% of the median number of canvass inspections per week in the training period (93), so **k = 9 visits per week**. The value is computed from the data, so it updates if the scope or the data snapshot changes.

Overall accuracy would be a poor fit here. About 23% of inspections fail, so a model that predicts "pass" for everyone is already about 77% accurate while being useless for choosing where to send inspectors. Precision@k measures what the city actually cares about: how many of a week's visits find a problem. ROC-AUC, PR-AUC, precision at 5%, 10% and 20% of the window, and lift per 100 visits are reported alongside it.

#### 1.4 Baselines

A model is only worth deploying if it ranks inspections better than simple rules the city could use without machine learning. The team scored random ranking, single-feature rules (the city's risk category, the previous result, the restaurant's earlier fails, the ZIP code's fail rate) and two logistic regressions. They are shown in Session 2 after the data pipeline runs. The best baseline (a logistic regression) becomes the bar that a trained model must clear before it is registered (Part B).

#### 1.5 MLOps risks for this project

The Lecture 1 discussion of hidden technical debt in ML systems (Sculley et al., 2015) points to a few risks that seem especially relevant here:

- **Hidden feedback loop:** if the city follows the model's ranking, the restaurants the model rates as risky are inspected more often, so future training data contains more records about them. Over time, the model could end up learning from its own choices instead of from the real condition of all restaurants.
- **External data dependency:** the data comes from the city's open data portal, which the project does not control. The 2018 change in violation codes is a real example of the source data changing underneath a model, which is why the monitoring in Part D is needed.
- **Configuration debt:** every date, path, threshold and feature list lives in one file, `src/config.py`, so a change made there applies to the whole pipeline instead of being scattered through the code.

#### 1.6 System architecture

The project follows the three tiers of the ML lifecycle from Lecture 1:

| Lifecycle tier | What it covers | Tools in this project | Where in this notebook |
|---|---|---|---|
| DataOps | ingestion, validation, versioning, features, EDA | Python rules, DVC with a shared Google Drive remote | Part A |
| ModelOps | training pipeline, experiment tracking, AutoML, model registry | Prefect, FLAML, MLflow | Part B |
| Deployment | prediction API, container, monitoring | FastAPI, Docker, Evidently | Parts C and D |

The full diagram is in `docs/architecture.md`:

```
Chicago Data Portal (4ijn-s7e5)
        |  CSV export or API, cut at 2026-10-29
        v
Data pipeline (DVC):  ingest -> validate -> clean -> features -> split -> baselines
        |  data/processed/{train,valid,test,drift_pre2018}.parquet
        v
Training pipeline (Prefect):  update_data -> load_inputs -> search_version (v1, v2)
                              -> choose_and_gate -> register_model -> write_reports
        |                                   |
        v                                   v
   MLflow tracking                    MLflow Model Registry (models:/inspection-ranker@production)
                                            |
                                            v
                         export_model.py -> Serving API (Docker, FastAPI) -> prediction log
                                            |
                                            v
                         Monitoring (Evidently) against results/reference.parquet
```
