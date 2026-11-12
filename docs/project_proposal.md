# Project Proposal: Chicago Restaurant Inspection Prioritization

## Team

- John Henry Oliphant
- Brooke Rios
- Mpiira Tabuti

## Dataset

The City of Chicago Food Inspections dataset (Chicago Data Portal, `4ijn-s7e5`) records 317,961 inspections from January 2010 to October 2026. Each row lists the license, facility type, risk category, inspection type, result, violations, and location. Pass, Pass w/ Conditions, and Fail results cover 272,941 rows, and 22.4% of those are Fail.

## Problem statement

Chicago has a limited number of inspectors, so the pipeline predicts the probability that an establishment will fail its next inspection and ranks establishments for routine visits. Features come only from earlier inspections of the same license: previous result, count of earlier fails, days since the last visit, risk category, facility type, a neighborhood fail rate, and recent complaint inspections nearby.

The model is evaluated by precision@k, the fail rate among the k highest-ranked inspections, because it measures how many failures a fixed inspection budget finds. The best single-feature baseline reaches 0.318 against a base rate of 0.223. Training uses inspections from July 2018 onward, and the final 12 months are held out as the test set until production validation.

## Proposed toolstack

| Stage | Tool |
|---|---|
| Orchestration | Prefect (ingestion, features, training) |
| Tracking and AutoML | MLflow with FLAML |
| Registry | MLflow Model Registry, semantic versions |
| Serving | FastAPI in Docker |
| Monitoring | Evidently, against the training reference |

```
Data portal export -> Prefect (ingest, features, train) -> MLflow (tracking, registry)
                   -> FastAPI in Docker -> Evidently drift reports and alerts
```

## Drift simulation

Monitoring is validated first on the clean test set, then on corrupted copies with out-of-range values, swapped columns, and an altered schema. The July 2018 food code change renumbered the violation codes, which gives a real schema shift: inspections from before the change can be replayed through the deployed model as a natural drift batch.

Assisting sources: Claude Opus 5.5 (Anthropic).
