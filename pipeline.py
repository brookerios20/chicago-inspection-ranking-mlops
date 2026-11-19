"""Prefect flow: update the data, search for a model with AutoML, track it, and register it.

Tasks, in order:

1. update_data: `dvc repro` brings data/processed and the baselines up to date. DVC reruns a
   stage only when its inputs changed, so on an unchanged repository this step takes seconds.
2. load_inputs: the training split divided at `config.INNER_VALID_START` into the rows the
   search fits on and the inner holdout it selects by, the canvass rows of the validation split,
   k from results/capacity.json, and the best baseline. The test split is never loaded here.
3. search_version (once per feature version): FLAML searches the algorithms in
   `config.FLAML_ESTIMATORS`, fitting on the earlier training rows and scoring by weekly
   precision@k on the inner holdout. Each algorithm's best configuration is logged as its own
   MLflow run, which the MLflow UI compares side by side. The winning configuration is then
   refit on the whole training window and scored once on the validation window.
4. choose_and_gate: the feature version with the better inner-holdout score supplies the
   challenger. Its validation score must beat the best baseline, and the current production
   model if one exists, by one standard error.

Data use: model selection sees only the training window, the gate is the first and only use of
the validation window, and the test year waits for production validation. Separating them keeps
the gate's score free of the optimism that comes from picking the best of many configurations.
5. register_model: the challenger's configuration is refit on training plus validation rows,
   logged with its input signature, registered with a semantic version, and given the
   production alias. Its validation predictions are saved as results/reference.parquet, the
   reference data for drift monitoring. When the challenger fails the gate and no production
   model exists yet, the best logistic regression baseline (baseline_candidate) is registered
   instead as the first production version, and later challengers must beat it.
6. write_reports: tables and figures in results/training/.

Every MLflow run carries lineage tags (git commit, data tag, DVC hash of data/processed) from
`src.tracking.lineage_tags`.

Usage:
    python pipeline.py                         # both feature versions, register the winner
    python pipeline.py --feature-versions v2   # one feature version
    python pipeline.py --time-budget 60        # a shorter search, for a quick check
    python pipeline.py --no-register           # search and compare only
    python pipeline.py --skip-dvc              # use the data already in data/processed
"""

import argparse
import json
import subprocess
from datetime import datetime

import pandas as pd
from prefect import flow, get_run_logger, task
from sklearn.linear_model import LogisticRegression
from prefect.cache_policies import NO_CACHE

from src import automl, config, registry, tracking, train
from src.baselines import markdown_table
from src.metrics import HEADLINE
from src.split import load_split


@task(cache_policy=NO_CACHE)
def update_data(skip: bool) -> None:
    """Run `dvc repro`, so the splits and baselines match the code and the snapshot."""
    logger = get_run_logger()
    if skip:
        logger.info("Skipping dvc repro; using the data already in data/processed.")
        return
    subprocess.run(["dvc", "repro"], cwd=config.PROJECT_ROOT, check=True)


@task(cache_policy=NO_CACHE)
def load_inputs() -> dict:
    """Training rows (whole and split for selection), validation canvass rows, k, and the best baseline."""
    train_rows = load_split("train")
    valid_rows = load_split("valid")
    capacity = json.loads((config.RESULTS_DIR / "capacity.json").read_text())
    inner_fit, inner_holdout = train.inner_split(train_rows)
    return {
        "train": train_rows,
        "inner_fit": inner_fit,
        "inner_holdout": inner_holdout,
        "valid": valid_rows,
        "valid_canvass": valid_rows[valid_rows["is_canvass"]].reset_index(drop=True),
        "k_week": int(capacity["k_week"]),
        "baseline": registry.best_baseline(),
    }


@task(cache_policy=NO_CACHE)
def search_version(inputs: dict, version: str, time_budget: int, parent_id: str) -> dict:
    """AutoML search on the inner split, then one validation score, logged as an MLflow child run."""
    logger = get_run_logger()
    train_rows, valid_c, k_week = inputs["train"], inputs["valid_canvass"], inputs["k_week"]
    fit_rows, holdout = inputs["inner_fit"], inputs["inner_holdout"]

    # Selection: fit on training rows before INNER_VALID_START, score on the inner holdout.
    preprocessor = train.fit_preprocessor(fit_rows, version)
    X_fit = preprocessor.transform(train.feature_frame(fit_rows, version))
    X_holdout = preprocessor.transform(train.feature_frame(holdout, version))

    run_id = tracking.start_run(f"automl {version}", parent_id, {**tracking.lineage_tags(), "feature_version": version})
    tracking.log_params(run_id, {
        "feature_version": version, "features": config.FEATURE_COLUMNS[version],
        "time_budget_s": time_budget, "estimators": config.FLAML_ESTIMATORS, "k_week": k_week,
        "inner_valid_start": config.INNER_VALID_START.date(), "inner_fit_rows": len(fit_rows),
        "inner_holdout_canvass_rows": len(holdout), "train_rows": len(train_rows),
        "valid_canvass_rows": len(valid_c), "seed": config.SEED,
    })

    search = automl.run_search(X_fit, fit_rows[config.TARGET].to_numpy(), X_holdout,
                               holdout[config.TARGET].to_numpy(), holdout["inspection_date"], k_week, time_budget)
    summary = automl.per_estimator_summary(search).rename(
        columns={"weekly_precision_at_k": f"inner_{HEADLINE}"})
    inner_headline = 1.0 - search.best_loss

    # One run per algorithm: its best configuration and inner-holdout headline.
    for row in summary.itertuples():
        child = tracking.start_run(f"{version} {row.estimator}", run_id, {"feature_version": version,
                                                                         "estimator": row.estimator})
        tracking.log_params(child, {"estimator": row.estimator, **(row.config or {})})
        tracking.log_metrics(child, {f"inner_{HEADLINE}": getattr(row, f"inner_{HEADLINE}")})
        tracking.end_run(child)

    # Evaluation: refit the winning configuration on the whole training window, score it once on
    # the validation window. These scores feed the gate and the monitoring reference.
    model = train.refit(automl.best_estimator(search), train_rows, version)
    scores, metrics, slices = train.evaluate(model, valid_c, version, k_week)
    tracking.set_tags(run_id, {"best_estimator": search.best_estimator})
    tracking.log_metrics(run_id, {f"inner_{HEADLINE}": inner_headline, **metrics})

    out_dir = config.TRAINING_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.assign(feature_version=version).to_csv(out_dir / f"automl_summary_{version}.csv", index=False)
    slices.to_csv(out_dir / f"slices_{version}.csv", index=False)
    tracking.log_artifact(run_id, out_dir / f"automl_summary_{version}.csv")
    tracking.log_artifact(run_id, out_dir / f"slices_{version}.csv")
    importance = train.plot_feature_importance(model, out_dir / f"feature_importance_{version}.png")
    if importance:
        tracking.log_artifact(run_id, importance)
    tracking.end_run(run_id)

    logger.info("%s: best %s, inner holdout %.3f, validation %s %.3f (se %.3f)", version, search.best_estimator,
                inner_headline, HEADLINE, metrics[HEADLINE], metrics["se_headline"])
    return {"version": version, "run_id": run_id, "estimator_name": search.best_estimator,
            "estimator": automl.best_estimator(search), "model": model, "scores": scores,
            "inner_headline": inner_headline, "metrics": metrics, "summary": summary}


@task(cache_policy=NO_CACHE)
def choose_and_gate(results: list[dict], baseline: registry.Bar) -> tuple[dict, bool, str]:
    """The feature version with the better inner-holdout score, and whether it passes the gate.

    The choice uses the inner holdout, so the validation score the gate reads has not been used
    to pick anything.
    """
    logger = get_run_logger()
    winner = max(results, key=lambda r: r["inner_headline"])
    production = registry.production_bar(registry.production_version())
    passed, reason = registry.gate(winner["metrics"][HEADLINE], [baseline, production])
    logger.info("Gate %s: %s", "passed" if passed else "failed", reason)
    return winner, passed, reason


@task(cache_policy=NO_CACHE)
def baseline_candidate(inputs: dict) -> dict:
    """The best logistic regression baseline, in the same form as an AutoML result.

    It is refit on the training window and scored on the validation canvass rows, so its scores
    and metrics match results/baselines.csv and can serve as the monitoring reference.
    """
    baselines = pd.read_csv(config.RESULTS_DIR / "baselines.csv")
    models = baselines[baselines["baseline"].str.startswith("logistic_regression_")]
    name = models.sort_values(HEADLINE, ascending=False).iloc[0]["baseline"]
    version = name.rsplit("_", 1)[1]  # logistic_regression_v2 -> v2
    estimator = LogisticRegression(max_iter=1000)
    model = train.refit(estimator, inputs["train"], version)
    scores, metrics, _ = train.evaluate(model, inputs["valid_canvass"], version, inputs["k_week"])
    return {"version": version, "estimator_name": "logistic_regression", "estimator": estimator,
            "scores": scores, "metrics": metrics}


@task(cache_policy=NO_CACHE)
def register_model(winner: dict, inputs: dict, parent_id: str, promotion: str) -> dict:
    """Refit the chosen model on training plus validation rows, register it, and save reference data.

    `promotion` is "challenger" for an AutoML model that passed the gate, or "bootstrap_baseline"
    for the baseline registered as the first production version.
    """
    version, valid_c = winner["version"], inputs["valid_canvass"]
    rows = pd.concat([inputs["train"], inputs["valid"]], ignore_index=True)
    final_model = train.refit(winner["estimator"], rows, version)

    run_id = tracking.start_run(f"final {version} {winner['estimator_name']}", parent_id,
                                {**tracking.lineage_tags(), "feature_version": version,
                                 "estimator": winner["estimator_name"], "promotion": promotion})
    tracking.log_params(run_id, {"feature_version": version, "estimator": winner["estimator_name"],
                                 "refit_rows": len(rows), **winner["estimator"].get_params()})
    tracking.log_metrics(run_id, {f"valid_{k}": v for k, v in winner["metrics"].items()})

    # Reference data for monitoring: validation canvass rows scored by the train-only model.
    reference = valid_c[["inspection_id", "inspection_date", *config.FEATURE_COLUMNS[version], config.TARGET]].assign(
        prediction=winner["scores"])
    config.REFERENCE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(config.REFERENCE_PARQUET, index=False)
    tracking.log_artifact(run_id, config.REFERENCE_PARQUET)

    tags = {
        **tracking.lineage_tags(),
        "feature_version": version,
        "estimator": winner["estimator_name"],
        "signature": registry.signature_hash(version),
        "valid_headline": round(winner["metrics"][HEADLINE], 4),
        "valid_se": round(winner["metrics"]["se_headline"], 4),
        "k_week": inputs["k_week"],
        "promotion": promotion,
        "training_run_id": run_id,
    }
    X_example = train.feature_frame(rows.head(50), version)
    model_version, semver = registry.register(
        run_id, final_model, X_example, tags,
        f"{winner['estimator_name']} on feature version {version}, data {tags['data_tag']} ({promotion})")
    tracking.set_tags(run_id, {"registered_version": model_version, "semver": semver})
    tracking.end_run(run_id)
    return {"model_version": model_version, "semver": semver, "run_id": run_id, "promotion": promotion,
            "model": f"{winner['estimator_name']} {version}"}


@task(cache_policy=NO_CACHE)
def write_reports(results: list[dict], inputs: dict, gate_result: tuple, registered: dict | None,
                  parent_id: str) -> None:
    """Comparison table and precision-by-k figure in results/training/, also logged to MLflow."""
    winner, passed, reason = gate_result
    baseline, valid_c = inputs["baseline"], inputs["valid_canvass"]
    out_dir = config.TRAINING_RESULTS_DIR

    rows = [{"model": f"AutoML {r['version']} ({r['estimator_name']})", f"inner_{HEADLINE}": r["inner_headline"],
             HEADLINE: r["metrics"][HEADLINE], "se_headline": r["metrics"]["se_headline"],
             "lift_per_100": r["metrics"]["lift_per_100"], "roc_auc": r["metrics"]["roc_auc"]} for r in results]
    baselines = pd.read_csv(config.RESULTS_DIR / "baselines.csv")
    best = baselines.sort_values(HEADLINE, ascending=False).iloc[0]
    rows.append({"model": f"best baseline ({best['baseline']})", f"inner_{HEADLINE}": "",  # not searched
                 HEADLINE: best[HEADLINE], "se_headline": best["se_headline"],
                 "lift_per_100": best["lift_per_100"], "roc_auc": best["roc_auc"]})
    table = pd.DataFrame(rows).sort_values(HEADLINE, ascending=False)
    table.to_csv(out_dir / "comparison.csv", index=False)

    if registered is None:
        decision = "Nothing was registered."
    elif registered["promotion"] == "challenger":
        decision = (f"The challenger was registered as version {registered['model_version']} "
                    f"({registered['semver']}) and is now production.")
    else:
        decision = (f"No production model existed, so the best baseline ({registered['model']}) was registered as "
                    f"version {registered['model_version']} ({registered['semver']}) and is now production. "
                    f"Later challengers must beat it.")
    (out_dir / "comparison.md").write_text(
        "# Model comparison\n\n"
        f"Facility scope: {config.FACILITY_SCOPE}. AutoML selected each configuration on "
        f"{len(inputs['inner_holdout']):,} canvass inspections in the inner holdout (training rows from "
        f"{config.INNER_VALID_START.date()}), shown as inner_{HEADLINE}. The other columns score each model "
        f"once on {len(valid_c):,} canvass inspections in the validation window, with k = {inputs['k_week']} "
        f"visits per week and a base rate of {valid_c[config.TARGET].mean():.3f}. The feature version with "
        f"the higher inner score is the challenger, and the gate reads its validation score.\n\n"
        + markdown_table(table) + "\n\n"
        f"Gate: {reason}. {decision}\n")

    scores = {f"AutoML {r['version']}": r["scores"] for r in results}
    figure = train.plot_precision_by_k(scores, valid_c, inputs["k_week"], out_dir / "precision_by_k.png")
    for path in (out_dir / "comparison.csv", out_dir / "comparison.md", figure):
        tracking.log_artifact(parent_id, path)


@flow(name="inspection-ranking-training")
def training_flow(feature_versions: tuple[str, ...] = ("v1", "v2"), time_budget: int = config.FLAML_TIME_BUDGET,
                  skip_dvc: bool = False, register: bool = True) -> dict:
    """Update the data, search each feature version, gate the winner, and register it."""
    update_data(skip_dvc)
    inputs = load_inputs()

    parent_id = tracking.start_run(f"training {datetime.now():%Y-%m-%d %H:%M}", tags=tracking.lineage_tags())
    tracking.log_params(parent_id, {"feature_versions": list(feature_versions), "time_budget_s": time_budget,
                                    "k_week": inputs["k_week"], "best_baseline": inputs["baseline"].name,
                                    "best_baseline_headline": inputs["baseline"].headline})
    try:
        results = [search_version(inputs, v, time_budget, parent_id) for v in feature_versions]
        gate_result = choose_and_gate(results, inputs["baseline"])
        winner, passed, reason = gate_result
        tracking.set_tags(parent_id, {"winner": f"{winner['version']} {winner['estimator_name']}",
                                      "gate_passed": passed, "gate_reason": reason})
        action = registry.promotion_action(passed, registry.production_version() is not None)
        registered = None
        if register and action != "none":
            candidate = winner if action == "challenger" else baseline_candidate(inputs)
            registered = register_model(candidate, inputs, parent_id, action)
        tracking.set_tags(parent_id, {"promotion": action if register else "not requested"})
        write_reports(results, inputs, gate_result, registered, parent_id)
    except Exception:
        tracking.end_run(parent_id, status="FAILED")
        raise
    tracking.end_run(parent_id)
    return {"winner": winner["version"], "gate_passed": passed, "registered": registered}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-versions", nargs="+", choices=list(config.FEATURE_COLUMNS),
                        default=list(config.FEATURE_COLUMNS), help="feature versions to search (default: all)")
    parser.add_argument("--time-budget", type=int, default=config.FLAML_TIME_BUDGET,
                        help="FLAML search time per feature version, in seconds")
    parser.add_argument("--skip-dvc", action="store_true", help="do not run dvc repro first")
    parser.add_argument("--no-register", action="store_true", help="search and compare without registering")
    args = parser.parse_args(argv)
    result = training_flow(tuple(args.feature_versions), args.time_budget, args.skip_dvc, not args.no_register)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
