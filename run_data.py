"""Run the data ingestion and baseline stage end to end.

Stages, in order:

1. ingest: load the snapshot and write data/raw/inspections.parquet with its manifest
2. validate: check the raw snapshot; a failed validation rule stops the run
3. clean: keep modeling rows, add the target and cleaned categories
4. features: add history and ZIP code features, then check them against the input schema
5. split: write the fixed-date splits and their manifest to data/processed/
6. baselines: score the baselines on the validation window and write them to results/

Usage:
    python run_data.py                  # local CSV if present, otherwise the portal API
    python run_data.py --source api     # always download from the portal API
    python run_data.py --skip-baselines # stop after writing the splits
"""

import argparse
import logging

from src import baselines, clean, config, ingest, split, validate
from src import features as feature_stage

logger = logging.getLogger("run_data")


def save_report(report, name: str) -> None:
    """Write a validation report to results/validation/ and log any warnings."""
    out = config.RESULTS_DIR / "validation" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    for row in validate.warnings_of(report).itertuples():
        logger.warning("%s check %s: %s", name, row.rule, row.detail)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["auto", "local", "api"], default="auto",
                        help="where to read the snapshot from (default: auto)")
    parser.add_argument("--skip-baselines", action="store_true", help="stop after writing the splits")
    args = parser.parse_args(argv)

    raw, raw_manifest = ingest.ingest(args.source)

    raw_report = validate.validate_raw(raw)
    save_report(raw_report, "raw")
    validate.raise_on_errors(raw_report, "raw")

    cleaned = clean.clean(raw)
    logger.info("Modeling rows: %s of %s", f"{len(cleaned):,}", f"{len(raw):,}")

    features = feature_stage.build_features(cleaned)
    feature_report = validate.validate_features(features, version="v2", require_target=True)
    save_report(feature_report, "features")
    validate.raise_on_errors(feature_report, "features")

    split.write_splits(features, raw_manifest=raw_manifest)

    if not args.skip_baselines:
        table, _ = baselines.run_baselines(split.load_split("train"), split.load_split("valid"))
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
