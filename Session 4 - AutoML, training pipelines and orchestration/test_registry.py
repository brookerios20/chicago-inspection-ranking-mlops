"""Tests for the promotion gate and the semantic version rules."""

from src.registry import Bar, bump_semver, gate, promotion_action, signature_hash

CURRENT = {"signature": "abc", "data_processed_md5": "d1", "estimator": "lgbm"}


def test_first_version_is_1_0_0():
    assert bump_semver(None, CURRENT) == "1.0.0"


def test_signature_change_bumps_major():
    previous = {**CURRENT, "semver": "1.2.3", "signature": "old"}
    assert bump_semver(previous, CURRENT) == "2.0.0"


def test_new_data_or_algorithm_bumps_minor():
    assert bump_semver({**CURRENT, "semver": "1.2.3", "data_processed_md5": "d0"}, CURRENT) == "1.3.0"
    assert bump_semver({**CURRENT, "semver": "1.2.3", "estimator": "xgboost"}, CURRENT) == "1.3.0"


def test_same_inputs_bump_patch():
    assert bump_semver({**CURRENT, "semver": "1.2.3"}, CURRENT) == "1.2.4"


def test_signature_hash_differs_between_feature_versions():
    assert signature_hash("v1") != signature_hash("v2")
    assert signature_hash("v2") == signature_hash("v2")


def test_gate_requires_beating_every_bar_by_one_se():
    baseline = Bar("baseline", 0.449, 0.023)  # threshold 0.472
    assert gate(0.480, [baseline, None])[0]
    passed, reason = gate(0.470, [baseline, None])
    assert not passed and "baseline" in reason


def test_trusted_types_lets_skops_reload_the_model(split_raw):
    import numpy as np
    import skops.io
    from sklearn.linear_model import LogisticRegression

    from src.clean import clean
    from src.features import build_features
    from src.registry import trusted_types
    from src.train import feature_frame, refit

    features = build_features(clean(split_raw))
    features.loc[features.index[:3], "fail"] = np.array([0, 1, 0], dtype="int8")
    model = refit(LogisticRegression(max_iter=200), features, "v2")
    reloaded = skops.io.loads(skops.io.dumps(model), trusted=trusted_types(model))
    X = feature_frame(features, "v2")
    assert np.allclose(reloaded.predict_proba(X), model.predict_proba(X))


def test_promotion_action():
    assert promotion_action(gate_passed=True, has_production=False) == "challenger"
    assert promotion_action(gate_passed=True, has_production=True) == "challenger"
    assert promotion_action(gate_passed=False, has_production=False) == "bootstrap_baseline"
    assert promotion_action(gate_passed=False, has_production=True) == "none"


def test_gate_uses_the_highest_bar():
    baseline = Bar("baseline", 0.449, 0.023)
    production = Bar("production", 0.480, 0.020)  # threshold 0.500
    assert not gate(0.490, [baseline, production])[0]
    assert gate(0.510, [baseline, production])[0]
