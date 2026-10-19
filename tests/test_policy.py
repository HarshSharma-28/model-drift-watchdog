"""Sanity checks that the locked policy is internally consistent."""

import policy


def test_psi_thresholds_ordered():
    assert 0 < policy.PSI_MODERATE < policy.PSI_MAJOR


def test_critical_features_exist():
    all_features = set(policy.NUMERIC_FEATURES) | set(policy.CATEGORICAL_FEATURES)
    assert set(policy.CRITICAL_FEATURES) <= all_features


def test_retry_limit():
    assert policy.MAX_RETRIES == 2
