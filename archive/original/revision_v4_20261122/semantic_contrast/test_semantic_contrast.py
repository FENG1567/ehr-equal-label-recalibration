import math

import numpy as np
import pytest
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)

from semantic_contrast import METRIC_NAMES, paired_contrasts, weighted_metrics


def test_weighted_metrics_match_sklearn_with_tied_scores_and_zero_weights():
    y = np.asarray([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8)
    p = np.asarray([0.8, 0.8, 0.4, 0.4, 0.2, 0.2, 0.6, 0.9], dtype=float)
    w = np.asarray([2.0, 3.0, 1.0, 4.0, 0.0, 2.0, 1.5, 0.0], dtype=float)

    observed = weighted_metrics(y, p, w)
    expected = {
        "auroc": roc_auc_score(y, p, sample_weight=w),
        "average_precision": average_precision_score(y, p, sample_weight=w),
        "brier": brier_score_loss(y, p, sample_weight=w),
        "logloss": log_loss(y, p, labels=[0, 1], sample_weight=w),
    }
    for metric in METRIC_NAMES:
        assert observed[metric] == pytest.approx(expected[metric], abs=1e-14, rel=0)


def test_weighted_metrics_returns_nan_for_degenerate_weighted_outcome():
    observed = weighted_metrics(np.asarray([1, 0]), np.asarray([0.8, 0.2]),
                                np.asarray([2.0, 0.0]))
    assert all(math.isnan(observed[name]) for name in METRIC_NAMES)


def test_weighted_metrics_rejects_negative_weights():
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_metrics(np.asarray([1, 0]), np.asarray([0.8, 0.2]),
                         np.asarray([1.0, -1.0]))


def test_three_paired_contrasts_use_requested_signs():
    arms = {
        "original_real": {"auroc": 0.6, "average_precision": 0.2, "brier": 0.1, "logloss": 0.3},
        "original_constant": {"auroc": 0.5, "average_precision": 0.1, "brier": 0.2, "logloss": 0.4},
        "repaired_real": {"auroc": 0.7, "average_precision": 0.3, "brier": 0.08, "logloss": 0.25},
        "repaired_constant": {"auroc": 0.55, "average_precision": 0.12, "brier": 0.18, "logloss": 0.36},
    }
    observed = paired_contrasts(arms)
    assert observed["repaired_real_minus_original_real"]["auroc"] == pytest.approx(0.1)
    assert observed["repaired_constant_minus_original_constant"]["brier"] == pytest.approx(-0.02)
    assert observed["difference_in_differences"]["logloss"] == pytest.approx(-0.01)
