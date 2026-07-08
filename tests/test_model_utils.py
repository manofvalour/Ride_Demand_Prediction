"""Tests for model_utils.py."""

import numpy as np
from src.DynamicPricingEngine.utils.model_utils import compute_metrics


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics['mae'] == 0.0
        assert metrics['mse'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['R2_score'] == 1.0

    def test_off_by_constant(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics['mae'] == 1.0
        assert metrics['mse'] == 1.0
        assert metrics['rmse'] == 1.0

    def test_all_same_true(self):
        y_true = np.array([5, 5, 5, 5])
        y_pred = np.array([5, 5, 5, 5])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics['R2_score'] == 1.0

    def test_all_zeros_true(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0

    def test_single_element(self):
        y_true = np.array([42])
        y_pred = np.array([42])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics['mae'] == 0.0
        assert metrics['R2_score'] == 1.0

    def test_metric_keys(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        metrics = compute_metrics(y_true, y_pred)
        assert set(metrics.keys()) == {'mae', 'mse', 'rmse', 'R2_score'}
