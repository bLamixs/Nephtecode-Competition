"""
Модуль: tests/test_metrics.py
Назначение: Автоматические тесты для модуля расчета метрик качества (src/utils/metrics.py).
"""

import pytest
import numpy as np
from src.utils.metrics import calc_regression_metrics


class TestRegressionMetrics:
    """Тестирование функции calc_regression_metrics."""

    def test_perfect_prediction(self):
        """Проверка нулевой ошибки при идеальном прогнозе."""
        y_true = np.array([10.0, 15.5, 8.2, 9.1])
        y_pred = np.array([10.0, 15.5, 8.2, 9.1])

        metrics = calc_regression_metrics(y_true, y_pred)
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['bias'] == 0.0
        assert metrics['count'] == 4

    def test_positive_bias(self):
        """Проверка систематического завышения прогноза (+2.0)."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 22.0, 32.0])

        metrics = calc_regression_metrics(y_true, y_pred)
        assert metrics['mae'] == 2.0
        assert metrics['rmse'] == 2.0
        assert metrics['bias'] == 2.0
        assert metrics['count'] == 3

    def test_negative_bias(self):
        """Проверка систематического занижения прогноза (-1.5)."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([8.5, 18.5, 28.5])

        metrics = calc_regression_metrics(y_true, y_pred)
        assert metrics['mae'] == 1.5
        assert metrics['rmse'] == 1.5
        assert metrics['bias'] == -1.5
        assert metrics['count'] == 3

    def test_nan_handling(self):
        """Проверка фильтрации NaN из фактических или прогнозных значений."""
        y_true = np.array([10.0, np.nan, 30.0, 40.0, np.nan])
        y_pred = np.array([12.0, 20.0, np.nan, 42.0, np.nan])

        # Валидные пары только на индексах 0 и 3:
        # (10.0, 12.0) -> diff = 2.0
        # (40.0, 42.0) -> diff = 2.0
        metrics = calc_regression_metrics(y_true, y_pred)
        assert metrics['count'] == 2
        assert metrics['mae'] == 2.0
        assert metrics['rmse'] == 2.0
        assert metrics['bias'] == 2.0

    def test_all_nan_or_empty(self):
        """Проверка поведения при полностью отсутствующих данных."""
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([np.nan, np.nan])

        metrics = calc_regression_metrics(y_true, y_pred)
        assert metrics['count'] == 0
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['bias'] == 0.0

        empty_metrics = calc_regression_metrics(np.array([]), np.array([]))
        assert empty_metrics['count'] == 0
