"""
Тесты для AGENT-06 (валидация моделей по времени и отчёт).
"""

import json
from pathlib import Path
import pytest
import numpy as np

from scripts.validate_models import compute_metrics


class TestAgent06Validation:
    """Проверка логики валидации моделей."""

    def test_compute_metrics_accuracy(self):
        y_true = np.array([10.0, 12.0, 14.0, 16.0])
        y_pred = np.array([11.0, 11.0, 15.0, 18.0])

        metrics = compute_metrics(y_true, y_pred)
        # Errors: +1, -1, +1, +2
        # Absolute errors: 1, 1, 1, 2 -> mean = 1.25
        # Squared errors: 1, 1, 1, 4 -> mean = 1.75 -> sqrt = 1.3229
        # Bias: (1 - 1 + 1 + 2)/4 = 0.75
        assert metrics['MAE'] == 1.25
        assert metrics['bias'] == 0.75
        assert pytest.approx(metrics['RMSE'], abs=0.01) == 1.3229
        assert metrics['count'] == 4

    def test_compute_metrics_handles_nans(self):
        y_true = np.array([10.0, np.nan, 14.0])
        y_pred = np.array([11.0, 12.0, np.nan])

        metrics = compute_metrics(y_true, y_pred)
        assert metrics['count'] == 1
        assert metrics['MAE'] == 1.0

    def test_validation_report_exists_and_valid(self):
        report_path = Path('output/validation_report.json')
        assert report_path.exists(), "Файл output/validation_report.json должен быть сгенерирован"

        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        assert 'metadata' in report
        assert 'indicators' in report
        assert 'walk_forward_folds' in report

        assert 'Sulfur' in report['indicators']
        sulfur_res = report['indicators']['Sulfur']
        assert 'model' in sulfur_res
        assert 'vac_baseline' in sulfur_res
        assert sulfur_res['mae_improvement_percent'] is not None
        assert sulfur_res['mae_improvement_percent'] > 50.0  # Улучшение к ВАК должно быть значительным

        folds = report['walk_forward_folds']
        assert len(folds) == 5
        for fold in folds:
            assert 'fold' in fold
            assert 'train_period' in fold
            assert 'test_period' in fold
            assert 'metrics' in fold
