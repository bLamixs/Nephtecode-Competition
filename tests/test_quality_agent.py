"""
Модуль: tests/test_quality_agent.py
Назначение: Автоматические тесты Quality Agent (AGENT-02).
Проверка подготовки признаков, расчёта остатков, обучения модели и контракта с Оркестратором.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.agents.quality_agent import QualityAgent
from src.agents.interfaces import QualityAssessment


@pytest.fixture
def sample_telemetry_batch():
    """Тестовая телеметрия КИП на 10 временных шагов."""
    dates = pd.date_range('2026-01-01', periods=10, freq='10min')
    df = pd.DataFrame({
        'date': dates,
        'T6_hydro': np.linspace(290.0, 300.0, 10),
        'F26_hydro': np.linspace(190.0, 210.0, 10),
        'P8': np.linspace(30.0, 34.0, 10),
        'T11_hydro': np.linspace(340.0, 350.0, 10),
        'F9_hydro': np.linspace(240.0, 260.0, 10),
        'P13': [3.2] * 10,
        'W7': [0.2] * 10,
        'F22': [5.0] * 10,
        'P24': [28.0] * 10,
        'F2': [170.0] * 10,
        'T16': [170.0] * 10,
        'T23': [355.0] * 10,
        'W4': [2.5] * 10,
        'F14_hydro': [3.0] * 10
    })
    return df


@pytest.fixture
def sample_quality_batch(sample_telemetry_batch):
    """Тестовые данные качества LIMS/PAK."""
    return pd.DataFrame({
        'date': sample_telemetry_batch['date'],
        'value_quality': [8.5, 8.7, 8.2, 8.9, 9.1, 8.4, 8.8, 8.6, 9.0, 8.5],
        'source': ['LIMS', 'PAK', 'PAK', 'LIMS', 'PAK', 'PAK', 'LIMS', 'PAK', 'PAK', 'LIMS'],
        'age_min': [15.0, 5.0, 15.0, 25.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    })


@pytest.fixture
def sample_vac_batch(sample_telemetry_batch):
    """Тестовые прогнозы ВАК."""
    return pd.DataFrame({
        'date': sample_telemetry_batch['date'],
        'vac_value': [52.0] * 10
    })


class TestQualityAgentFeatures:
    """Тестирование подготовки признаков и остатков."""

    def test_prepare_features_structure(self, sample_telemetry_batch, sample_quality_batch):
        agent = QualityAgent(model_path='output/models/non_existent.pkl')
        X = agent._prepare_features(sample_telemetry_batch, sample_quality_batch)

        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(sample_telemetry_batch)

        # Проверка обязательных признаков по ТЗ
        expected_cols = [
            'T6', 'T6_lag10', 'T6_lag30', 'T6_lag60', 'T6_zscore',
            'F26', 'F26_lag10', 'F26_lag30', 'F26_lag60', 'F26_zscore',
            'P8', 'P8_lag10', 'P8_lag30', 'P8_lag60', 'P8_zscore',
            'age_min', 'is_lims'
        ]
        for col in expected_cols:
            assert col in X.columns, f"Обязательный признак {col} отсутствует в матрице признаков"

        # Проверка отсутствия NaN
        assert not X.isna().any().any(), "В матрице признаков не должно быть NaN"

    def test_calculate_residuals(self, sample_quality_batch, sample_vac_batch):
        agent = QualityAgent(model_path='output/models/non_existent.pkl')
        residuals = agent._calculate_residuals(sample_quality_batch, sample_vac_batch)

        assert isinstance(residuals, pd.Series)
        assert len(residuals) == len(sample_quality_batch)
        expected_first = sample_quality_batch['value_quality'].iloc[0] - sample_vac_batch['vac_value'].iloc[0]
        assert np.isclose(residuals.iloc[0], expected_first)


class TestQualityAgentTrainingAndPredict:
    """Тестирование обучения модели и генерации прогнозов."""

    def test_train_and_predict(self, sample_telemetry_batch, sample_quality_batch, sample_vac_batch, tmp_path):
        tmp_model = str(tmp_path / "test_quality_model.pkl")
        agent = QualityAgent(model_path=tmp_model)

        # Обучение
        model = agent.train(
            telemetry=sample_telemetry_batch,
            quality=sample_quality_batch,
            vac=sample_vac_batch,
            model_type='auto'
        )
        assert model is not None
        assert Path(tmp_model).exists(), "Файл модели должен быть сохранен на диск"

        # Предсказание
        preds = agent.predict(sample_telemetry_batch, sample_quality_batch, vac=sample_vac_batch['vac_value'])
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(sample_telemetry_batch)
        assert (preds >= 0.0).all(), "Прогноз серы не должен быть отрицательным"

        # Проверка загрузки обученной модели в новый экземпляр
        new_agent = QualityAgent(model_path=tmp_model)
        assert new_agent.model is not None
        new_preds = new_agent.predict(sample_telemetry_batch, sample_quality_batch, vac=sample_vac_batch['vac_value'])
        np.testing.assert_array_almost_equal(preds.values, new_preds.values)


class TestQualityAgentAssessmentContract:
    """Тестирование межагентного контракта с Оркестратором."""

    def test_assess_contract_structure(self, sample_telemetry_batch, sample_quality_batch):
        import asyncio
        model_path = 'output/models/quality_lgbm.pkl'
        agent = QualityAgent(model_path=model_path)

        assessment = asyncio.run(agent.assess(sample_telemetry_batch, sample_quality_batch))

        assert isinstance(assessment, QualityAssessment)
        assert hasattr(assessment, 'sulfur_forecast_mg_kg')
        assert hasattr(assessment, 'risk_sulfur_violation')
        assert hasattr(assessment, 'confidence')
        assert hasattr(assessment, 'data_source')
        assert hasattr(assessment, 'warnings')

        # Проверка физических диапазонов
        assert assessment.sulfur_forecast_mg_kg >= 0.0
        assert 0.0 <= assessment.risk_sulfur_violation <= 1.0
        assert 0.0 <= assessment.confidence <= 1.0
        assert assessment.d15_forecast_kg_m3 is not None
        assert assessment.t95_forecast_c is not None
