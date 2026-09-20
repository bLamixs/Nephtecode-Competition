"""
Тесты для AGENT-03 (мультигоризонтный прогноз качества) и AGENT-04 (оценка рисков и уверенности).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.agents.quality_agent import QualityAgent
from src.agents.interfaces import QualityAssessment


@pytest.fixture
def sample_telemetry():
    """Синтетическая телеметрия для тестирования."""
    dates = pd.date_range(start='2026-05-01 00:00:00', periods=20, freq='10min')
    df = pd.DataFrame({
        'date': dates,
        'T6_hydro': np.linspace(294.0, 298.0, 20),
        'F26_hydro': np.linspace(245.0, 255.0, 20),
        'P8': np.linspace(30.0, 32.0, 20),
        'T11_hydro': np.linspace(280.0, 285.0, 20),
        'F9_avt': np.linspace(240.0, 260.0, 20),
        'P13': np.linspace(15.0, 16.0, 20),
        'W7': np.linspace(0.18, 0.22, 20),
        'F30': np.linspace(80.0, 90.0, 20),
        'F32': np.linspace(60.0, 65.0, 20),
        'F34': np.linspace(40.0, 45.0, 20),
        'F7': np.linspace(110.0, 120.0, 20),
        'F8': np.linspace(110.0, 120.0, 20),
        'F56': np.linspace(20.0, 25.0, 20),
        'F57': np.linspace(20.0, 25.0, 20),
        'F59': np.linspace(15.0, 20.0, 20),
        'T55': np.linspace(318.0, 322.0, 20)
    })
    return df


class TestAgent03MultiHorizonForecast:
    """Тестирование мультигоризонтного прогноза качества (AGENT-03)."""

    def test_predict_forecast_structure(self, sample_telemetry):
        agent = QualityAgent()
        horizons = [30, 60, 120]
        forecast = agent.predict_forecast(sample_telemetry, horizons_min=horizons)

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == len(sample_telemetry) * len(horizons)
        expected_cols = ['date', 'horizon_min', 'Sulfur', 'D15', 'T50', 'T90', 'T95', 'CFPP', 'flash']
        for col in expected_cols:
            assert col in forecast.columns

        # Проверка горизонтов
        assert set(forecast['horizon_min'].unique()) == {30, 60, 120}

    def test_predict_with_timestamps(self, sample_telemetry):
        agent = QualityAgent()
        timestamps = [sample_telemetry['date'].iloc[0], sample_telemetry['date'].iloc[5]]
        forecast = agent.predict(sample_telemetry, quality=timestamps, horizons_min=[30, 60])

        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast['date'].unique()) == 2
        assert set(forecast['horizon_min'].unique()) == {30, 60}

    def test_backward_compatible_predict_series(self, sample_telemetry):
        agent = QualityAgent()
        # Вызов без параметров горизонтов должен возвращать pd.Series (прогноз серы)
        res = agent.predict(sample_telemetry)
        assert isinstance(res, pd.Series)
        assert len(res) == len(sample_telemetry)
        assert res.name == 'predicted_sulfur'


class TestAgent04RiskAndConfidence:
    """Тестирование оценки рисков нарушения спецификаций и доверия (AGENT-04)."""

    def test_assess_risk_metrics(self, sample_telemetry):
        agent = QualityAgent()
        forecast = agent.predict_forecast(sample_telemetry, horizons_min=[30])

        # Принудительно задаем превышение для проверки скользящей оценки риска
        forecast.loc[0:2, 'Sulfur'] = 12.0
        forecast.loc[3:, 'Sulfur'] = 8.0

        assessed = agent.assess_risk(forecast, specs={'Sulfur': 10.0}, window=3)

        assert 'P_S_gt_10' in assessed.columns
        assert 'P_T95_gt_spec' in assessed.columns
        assert 'risk_overall' in assessed.columns

        # В точке 2 все 3 точки в окне равны 12.0 > 10.0 -> P = 1.0
        assert assessed['P_S_gt_10'].iloc[2] == 1.0
        # В точке 5 все точки в окне равны 8.0 <= 10.0 -> P = 0.0
        assert assessed['P_S_gt_10'].iloc[5] == 0.0

    def test_confidence_decay_with_age(self, sample_telemetry):
        import asyncio
        agent = QualityAgent()

        # 1. Свежие анализы (age = 10 мин)
        fresh_qual = pd.DataFrame([{'age_min': 10.0, 'source': 'LIMS'}])
        assessment_fresh = asyncio.run(agent.assess(sample_telemetry, fresh_qual))
        assert assessment_fresh.confidence > 0.9
        assert assessment_fresh.age_min == 10

        # 2. Устаревшие анализы (age = 200 мин)
        old_qual = pd.DataFrame([{'age_min': 200.0, 'source': 'LIMS'}])
        assessment_old = asyncio.run(agent.assess(sample_telemetry, old_qual))
        expected_conf = max(0.3, 1.0 - 200.0 / 240.0)
        assert pytest.approx(assessment_old.confidence, abs=0.01) == expected_conf

        # 3. Полностью устаревшие (> 240 мин) -> confidence = 0.3
        stale_qual = pd.DataFrame([{'age_min': 350.0, 'source': 'LIMS'}])
        assessment_stale = asyncio.run(agent.assess(sample_telemetry, stale_qual))
        assert assessment_stale.confidence == 0.3

    def test_quality_assessment_contract_fields(self, sample_telemetry):
        import asyncio
        agent = QualityAgent()
        assessment = asyncio.run(agent.assess(sample_telemetry))

        assert isinstance(assessment, QualityAssessment)
        assert isinstance(assessment.predictions, dict)
        assert 'Sulfur' in assessment.predictions
        assert 'D15' in assessment.predictions
        assert 'T95' in assessment.predictions

        assert isinstance(assessment.risk_spec_violation, dict)
        assert 'P_S_gt_10' in assessment.risk_spec_violation
        assert 'P_T95_gt_spec' in assessment.risk_spec_violation

        assert 0.0 <= assessment.confidence <= 1.0
        assert assessment.data_source is not None
        assert isinstance(assessment.age_min, int)

        # Проверка dict-like get метода
        assert assessment.get('confidence') == assessment.confidence
        assert assessment.get('unknown_key', 'def') == 'def'
