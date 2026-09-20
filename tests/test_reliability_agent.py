"""
Тесты для ReliabilityAgent (AGENT-05).
"""

import pytest
import numpy as np
import pandas as pd
from src.agents.reliability_agent import ReliabilityAgent
from src.agents.interfaces import ReliabilityAssessment


@pytest.fixture
def sample_telemetry():
    """Синтетическая телеметрия для тестирования надёжности."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        'date': pd.date_range('2026-06-01', periods=n, freq='10min'),
        'T6_hydro': np.random.normal(295.0, 2.0, n),
        'F26_hydro': np.random.normal(250.0, 5.0, n),
        'P8': np.random.normal(32.0, 1.0, n),
        'T55': np.random.normal(320.0, 2.0, n),
        'F7': np.random.normal(120.0, 5.0, n),
        'F8': np.random.normal(120.0, 5.0, n),
        'F9_avt': np.random.normal(250.0, 10.0, n),
        'T1': np.random.normal(127.0, 3.0, n),
        'W7': np.random.normal(0.20, 0.02, n)
    })
    return df


class TestReliabilityAgent:
    """Проверка функциональности ReliabilityAgent."""

    def test_zscore_calculation(self, sample_telemetry):
        agent = ReliabilityAgent(
            norm_mean={'T6': 295.0, 'P8': 32.0},
            norm_std={'T6': 5.0, 'P8': 2.0}
        )
        z_df = agent._calculate_zscore(sample_telemetry)
        assert isinstance(z_df, pd.DataFrame)
        assert 'T6_hydro' in z_df.columns or 'T6' in z_df.columns
        assert not z_df.isna().any().any()

    def test_mahalanobis_calculation(self, sample_telemetry):
        agent = ReliabilityAgent()
        m_dist = agent._calculate_mahalanobis(sample_telemetry)
        assert isinstance(m_dist, pd.Series)
        assert len(m_dist) == len(sample_telemetry)
        assert (m_dist >= 0.0).all()

    def test_risk_index_bounds(self, sample_telemetry):
        agent = ReliabilityAgent()
        z_df = agent._calculate_zscore(sample_telemetry)
        m_dist = agent._calculate_mahalanobis(sample_telemetry)
        risk = agent._calculate_risk_index(z_df, m_dist)

        assert isinstance(risk, pd.Series)
        assert (risk >= 0.0).all()
        assert (risk <= 1.0).all()

    def test_risk_factors_top3(self, sample_telemetry):
        agent = ReliabilityAgent()
        z_df = agent._calculate_zscore(sample_telemetry)
        factors = agent._get_risk_factors(z_df)

        assert isinstance(factors, list)
        assert len(factors) <= 3
        for f in factors:
            assert 'tag' in f
            assert 'deviation' in f
            assert 'weight' in f
            assert f['deviation'] >= 0.0

    def test_assess_contract_structure(self, sample_telemetry):
        import asyncio
        agent = ReliabilityAgent()
        assessment = asyncio.run(agent.assess(sample_telemetry))

        assert isinstance(assessment, ReliabilityAssessment)
        assert 0.0 <= assessment.risk_index <= 1.0
        assert assessment.risk_class in ['low', 'medium', 'high']
        assert isinstance(assessment.is_safe, bool)
        assert isinstance(assessment.risk_factors, list)
        assert isinstance(assessment.constraints_for_optimizer, list)
        assert len(assessment.constraints_for_optimizer) > 0

        for c in assessment.constraints_for_optimizer:
            assert 'tag' in c
            assert 'min' in c
            assert 'max' in c
            assert c['min'] <= c['max']

        # Проверка get метода
        assert assessment.get('risk_class') == assessment.risk_class
        assert assessment.get('not_found', 99) == 99

    def test_extreme_regime_triggers_high_risk(self, sample_telemetry):
        agent = ReliabilityAgent()
        # Искусственно создаем критическую перегрузку (температура +10 сигм)
        extreme_df = sample_telemetry.copy()
        extreme_df['T6_hydro'] = 450.0  # Экстремальный перегрев
        extreme_df['P8'] = 60.0         # Экстремальное давление

        assessment = agent.assess_telemetry(extreme_df)
        assert assessment.risk_index > 0.50
        assert assessment.risk_class in ['medium', 'high']
