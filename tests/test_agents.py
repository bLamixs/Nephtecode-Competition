"""
Модуль тестирования интерфейсов агентов (Блок: Tests-02).

Проверяет:
- QualityAgent: возвращает QualityAssessment, прогноз серы, уверенность в [0, 1]
- ReliabilityAgent: возвращает ReliabilityAssessment, индекс риска в [0, 1], класс риска (low/medium/high)
- OptimizationAgent: возвращает OptimizationResult, генерирует >= 2 альтернатив
- Корректную заполненность всех полей Dataclass-ов (post_init, to_dict).
"""

from dataclasses import asdict
import pytest
import pandas as pd

from src.agents.quality_agent import QualityAgent, QualityAssessment
from src.agents.reliability_agent import ReliabilityAgent, ReliabilityAssessment
from src.agents.optimization_agent import OptimizationAgent, OptimizationResult


@pytest.fixture
def mock_telemetry():
    """Фикстура мок-телеметрии с 10-минутными срезами параметров T6 и F26."""
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=10, freq='10min'),
        'T6': [295, 296, 297, 298, 299, 300, 301, 302, 303, 304],
        'F26': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })


def test_quality_agent_interface(mock_telemetry):
    """
    Проверка контракта QualityAgent:
    - возвращает экземпляр QualityAssessment
    - содержит прогноз серы ('Sulfur') в predictions
    - уверенность находится в диапазоне [0, 1]
    """
    agent = QualityAgent()
    assessment = agent.assess(mock_telemetry)

    assert isinstance(assessment, QualityAssessment)
    assert 'Sulfur' in assessment.predictions
    assert 0 <= assessment.confidence <= 1
    assert assessment.timestamp is not None
    assert assessment.sulfur_forecast_mg_kg > 0
    assert 0 <= assessment.risk_overall_quality <= 1

    # Проверка заполненности dataclass
    d = asdict(assessment)
    assert 'predictions' in d
    assert 'confidence' in d
    assert 'sulfur_forecast_mg_kg' in d


def test_reliability_agent_interface(mock_telemetry):
    """
    Проверка контракта ReliabilityAgent:
    - возвращает экземпляр ReliabilityAssessment
    - индекс риска находится в диапазоне [0, 1]
    - класс риска входит в допустимый перечень ['low', 'medium', 'high']
    """
    agent = ReliabilityAgent()
    assessment = agent.assess(mock_telemetry)

    assert isinstance(assessment, ReliabilityAssessment)
    assert 0 <= assessment.risk_index <= 1
    assert assessment.risk_class in ['low', 'medium', 'high']
    assert isinstance(assessment.is_safe, bool)
    assert isinstance(assessment.constraints_for_optimizer, list)

    # Проверка заполненности dataclass
    d = asdict(assessment)
    assert 'risk_index' in d
    assert 'risk_class' in d
    assert 'constraints_for_optimizer' in d


def test_optimization_agent_interface(mock_telemetry):
    """
    Проверка контракта OptimizationAgent:
    - возвращает экземпляр OptimizationResult
    - содержит не менее 2 альтернативных вариантов управления
    """
    agent = OptimizationAgent()
    result = agent.optimize(mock_telemetry)

    assert isinstance(result, OptimizationResult)
    assert len(result.alternatives) >= 2
    assert result.recommended is not None
    assert len(result.feasible) > 0

    # Проверка распаковки и словаря результата
    rec, alts = result
    assert rec is not None
    assert len(alts) >= 2
    assert result.timestamp is not None
