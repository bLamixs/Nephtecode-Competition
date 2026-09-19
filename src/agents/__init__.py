"""
Модуль: src/agents
Экспорт основных агентов мультиагентной системы.
"""

from src.agents.quality_agent import QualityAgent
from src.agents.reliability_agent import ReliabilityAgent
from src.agents.optimization_agent import OptimizationAgent
from src.agents.interfaces import (
    QualityAssessment,
    ReliabilityAssessment,
    CandidateAction,
    OptimizationResult,
    Recommendation
)

__all__ = [
    'QualityAgent',
    'ReliabilityAgent',
    'OptimizationAgent',
    'QualityAssessment',
    'ReliabilityAssessment',
    'CandidateAction',
    'OptimizationResult',
    'Recommendation'
]
