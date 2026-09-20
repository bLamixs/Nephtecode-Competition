"""
Тесты оркестратора: проверка сквозного цикла и всех 4 сценариев (normal, risk, missing, no_solution).
"""

import pytest
import asyncio
import pandas as pd
import numpy as np

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.recommendation import Recommendation


@pytest.fixture
def orchestrator():
    return Orchestrator()


def test_orchestrator_scenario_normal(orchestrator):
    """Штатный режим: рекомендация должна быть успешно сформирована."""
    rec = asyncio.run(orchestrator.run_cycle(scenario='normal'))
    assert isinstance(rec, Recommendation)
    assert rec.status == "RECOMMENDED"
    assert rec.confidence > 0.0
    assert len(rec.action) >= 1
    assert rec.expected_effect.sulfur_60min is not None
    assert rec.expected_effect.throughput is not None
    assert rec.problem_type in ["RISK_SPEC_VIOLATION", "SUBOPTIMAL"]


def test_orchestrator_scenario_risk(orchestrator):
    """Сценарий риска по сере: рекомендация с повышением температуры или стабилизацией."""
    rec = asyncio.run(orchestrator.run_cycle(scenario='risk'))
    assert isinstance(rec, Recommendation)
    assert rec.status == "RECOMMENDED"
    assert rec.expected_effect.sulfur_60min <= 10.0
    assert len(rec.constraints_checked) > 0


def test_orchestrator_scenario_missing(orchestrator):
    """Сбой датчиков / устаревшие данные: оркестратор должен дать отказ NO_RECOMMENDATION."""
    rec = asyncio.run(orchestrator.run_cycle(scenario='missing'))
    assert isinstance(rec, Recommendation)
    assert rec.status == "NO_RECOMMENDATION"
    assert any(w in rec.explanation.lower() for w in ["устарел", "недостаточн", "пропуск"])


def test_orchestrator_scenario_no_solution(orchestrator):
    """Технологический тупик: отказ NO_RECOMMENDATION из-за нарушения ограничений."""
    rec = asyncio.run(orchestrator.run_cycle(scenario='no_solution'))
    assert isinstance(rec, Recommendation)
    assert rec.status == "NO_RECOMMENDATION"


def test_orch_04_recommendation_dataclass():
    """ORCH-04: Проверка dataclass Recommendation, словарей и сериализации."""
    from src.orchestrator.recommendation import (
        Recommendation, ActionItem, ExpectedEffect, ConstraintCheck, Alternative
    )
    rec = Recommendation(
        recommendation_id="rec_20260920_120000",
        timestamp="2026-09-20T12:00:00",
        state={"T6": 360.0, "Sulfur_current": 9.2},
        problem_type="RISK_SPEC_VIOLATION",
        action=[
            ActionItem(tag="T6", name="Температура реактора", from_value=360.0, to_value=363.0, unit="°C", delta=3.0, delta_percent=0.83)
        ],
        expected_effect=ExpectedEffect(
            sulfur_60min=7.5,
            sulfur_delta=-1.7,
            throughput=215.0,
            throughput_delta=0.0,
            risk_index=0.20
        ),
        constraints_checked=[
            ConstraintCheck(constraint_id="C001", constraint="Сера <= 10", predicted_value=7.5, threshold=10.0, status="PASS", margin=2.5)
        ],
        confidence=0.85,
        status="RECOMMENDED",
        alternatives=[
            Alternative(id=1, action={"T6": 362.0}, score=0.88, throughput=215.0, energy_proxy=0.25, risk_index=0.18, delta_score=-0.02, delta_throughput=0.0),
            Alternative(id=2, action={"T6": 364.0}, score=0.85, throughput=215.0, energy_proxy=0.26, risk_index=0.22, delta_score=-0.05, delta_throughput=0.0)
        ],
        explanation="Повышение T6 на 3°C снизит серу с 9.2 до 7.5 мг/кг."
    )

    # Проверка полей DoD
    assert rec.recommendation_id.startswith("rec_")
    assert rec.status == "RECOMMENDED"
    assert rec.confidence == 0.85
    assert len(rec.action) >= 1
    assert rec.action["T6"].to_value == 363.0
    assert rec.action[0].tag == "T6"
    assert rec.expected_effect.sulfur_60min == 7.5
    assert rec.expected_effect["Sulfur_60min"] == 7.5
    assert len(rec.constraints_checked) == 1
    assert rec.constraints_checked[0]["status"] == "PASS"
    assert len(rec.alternatives) == 2
    assert "3°C" in rec.explanation

    # Проверка to_dict и to_json
    d = rec.to_dict()
    assert isinstance(d, dict)
    assert d["status"] == "RECOMMENDED"
    assert d["expected_effect"]["Sulfur_60min"] == 7.5
    assert len(d["alternatives"]) == 2

    json_str = rec.to_json()
    assert '"status": "RECOMMENDED"' in json_str


def test_orch_04_build_recommendation_interface(orchestrator):
    """ORCH-04: Проверка метода build_recommendation(quality, reliability, optimization)."""
    from src.agents.interfaces import QualityAssessment, ReliabilityAssessment, OptimizationResult

    quality = QualityAssessment(
        timestamp="2026-09-20T12:00:00",
        sulfur_forecast_mg_kg=7.5,
        confidence=0.88
    )
    reliability = ReliabilityAssessment(
        timestamp="2026-09-20T12:00:00",
        risk_index=0.22,
        risk_class="LOW"
    )
    optimization = OptimizationResult(
        timestamp="2026-09-20T12:00:00",
        recommended={
            "id": 101,
            "action": {"T6": 363.0, "F2_F26_ratio": 0.86},
            "predicted_quality": {"Sulfur_60min": 7.4},
            "throughput": 218.0,
            "energy_proxy": 0.44,
            "risk_index": 0.22,
            "score": 0.91
        },
        alternatives=[
            {"id": 102, "action": {"T6": 362.0}, "throughput": 215.0, "score": 0.89, "risk_index": 0.20},
            {"id": 103, "action": {"T6": 364.0}, "throughput": 220.0, "score": 0.87, "risk_index": 0.25}
        ]
    )

    rec = orchestrator.build_recommendation(quality, reliability, optimization)

    assert isinstance(rec, Recommendation)
    assert rec.status == "RECOMMENDED"
    assert rec.confidence == 0.88
    assert len(rec.action) >= 1
    assert rec.expected_effect.sulfur_60min is not None
    assert rec.expected_effect.throughput is not None
    assert len(rec.alternatives) == 2
    assert "T6" in rec.explanation
    assert len(rec.constraints_checked) >= 1


def test_orch_05_no_recommendation_and_logging(orchestrator):
    """ORCH-05: Проверка no_recommendation и логирования в logs/orchestrator.log."""
    from pathlib import Path

    reason = "Последнее ЛИМС устарело (360 мин)"
    rec = orchestrator.no_recommendation(reason)

    assert isinstance(rec, Recommendation)
    assert rec.status == "NO_RECOMMENDATION"
    assert rec.confidence == 0.0
    assert len(rec.action) == 0
    assert len(rec.alternatives) == 0
    assert "360 мин" in rec.explanation
    assert rec.problem_type == "NO_DATA"

    # Проверка записи в logs/orchestrator.log
    log_path = Path("logs/orchestrator.log")
    assert log_path.exists(), "Файл logs/orchestrator.log должен существовать"

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "NO_RECOMMENDATION" in content
    assert "360 мин" in content
