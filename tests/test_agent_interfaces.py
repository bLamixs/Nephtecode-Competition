"""
Модуль: tests/test_agent_interfaces.py
Назначение: Автоматические тесты контрактов межагентного взаимодействия (src/agents/interfaces.py).
"""

import pytest
from datetime import datetime
from src.agents.interfaces import (
    QualityAssessment,
    ReliabilityAssessment,
    CandidateAction,
    OptimizationResult as InterfaceOptimizationResult,
    Recommendation
)


class TestAgentInterfaces:
    """Тестирование строго типизированных контрактов взаимодействия агентов."""

    def test_quality_assessment_instantiation(self):
        qa = QualityAssessment(
            timestamp=datetime.now(),
            sulfur_forecast_mg_kg=8.4,
            d15_forecast_kg_m3=832.0,
            t95_forecast_c=354.0,
            risk_sulfur_violation=0.05,
            confidence=0.98,
            data_source="HYBRID_VAC_ML"
        )
        assert qa.sulfur_forecast_mg_kg == 8.4
        assert qa.risk_sulfur_violation < 0.10
        assert qa.confidence == 0.98
        assert isinstance(qa.warnings, list)

    def test_reliability_assessment_instantiation(self):
        ra = ReliabilityAssessment(
            timestamp=datetime.now(),
            risk_index=0.15,
            risk_class="LOW",
            is_safe=True,
            risk_factors=["Температурный режим в норме"],
            active_constraints=["T6 <= 305.0"]
        )
        assert ra.risk_class == "LOW"
        assert ra.is_safe is True
        assert len(ra.active_constraints) == 1

    def test_candidate_action_instantiation(self):
        action = CandidateAction(
            action_id="ACT-001",
            changes={'T6': {'current': 295.0, 'target': 297.0, 'delta': 2.0}},
            expected_sulfur_effect=-0.4,
            expected_throughput_change=0.0,
            is_hard_valid=True
        )
        assert action.action_id == "ACT-001"
        assert action.expected_sulfur_effect == -0.4
        assert action.is_hard_valid is True

    def test_optimization_result_contract(self):
        action = CandidateAction(
            action_id="ACT-001",
            changes={'T6': {'current': 295.0, 'target': 297.0, 'delta': 2.0}},
            expected_sulfur_effect=-0.4
        )
        opt_res = InterfaceOptimizationResult(
            timestamp=datetime.now(),
            is_solution_found=True,
            top_recommendation=action,
            evaluated_candidates_count=50,
            valid_candidates_count=35
        )
        assert opt_res.is_solution_found is True
        assert opt_res.top_recommendation.action_id == "ACT-001"
        assert opt_res.evaluated_candidates_count == 50

    def test_recommendation_contract(self):
        rec = Recommendation(
            timestamp=datetime.now(),
            is_refusal=False,
            status="ACTION_RECOMMENDED",
            problem_detected="Прогноз серы приближается к 9.5 мг/кг",
            proposed_action={'tag': 'T6', 'current': 295.0, 'target': 297.0},
            confidence_score=0.95,
            explanation="Увеличение температуры реактора на +2°C снизит содержание серы на 0.4 мг/кг."
        )
        assert rec.is_refusal is False
        assert rec.status == "ACTION_RECOMMENDED"
        assert rec.confidence_score == 0.95

    def test_agent_request_instantiation(self):
        """Проверка контракта AgentRequest."""
        import pandas as pd
        from src.agents.interfaces import AgentRequest

        req = AgentRequest(
            timestamp=datetime.now(),
            scenario="risk",
            telemetry=pd.DataFrame([{"T6": 360.0, "F9": 215.0}]),
            quality=pd.DataFrame([{"tag": "Sulfur", "value": 9.2}]),
            features=pd.DataFrame([{"f1": 1.0}]),
            constraints=[{"constraint": "Sulfur <= 10.0", "limit": 10.0}],
            metadata={"cycle_id": "c1"}
        )

        assert req.scenario == "risk"
        assert not req.telemetry.empty
        assert not req.quality.empty
        assert req.features is not None
        assert len(req.constraints) == 1
        assert req.metadata["cycle_id"] == "c1"

    @pytest.mark.anyio
    async def test_orchestrator_call_agents_with_agent_request(self):
        """Проверка вызова агентов через _call_*_agent и call_*_agent с AgentRequest."""
        import pandas as pd
        from src.agents.interfaces import AgentRequest, QualityAssessment, ReliabilityAssessment
        from src.orchestrator.orchestrator import Orchestrator

        orch = Orchestrator()
        req = AgentRequest(
            timestamp=datetime.now(),
            scenario="normal",
            telemetry=pd.DataFrame([{"T6": 360.0, "F2_F26_ratio": 0.85, "T55": 380.0, "F9": 215.0}]),
            quality=pd.DataFrame([{"tag": "Sulfur", "value": 8.0, "age_min": 30.0}])
        )

        qa = await orch.call_quality_agent(req)
        assert isinstance(qa, QualityAssessment)
        assert qa.confidence > 0.0

        ra = await orch.call_reliability_agent(req)
        assert isinstance(ra, ReliabilityAssessment)
        assert ra.risk_class.upper() in ["LOW", "MEDIUM", "HIGH"]

        opt = await orch.call_optimization_agent(req, qa, ra)
        assert opt is not None
        assert hasattr(opt, "recommended")
