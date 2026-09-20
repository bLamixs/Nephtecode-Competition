"""
Модуль: tests/test_optimization_agent.py
Назначение: Автоматические тесты для Агента Оптимизации (src/agents/optimization_agent.py).
Поддерживает ветки с датаклассами Candidate и ScoredCandidate (OPT-03, OPT-04).
"""

import pytest
import numpy as np
import pandas as pd
from src.agents.optimization_agent import (
    OptimizationAgent,
    Candidate,
    ScoredCandidate,
    OptimizationResult,
    CONTROLLED_PARAMS,
    BLENDING_FRACTIONS
)


@pytest.fixture
def opt_agent():
    """Фикстура создания экземпляра OptimizationAgent."""
    return OptimizationAgent()


class TestOptimizationAgentGeneration:
    """Тестирование алгоритмов генерации кандидатов."""

    def test_grid_search_generation(self, opt_agent):
        candidates = opt_agent._grid_search(['T6', 'F2_F26_ratio'])
        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c, Candidate)
            assert CONTROLLED_PARAMS['T6'].min <= c.params['T6'] <= CONTROLLED_PARAMS['T6'].max
            assert CONTROLLED_PARAMS['F2_F26_ratio'].min <= c.params['F2_F26_ratio'] <= CONTROLLED_PARAMS['F2_F26_ratio'].max

    def test_sobol_sampling(self, opt_agent):
        candidates = opt_agent._sobol_sampling(num_candidates=16)
        assert len(candidates) == 16
        for c in candidates:
            assert isinstance(c, Candidate)
            for param_name, param_cfg in CONTROLLED_PARAMS.items():
                assert param_name in c.params
                val = c.params[param_name]
                assert param_cfg.min - 1e-4 <= val <= param_cfg.max + 1e-4

    def test_generate_candidates_combined(self, opt_agent):
        candidates = opt_agent.generate_candidates(
            num_mode_candidates=10,
            num_blending_candidates=5,
            use_grid=False
        )
        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c, Candidate)
            assert 'T6' in c.params


class TestOptimizationAgentVetoAndScoring:
    """Тестирование механизма Veto (OPT-03) и целевой функции скоринга (OPT-04)."""

    def test_check_param_ranges_veto(self, opt_agent):
        valid = Candidate(
            params={'T6': 360.0, 'F2_F26_ratio': 0.85, 'T55': 380.0, 'F9': 215.0},
            blending={},
            source='test',
            id=1
        )
        veto, reason = opt_agent._check_param_ranges_veto(valid)
        assert veto is False
        assert reason is None

        invalid = Candidate(
            params={'T6': 390.0},  # Выход за 375°C
            blending={},
            source='test',
            id=2
        )
        veto, reason = opt_agent._check_param_ranges_veto(invalid)
        assert veto is True
        assert 'T6=390.0 вне' in reason

    def test_check_sulfur_veto_with_qa(self, opt_agent):
        cand = Candidate(params={'T6': 360.0}, blending={}, source='test', id=1)

        # Качество в норме
        qa_ok = {'predictions': {'Sulfur': 8.2}, 'risk_spec_violation': {'P_S_gt_10': 0.02}}
        veto, reason = opt_agent._check_sulfur_veto(cand, qa_ok)
        assert veto is False

        # Превышение серы
        qa_high_sulfur = {'predictions': {'Sulfur': 10.8}, 'risk_spec_violation': {'P_S_gt_10': 0.05}}
        veto, reason = opt_agent._check_sulfur_veto(cand, qa_high_sulfur)
        assert veto is True

        # Высокий риск нарушения P(S>10) > 0.1
        qa_high_risk = {'predictions': {'Sulfur': 9.6}, 'risk_spec_violation': {'P_S_gt_10': 0.25}}
        veto, reason = opt_agent._check_sulfur_veto(cand, qa_high_risk)
        assert veto is True

    def test_check_sulfur_veto_fallback_heuristic(self, opt_agent):
        # T6=360 (норма) -> сера ~8.5 -> проходит
        cand_norm = Candidate(params={'T6': 360.0}, blending={}, source='test', id=1)
        veto, _ = opt_agent._check_sulfur_veto(cand_norm, quality_assessment=None)
        assert veto is False

        # T6=355 (холодный реактор) -> сера ~11.5 > 10 -> бракуется
        cand_cold = Candidate(params={'T6': 355.0}, blending={}, source='test', id=2)
        veto, reason = opt_agent._check_sulfur_veto(cand_cold, quality_assessment=None)
        assert veto is True
        assert 'Оценка серы' in reason

    def test_check_blending_sum_veto(self, opt_agent):
        valid = Candidate(
            params={},
            blending={'F30': 0.30, 'F32': 0.25, 'F34': 0.20, 'F56': 0.10, 'F57': 0.10, 'F59': 0.05},
            source='test',
            id=1
        )
        veto, _ = opt_agent._check_blending_sum_veto(valid)
        assert veto is False

        bad_sum = Candidate(
            params={},
            blending={'F30': 0.20, 'F32': 0.20},  # Сумма 0.4 != 1.0
            source='test',
            id=2
        )
        veto, reason = opt_agent._check_blending_sum_veto(bad_sum)
        assert veto is True
        assert 'Сумма=' in reason

    def test_score_candidates(self, opt_agent):
        feasible = [
            Candidate(params={'T6': 360.0, 'F9': 240.0, 'T55': 380.0, 'F2_F26_ratio': 0.85}, blending={}, source='t', id=1),
            Candidate(params={'T6': 374.0, 'F9': 180.0, 'T55': 385.0, 'F2_F26_ratio': 0.94}, blending={}, source='t', id=2)
        ]
        scored = opt_agent.score_candidates(feasible, current_state={'T6': 360.0, 'F9': 215.0})
        assert len(scored) == 2
        for sc in scored:
            assert isinstance(sc, ScoredCandidate)
            assert sc.throughput > 0
            assert 0.0 <= sc.energy_normalized <= 1.0
            assert 0.0 <= sc.risk_normalized <= 1.0
            assert 'throughput_component' in sc.score_breakdown
        # Первый кандидат имеет больший throughput и меньшие энергозатраты/риск
        assert scored[0].score > scored[1].score

    def test_score_candidates_empty_feasible_safe(self, opt_agent):
        """Проверка безопасности при пустом списке допустимых вариантов (без IndexError)."""
        scored = opt_agent.score_candidates([], current_state={})
        assert scored == []


class TestOptimizationAgentFullCycle:
    """Тестирование полного цикла метода optimize()."""

    def test_optimize_execution(self, opt_agent):
        current_state = {'T6': 360.0, 'F9': 215.0, 'T55': 380.0, 'F2_F26_ratio': 0.85}
        result = opt_agent.optimize(
            current_state=current_state,
            quality_assessment={'predictions': {'Sulfur': 8.5}, 'risk_spec_violation': {'P_S_gt_10': 0.0}},
            reliability_assessment={'risk_class': 'low'}
        )

        assert result is not None
        assert hasattr(result, 'recommended')
        assert hasattr(result, 'alternatives')
        assert hasattr(result, 'metrics')
        assert result.metrics['num_candidates'] > 0
        assert result.metrics['num_feasible'] > 0
        assert isinstance(result.recommended, ScoredCandidate)
        assert result.metrics['best_score'] > 0


class TestOptimizationAgentParetoRanking:
    """Тестирование ранжирования по Парето (OPT-05)."""

    def test_rank_pareto_with_dicts(self, opt_agent):
        """Проверка _rank_pareto со словарным форматом согласно ТЗ."""
        scored_dicts = [
            {'id': 1, 'score': 0.75, 'throughput': 250.0, 'energy': 1.2, 'risk': 0.1},
            {'id': 2, 'score': 0.92, 'throughput': 265.0, 'energy': 1.1, 'risk': 0.05},
            {'id': 3, 'score': 0.60, 'throughput': 240.0, 'energy': 1.5, 'risk': 0.2},
            {'id': 4, 'score': 0.85, 'throughput': 258.0, 'energy': 1.3, 'risk': 0.08},
            {'id': 5, 'score': 0.80, 'throughput': 252.0, 'energy': 1.25, 'risk': 0.12},
        ]

        result = opt_agent._rank_pareto(scored_dicts)

        assert isinstance(result, OptimizationResult)
        # Топ-1 должен быть с максимальным score (0.92, id=2)
        assert result.recommended['score'] == 0.92
        assert result.recommended['id'] == 2

        # 2-3 альтернативы
        assert len(result.alternatives) == 3
        assert result.alternatives[0]['score'] == 0.85
        assert result.alternatives[1]['score'] == 0.80
        assert result.alternatives[2]['score'] == 0.75

        # Метрики сохранены
        assert 'throughput' in result.metrics
        assert result.metrics['throughput'] == 265.0
        assert 'energy' in result.metrics
        assert 'risk' in result.metrics
        assert result.metrics['best_score'] == 0.92

        # Проверка распаковки кортежем
        rec, alts = result
        assert rec['id'] == 2
        assert len(alts) == 3

    def test_rank_pareto_with_scored_candidates(self, opt_agent):
        """Проверка ранжирования с объектами ScoredCandidate и diversity."""
        cands = [
            Candidate(params={'T6': 350.0 + i * 2, 'F9': 200.0 + i * 5}, blending={}, source='test', id=i)
            for i in range(5)
        ]
        feasible = opt_agent.apply_veto(cands)
        scored = opt_agent.score_candidates(feasible)

        result = opt_agent._rank_pareto(scored, num_alternatives=3)

        assert isinstance(result, OptimizationResult)
        assert result.recommended == scored[0]
        assert len(result.alternatives) <= 3
        assert 'throughput' in result.metrics
        assert 'energy' in result.metrics
        assert 'risk' in result.metrics

