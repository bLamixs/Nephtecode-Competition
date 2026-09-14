"""
Оркестратор: координация агентов, разрешение конфликтов, формирование рекомендации.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio

from src.agents.quality_agent import QualityAgent
from src.agents.reliability_agent import ReliabilityAgent
from src.agents.optimization_agent import OptimizationAgent
from src.orchestrator.conflict_resolver import ConflictResolver, ConflictResolution
from src.orchestrator.recommendation import Recommendation

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Оркестратор: координация агентов.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Инициализация.

        Args:
            config_path: путь к конфигурации
        """
        self.config_path = config_path

        # Агенты
        self.quality_agent = QualityAgent()
        self.reliability_agent = ReliabilityAgent()
        self.optimization_agent = OptimizationAgent()

        # Разрешение конфликтов
        self.conflict_resolver = ConflictResolver(
            quality_risk_threshold=0.1,
            reliability_risk_threshold=0.7
        )

        logger.info("Orchestrator инициализирован")

    async def run_cycle(
            self,
            telemetry: Dict[str, Any],
            quality_data: Dict[str, Any],
            scenario: str = 'normal'
    ) -> Recommendation:
        """
        Один цикл принятия решения.

        Args:
            telemetry: телеметрия
            quality_data: данные о качестве
            scenario: сценарий (normal, risk, missing, no_solution)

        Returns:
            Recommendation
        """
        logger.info(f"Запуск цикла: scenario={scenario}")

        # ====================================================================
        # ШАГ 1: ПРОВЕРКА ВХОДА
        # ====================================================================

        logger.info("Шаг 1: Проверка входа")

        input_valid, input_reason = self._validate_input(telemetry, quality_data)

        if not input_valid:
            logger.warning(f"Вход не валиден: {input_reason}")
            return self._no_recommendation(f"Недостаточно данных: {input_reason}")

        # ====================================================================
        # ШАГ 2: ЗАПРОС АГЕНТОВ
        # ====================================================================

        logger.info("Шаг 2: Запрос агентов")

        # Качество
        quality_assessment = await self._call_quality_agent(telemetry, quality_data)
        logger.info(f"Качество: confidence={quality_assessment.confidence:.3f}")

        # Надёжность
        reliability_assessment = await self._call_reliability_agent(telemetry)
        logger.info(f"Надёжность: risk_class={reliability_assessment.risk_class}")

        # Оптимизация
        optimization_result = await self._call_optimization_agent(
            telemetry,
            quality_assessment,
            reliability_assessment
        )
        logger.info(f"Оптимизация: {optimization_result.metrics['num_feasible']} допустимых вариантов")

        # ====================================================================
        # ШАГ 3: РАЗРЕШЕНИЕ КОНФЛИКТОВ
        # ====================================================================

        logger.info("Шаг 3: Разрешение конфликтов")

        conflict_resolution = self.conflict_resolver.resolve(
            candidates=optimization_result.ranked,
            quality_assessment={
                'risk_spec_violation': quality_assessment.risk_spec_violation,
                'confidence': quality_assessment.confidence
            },
            reliability_assessment={
                'risk_class': reliability_assessment.risk_class,
                'risk_index': reliability_assessment.risk_index,
                'risk_factors': reliability_assessment.risk_factors
            }
        )

        if not conflict_resolution.is_resolved:
            logger.warning(f"Конфликт не разрешён: {conflict_resolution.conflict_type.value}")
            return self._no_recommendation(conflict_resolution.explanation)

        # ====================================================================
        # ШАГ 4: ФОРМИРОВАНИЕ РЕКОМЕНДАЦИИ
        # ====================================================================

        logger.info("Шаг 4: Формирование рекомендации")

        recommendation = self._build_recommendation(
            quality_assessment=quality_assessment,
            reliability_assessment=reliability_assessment,
            optimization_result=optimization_result,
            conflict_resolution=conflict_resolution
        )

        logger.info(f"Рекомендация: {recommendation.status}")
        logger.info(f"Объяснение: {recommendation.explanation}")

        return recommendation

    def _validate_input(
            self,
            telemetry: Dict[str, Any],
            quality_data: Dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Проверка входных данных.

        Returns:
            (валидно, причина)
        """
        # 1. Полнота (пропуски < 30%)
        # TODO: реализовать

        # 2. Актуальность (age_min < 120)
        # TODO: реализовать

        # 3. Согласованность (ЛИМС/ПАК не противоречат)
        # TODO: реализовать

        return True, "OK"

    async def _call_quality_agent(
            self,
            telemetry: Dict[str, Any],
            quality_data: Dict[str, Any]
    ):
        """Вызов Quality Agent."""
        return await self.quality_agent.assess(telemetry, quality_data)

    async def _call_reliability_agent(
            self,
            telemetry: Dict[str, Any]
    ):
        """Вызов Reliability Agent."""
        return await self.reliability_agent.assess(telemetry)

    async def _call_optimization_agent(
            self,
            telemetry: Dict[str, Any],
            quality_assessment: Any,
            reliability_assessment: Any
    ):
        """Вызов Optimization Agent."""
        return await self.optimization_agent.optimize(
            current_state=telemetry,
            quality_assessment=quality_assessment,
            reliability_assessment=reliability_assessment
        )

    def _build_recommendation(
            self,
            quality_assessment: Any,
            reliability_assessment: Any,
            optimization_result: Any,
            conflict_resolution: ConflictResolution
    ) -> Recommendation:
        """
        Формирование рекомендации.

        Args:
            quality_assessment: оценка качества
            reliability_assessment: оценка надёжности
            optimization_result: результат оптимизации
            conflict_resolution: разрешение конфликтов

        Returns:
            Recommendation
        """
        recommended = conflict_resolution.recommended_candidate

        return Recommendation(
            recommendation_id=f"rec_{datetime.now():%Y%m%d_%H%M%S}",
            timestamp=datetime.now().isoformat(),
            state={},  # TODO: текущее состояние
            problem_type="OPTIMIZATION",
            action=recommended.get('params', {}) if recommended else {},
            expected_effect={
                'throughput': recommended.get('throughput', 0) if recommended else 0,
                'energy_proxy': recommended.get('energy_proxy', 0) if recommended else 0,
                'risk_index': recommended.get('risk_index', 0) if recommended else 0
            },
            constraints_checked=conflict_resolution.checked_constraints,
            confidence=quality_assessment.confidence,
            status="RECOMMENDED",
            alternatives=conflict_resolution.alternatives,
            explanation=conflict_resolution.explanation
        )

    def _no_recommendation(self, reason: str) -> Recommendation:
        """
        Отказ от рекомендации.

        Args:
            reason: причина отказа

        Returns:
            Recommendation
        """
        return Recommendation(
            recommendation_id=f"rec_{datetime.now():%Y%m%d_%H%M%S}",
            timestamp=datetime.now().isoformat(),
            state={},
            problem_type="NO_RECOMMENDATION",
            action={},
            expected_effect={},
            constraints_checked=[],
            confidence=0.0,
            status="NO_RECOMMENDATION",
            alternatives=[],
            explanation=f"Надёжной рекомендации нет: {reason}"
        )


# ============================================================================
# CLI ДЛЯ ТЕСТИРОВАНИЯ
# ============================================================================

if __name__ == '__main__':
    import asyncio
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    orchestrator = Orchestrator()

    # Моки данных
    telemetry = {'T6': 295.0, 'F26': 100.0}
    quality_data = {'Sulfur': 9.2, 'age_min': 45}

    # Запуск цикла
    recommendation = asyncio.run(
        orchestrator.run_cycle(telemetry, quality_data, scenario='normal')
    )

    print("\n" + "=" * 80)
    print("Рекомендация")
    print("=" * 80)
    print(f"Статус: {recommendation.status}")
    print(f"Объяснение: {recommendation.explanation}")
    print(f"Действие: {recommendation.action}")
    print(f"Ожидаемый эффект: {recommendation.expected_effect}")