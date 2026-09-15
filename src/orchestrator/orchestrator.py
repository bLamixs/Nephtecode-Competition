"""
Оркестратор: координация агентов, разрешение конфликтов, формирование рекомендации.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
import pandas as pd

from src.agents.quality_agent import QualityAgent
from src.agents.reliability_agent import ReliabilityAgent
from src.agents.optimization_agent import OptimizationAgent
from src.orchestrator.conflict_resolver import ConflictResolver, ConflictResolution
from src.orchestrator.input_validator import InputValidator, ValidationResult
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

        # Валидатор входа
        self.input_validator = InputValidator(
            max_missing_ratio=0.3,
            max_age_min=120,
            consistency_threshold=0.2
        )

        # Разрешение конфликтов
        self.conflict_resolver = ConflictResolver(
            quality_risk_threshold=0.1,
            reliability_risk_threshold=0.7
        )

        logger.info("Orchestrator инициализирован")

    async def run_cycle(
        self,
        telemetry: pd.DataFrame,
        quality_data: pd.DataFrame,
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
        # ШАГ 1: ПРОВЕРКА ВХОДА (ORCH-01)
        # ====================================================================

        logger.info("Шаг 1: Проверка входа")

        validation_result = self.input_validator.validate(telemetry, quality_data)

        if not validation_result.is_valid:
            logger.warning(f"Вход не валиден: {validation_result.reasons}")
            return self._no_recommendation(
                f"Недостаточно данных: {'; '.join(validation_result.reasons)}"
            )

        if validation_result.warnings:
            logger.warning(f"Предупреждения: {validation_result.warnings}")

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
            conflict_resolution=conflict_resolution,
            validation_result=validation_result
        )

        logger.info(f"Рекомендация: {recommendation.status}")
        logger.info(f"Объяснение: {recommendation.explanation}")

        return recommendation

    def _validate_input(
        self,
        telemetry: pd.DataFrame,
        quality_data: pd.DataFrame
    ) -> ValidationResult:
        """
        Проверка входных данных.

        Returns:
            ValidationResult
        """
        return self.input_validator.validate(telemetry, quality_data)

    async def _call_quality_agent(
        self,
        telemetry: pd.DataFrame,
        quality_data: pd.DataFrame
    ):
        """Вызов Quality Agent."""
        return await self.quality_agent.assess(telemetry, quality_data)

    async def _call_reliability_agent(
        self,
        telemetry: pd.DataFrame
    ):
        """Вызов Reliability Agent."""
        return await self.reliability_agent.assess(telemetry)

    async def _call_optimization_agent(
        self,
        telemetry: pd.DataFrame,
        quality_assessment: Any,
        reliability_assessment: Any
    ):
        """Вызов Optimization Agent."""
        current_state = {
            'T6': telemetry['T6'].iloc[-1] if 'T6' in telemetry else 295.0,
            'F2_F26_ratio': telemetry['F2_F26_ratio'].iloc[-1] if 'F2_F26_ratio' in telemetry else 0.85,
            'T55': telemetry['T55'].iloc[-1] if 'T55' in telemetry else 320.0,
            'F9': telemetry['F9'].iloc[-1] if 'F9' in telemetry else 250.0,
        }

        return await self.optimization_agent.optimize(
            current_state=current_state,
            quality_assessment=quality_assessment,
            reliability_assessment=reliability_assessment
        )

    def _build_recommendation(
        self,
        quality_assessment: Any,
        reliability_assessment: Any,
        optimization_result: Any,
        conflict_resolution: ConflictResolution,
        validation_result: ValidationResult
    ) -> Recommendation:
        """
        Формирование рекомендации.

        Args:
            quality_assessment: оценка качества
            reliability_assessment: оценка надёжности
            optimization_result: результат оптимизации
            conflict_resolution: разрешение конфликтов
            validation_result: проверка входа

        Returns:
            Recommendation
        """
        recommended = conflict_resolution.recommended_candidate

        return Recommendation(
            recommendation_id=f"rec_{datetime.now():%Y%m%d_%H%M%S}",
            timestamp=datetime.now().isoformat(),
            state={
                'validation': {
                    'is_valid': validation_result.is_valid,
                    'checks': validation_result.checks,
                    'warnings': validation_result.warnings
                }
            },
            problem_type="OPTIMIZATION",
            action=recommended.candidate.params if recommended else {},
            expected_effect={
                'throughput': recommended.throughput if recommended else 0,
                'energy_proxy': recommended.energy_proxy if recommended else 0,
                'risk_index': recommended.risk_index if recommended else 0
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
    import pandas as pd
    import numpy as np

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    orchestrator = Orchestrator()

    # Моки данных
    telemetry = pd.DataFrame({
        'T6': np.random.normal(295, 2, 100),
        'F9': np.random.normal(250, 10, 100),
        'F2_F26_ratio': np.random.normal(0.85, 0.02, 100),
    })

    quality_data = pd.DataFrame({
        'tag': ['Sulfur', 'D15'],
        'value': [8.5, 835.0],
        'source': ['LIMS', 'PAK'],
        'age_min': [45, 30],
        'timestamp': pd.date_range('2026-01-01', periods=2, freq='1h')
    })

    # Запуск цикла
    recommendation = asyncio.run(
        orchestrator.run_cycle(telemetry, quality_data, scenario='normal')
    )

    print("\n" + "="*80)
    print("Рекомендация")
    print("="*80)
    print(f"Статус: {recommendation.status}")
    print(f"Объяснение: {recommendation.explanation}")
    print(f"Действие: {recommendation.action}")
    print(f"Ожидаемый эффект: {recommendation.expected_effect}")