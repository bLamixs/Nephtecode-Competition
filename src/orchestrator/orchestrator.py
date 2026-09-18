"""
Оркестратор: координация агентов, разрешение конфликтов, формирование рекомендации.

Логирование:
- Каждый шаг логируется (INFO)
- Ошибки и отказы логируются (ERROR)
- Рекомендации сохраняются в БД и JSON
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
import pandas as pd
from pathlib import Path

from src.agents.quality_agent import QualityAgent
from src.agents.reliability_agent import ReliabilityAgent
from src.agents.optimization_agent import OptimizationAgent, ScoredCandidate
from src.orchestrator.conflict_resolver import ConflictResolver, ConflictResolution
from src.orchestrator.input_validator import InputValidator, ValidationResult
from src.orchestrator.recommendation import (
    Recommendation,
    ActionItem,
    ExpectedEffect,
    ConstraintCheck,
    Alternative,
    create_action_item,
    create_constraint_check,
    create_alternative
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Оркестратор: координация агентов.
    """

    def __init__(
        self,
        config_path: str = 'config.yaml',
        log_dir: str = 'logs',
        output_dir: str = 'output'
    ):
        """
        Инициализация.

        Args:
            config_path: путь к конфигурации
            log_dir: папка для логов
            output_dir: папка для результатов
        """
        self.config_path = config_path
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)

        # Создаём папки
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Настройка логирования
        self._setup_logging()

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

    def _setup_logging(self):
        """Настройка логирования в файлы."""
        # Лог оркестратора
        orchestrator_log = self.log_dir / f"orchestrator_{datetime.now():%Y%m%d_%H%M%S}.log"

        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler (оркестратор)
        fh = logging.FileHandler(orchestrator_log, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # File handler (рекомендации)
        recommendation_log = self.log_dir / f"recommendations_{datetime.now():%Y%m%d}.jsonl"
        rh = logging.FileHandler(recommendation_log, encoding='utf-8')
        rh.setLevel(logging.INFO)
        rh.setFormatter(logging.Formatter('%(message)s'))

        # Отдельный logger для рекомендаций
        self.recommendation_logger = logging.getLogger('orchestrator.recommendations')
        self.recommendation_logger.setLevel(logging.INFO)
        self.recommendation_logger.addHandler(rh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

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
            scenario: сценарий

        Returns:
            Recommendation
        """
        cycle_id = f"cycle_{datetime.now():%Y%m%d_%H%M%S}"

        logger.info("="*80)
        logger.info(f"Запуск цикла: {cycle_id}, scenario={scenario}")
        logger.info("="*80)

        # Сохранение входных данных
        self._save_input_data(cycle_id, telemetry, quality_data)

        # ====================================================================
        # ШАГ 1: Проверка входа (ORCH-01)
        # ====================================================================

        logger.info("Шаг 1: Проверка входа")

        validation_result = self.input_validator.validate(telemetry, quality_data)

        if not validation_result.is_valid:
            logger.error(f"Вход не валиден: {validation_result.reasons}")
            recommendation = self._no_recommendation(
                f"Недостаточно данных: {'; '.join(validation_result.reasons)}",
                cycle_id=cycle_id
            )
            self._save_recommendation(recommendation, cycle_id)
            return recommendation

        if validation_result.warnings:
            logger.warning(f"Предупреждения: {validation_result.warnings}")

        # ====================================================================
        # ШАГ 2: Запрос агентов
        # ====================================================================

        logger.info("Шаг 2: Запрос агентов")

        try:
            quality_assessment = await self._call_quality_agent(telemetry, quality_data)
            logger.info(f"Качество: confidence={quality_assessment.confidence:.3f}")

            reliability_assessment = await self._call_reliability_agent(telemetry)
            logger.info(f"Надёжность: risk_class={reliability_assessment.risk_class}")

            optimization_result = await self._call_optimization_agent(
                telemetry,
                quality_assessment,
                reliability_assessment
            )
            logger.info(f"Оптимизация: {optimization_result.metrics['num_feasible']} допустимых вариантов")

        except Exception as e:
            logger.error(f"Ошибка агентов: {e}", exc_info=True)
            recommendation = self._no_recommendation(
                f"Ошибка агентов: {str(e)}",
                cycle_id=cycle_id
            )
            self._save_recommendation(recommendation, cycle_id)
            return recommendation

        # ====================================================================
        # ШАГ 3: Разрешение конфликтов (ORCH-03)
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
            logger.error(f"Конфликт не разрешён: {conflict_resolution.conflict_type.value}")
            recommendation = self._no_recommendation(
                conflict_resolution.explanation,
                cycle_id=cycle_id
            )
            self._save_recommendation(recommendation, cycle_id)
            return recommendation

        # ====================================================================
        # ШАГ 4: Формирование рекомендации (ORCH-04)
        # ====================================================================

        logger.info("Шаг 4: Формирование рекомендации")

        recommendation = self._build_recommendation(
            telemetry=telemetry,
            quality_data=quality_data,
            quality_assessment=quality_assessment,
            reliability_assessment=reliability_assessment,
            optimization_result=optimization_result,
            conflict_resolution=conflict_resolution,
            validation_result=validation_result,
            cycle_id=cycle_id
        )

        logger.info(f"Рекомендация: {recommendation.status}")
        logger.info(f"Объяснение: {recommendation.explanation}")

        # ====================================================================
        # ШАГ 5: Сохранение рекомендации (ORCH-06)
        # ====================================================================

        self._save_recommendation(recommendation, cycle_id)

        logger.info("="*80)
        logger.info(f"Цикл завершён: {cycle_id}")
        logger.info("="*80)

        return recommendation

    def _no_recommendation(self, reason: str, cycle_id: str = None) -> Recommendation:
        """
        Отказ от рекомендации (ORCH-05).

        Args:
            reason: причина отказа
            cycle_id: ID цикла

        Returns:
            Recommendation
        """
        # Конкретизация причины
        specific_reason = self._specific_reason(reason)

        logger.error(f"NO_RECOMMENDATION: {specific_reason}")

        recommendation = Recommendation(
            recommendation_id=f"rec_{datetime.now():%Y%m%d_%H%M%S}",
            timestamp=datetime.now().isoformat(),
            state={},
            problem_type="NO_RECOMMENDATION",
            action=[],
            expected_effect=ExpectedEffect(),
            constraints_checked=[],
            confidence=0.0,
            status="NO_RECOMMENDATION",
            alternatives=[],
            explanation=f"Надёжной рекомендации нет: {specific_reason}",
            metadata={
                'cycle_id': cycle_id,
                'reason': reason
            }
        )

        return recommendation

    def _specific_reason(self, reason: str) -> str:
        """
        Конкретизация причины отказа.

        Args:
            reason: общая причина

        Returns:
            конкретизированная причина
        """
        if 'возраст' in reason.lower() or 'age_min' in reason.lower():
            return f"Последнее ЛИМС устарело ({reason}). ПАК offline."
        elif 'пропуски' in reason.lower() or 'missing' in reason.lower():
            return f"Критические теги содержат пропуски > 30% ({reason})."
        elif 'согласованность' in reason.lower() or 'consistency' in reason.lower():
            return f"ЛИМС и ПАК противоречат друг другу (разница > 20%) ({reason})."
        elif 'нет допустим' in reason.lower() or 'no_feasible' in reason.lower():
            return f"Все варианты нарушают жёсткие ограничения (сера ≤ 10, доли = 100%)."
        else:
            return reason

    def _build_recommendation(
        self,
        telemetry: pd.DataFrame,
        quality_data: pd.DataFrame,
        quality_assessment: Any,
        reliability_assessment: Any,
        optimization_result: Any,
        conflict_resolution: ConflictResolution,
        validation_result: ValidationResult,
        cycle_id: str
    ) -> Recommendation:
        """Формирование рекомендации (ORCH-04)."""
        # ... (код из предыдущей реализации)
        # Добавляем cycle_id в metadata
        pass

    def _save_input_data(
        self,
        cycle_id: str,
        telemetry: pd.DataFrame,
        quality_data: pd.DataFrame
    ):
        """
        Сохранение входных данных (ORCH-06).

        Args:
            cycle_id: ID цикла
            telemetry: телеметрия
            quality_data: качество
        """
        input_dir = self.output_dir / 'input_data'
        input_dir.mkdir(parents=True, exist_ok=True)

        # Сохранение телеметрии (последние 100 строк)
        telemetry_tail = telemetry.tail(100)
        telemetry_path = input_dir / f"{cycle_id}_telemetry.csv"
        telemetry_tail.to_csv(telemetry_path, index=False)

        # Сохранение качества
        quality_path = input_dir / f"{cycle_id}_quality.csv"
        quality_data.to_csv(quality_path, index=False)

        logger.debug(f"Входные данные сохранены: {telemetry_path}, {quality_path}")

    def _save_recommendation(
        self,
        recommendation: Recommendation,
        cycle_id: str
    ):
        """
        Сохранение рекомендации (ORCH-06).

        Args:
            recommendation: рекомендация
            cycle_id: ID цикла
        """
        # 1. Логирование в JSONL (recommendations_YYYYMMDD.jsonl)
        rec_dict = recommendation.to_dict()
        rec_dict['cycle_id'] = cycle_id

        self.recommendation_logger.info(json.dumps(rec_dict, ensure_ascii=False, default=str))

        # 2. Сохранение в отдельный JSON файл
        recommendations_dir = self.output_dir / 'recommendations'
        recommendations_dir.mkdir(parents=True, exist_ok=True)

        rec_path = recommendations_dir / f"{recommendation.recommendation_id}.json"

        with open(rec_path, 'w', encoding='utf-8') as f:
            json.dump(rec_dict, f, ensure_ascii=False, indent=2, default=str)

        # 3. Логирование в orchestrator.log
        if recommendation.status == "RECOMMENDED":
            logger.info(
                f"RECOMMENDATION: {recommendation.recommendation_id}, "
                f"action={len(recommendation.action)} действий, "
                f"confidence={recommendation.confidence:.2f}"
            )
        else:
            logger.error(
                f"NO_RECOMMENDATION: {recommendation.recommendation_id}, "
                f"reason={recommendation.explanation}"
            )

        logger.debug(f"Рекомендация сохранена: {rec_path}")

    async def _call_quality_agent(self, telemetry: pd.DataFrame, quality_data: pd.DataFrame):
        """Вызов Quality Agent."""
        return await self.quality_agent.assess(telemetry, quality_data)

    async def _call_reliability_agent(self, telemetry: pd.DataFrame):
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
            'T6': float(telemetry['T6'].iloc[-1]) if 'T6' in telemetry else 295.0,
            'F2_F26_ratio': float(telemetry['F2_F26_ratio'].iloc[-1]) if 'F2_F26_ratio' in telemetry else 0.85,
            'T55': float(telemetry['T55'].iloc[-1]) if 'T55' in telemetry else 320.0,
            'F9': float(telemetry['F9'].iloc[-1]) if 'F9' in telemetry else 250.0,
        }

        return await self.optimization_agent.optimize(
            current_state=current_state,
            quality_assessment=quality_assessment,
            reliability_assessment=reliability_assessment
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

    # Тест 1: Нормальный цикл
    print("\n" + "="*80)
    print("Тест 1: Нормальный цикл")
    print("="*80)

    telemetry = pd.DataFrame({
        'T6': np.random.normal(295, 2, 100),
        'F9': np.random.normal(250, 10, 100),
        'F2_F26_ratio': np.random.normal(0.85, 0.02, 100),
    })

    quality_data = pd.DataFrame({
        'tag': ['Sulfur', 'D15'],
        'value': [9.2, 835.0],
        'source': ['LIMS', 'PAK'],
        'age_min': [45, 30],
        'timestamp': pd.date_range('2026-01-01', periods=2, freq='1h')
    })

    recommendation = asyncio.run(
        orchestrator.run_cycle(telemetry, quality_data, scenario='normal')
    )

    print(f"Статус: {recommendation.status}")
    print(f"Объяснение: {recommendation.explanation}")

    # Тест 2: Отказ (нехватка данных)
    print("\n" + "="*80)
    print("Тест 2: Отказ (нехватка данных)")
    print("="*80)

    telemetry_missing = pd.DataFrame({
        'T6': [np.nan] * 100,
        'F9': [np.nan] * 100,
    })

    recommendation = asyncio.run(
        orchestrator.run_cycle(telemetry_missing, quality_data, scenario='missing')
    )

    print(f"Статус: {recommendation.status}")
    print(f"Объяснение: {recommendation.explanation}")

    # Тест 3: Отказ (устаревшие данные)
    print("\n" + "="*80)
    print("Тест 3: Отказ (устаревшие данные)")
    print("="*80)

    quality_stale = pd.DataFrame({
        'tag': ['Sulfur', 'D15'],
        'value': [9.2, 835.0],
        'source': ['LIMS', 'PAK'],
        'age_min': [360, 300],  # > 120 мин
        'timestamp': pd.date_range('2026-01-01', periods=2, freq='1h')
    })

    recommendation = asyncio.run(
        orchestrator.run_cycle(telemetry, quality_stale, scenario='missing')
    )

    print(f"Статус: {recommendation.status}")
    print(f"Объяснение: {recommendation.explanation}")