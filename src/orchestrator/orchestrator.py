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
from typing import Dict, Any, Optional, Tuple, List
import asyncio
import numpy as np
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

    def _load_scenario_data(self, scenario: str = 'normal') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Загрузка данных по имени сценария.
        Поддерживает: 'normal', 'risk', 'missing', 'no_solution'.
        """
        scenario_key = str(scenario).lower().strip()
        scenario_file = None
        for cand in [f"{scenario_key}.json", f"{scenario_key}_operation.json", f"{scenario_key}_sulfur_growth.json"]:
            path = Path('scenarios') / cand
            if path.exists():
                scenario_file = path
                break

        mock_state = {}
        if scenario_file and scenario_file.exists():
            try:
                with open(scenario_file, 'r', encoding='utf-8') as f:
                    sc_data = json.load(f)
                    mock_state = sc_data.get('mock_state', {})
            except Exception as e:
                logger.warning(f"Не удалось прочитать {scenario_file}: {e}")

        n_rows = 100
        dates = pd.date_range(end=datetime.now(), periods=n_rows, freq='1min')

        if 'missing' in scenario_key:
            telemetry = pd.DataFrame({
                'date': dates,
                'T6': [np.nan] * n_rows,
                'F9': [np.nan] * n_rows,
                'F2_F26_ratio': [0.85] * n_rows,
                'T55': [320.0] * n_rows,
            })
            quality_data = pd.DataFrame([
                {
                    'tag': 'Sulfur',
                    'value': 8.0,
                    'source': 'LIMS',
                    'age_min': float(mock_state.get('lims_age_hours', 52.0)) * 60.0,
                    'timestamp': dates[-1]
                },
                {
                    'tag': 'D15',
                    'value': 835.0,
                    'source': 'PAK',
                    'age_min': float(mock_state.get('pak_age_minutes', 250.0)),
                    'timestamp': dates[-1]
                }
            ])
        elif 'no_solution' in scenario_key:
            telemetry = pd.DataFrame({
                'date': dates,
                'T6': np.random.normal(380.0, 0.2, n_rows),
                'F9': np.random.normal(80.0, 1.0, n_rows),
                'F2_F26_ratio': np.random.normal(0.85, 0.01, n_rows),
                'T55': np.random.normal(370.0, 1.0, n_rows),
            })
            quality_data = pd.DataFrame([
                {
                    'tag': 'Sulfur',
                    'value': float(mock_state.get('pak_sulfur', 11.5)),
                    'source': 'PAK',
                    'age_min': float(mock_state.get('pak_age_minutes', 8.0)),
                    'timestamp': dates[-1]
                },
                {
                    'tag': 'D15',
                    'value': 838.0,
                    'source': 'LIMS',
                    'age_min': 45.0,
                    'timestamp': dates[-1]
                }
            ])
        elif 'risk' in scenario_key:
            telemetry = pd.DataFrame({
                'date': dates,
                'T6': [360.0] * n_rows,
                'T11_hydro': [365.0] * n_rows,
                'F9': [215.0] * n_rows,
                'F2_F26_ratio': [0.85] * n_rows,
                'T55': [380.0] * n_rows,
            })
            quality_data = pd.DataFrame([
                {
                    'tag': 'Sulfur',
                    'value': float(mock_state.get('pak_sulfur', 9.8)),
                    'source': 'PAK',
                    'age_min': float(mock_state.get('pak_age_minutes', 5.0)),
                    'timestamp': dates[-1]
                },
                {
                    'tag': 'D15',
                    'value': 835.0,
                    'source': 'LIMS',
                    'age_min': 40.0,
                    'timestamp': dates[-1]
                }
            ])
        else:
            telemetry = pd.DataFrame({
                'date': dates,
                'T6': [360.0] * n_rows,
                'T11_hydro': [365.0] * n_rows,
                'F9': [215.0] * n_rows,
                'F2_F26_ratio': [0.85] * n_rows,
                'T55': [380.0] * n_rows,
            })
            quality_data = pd.DataFrame([
                {
                    'tag': 'Sulfur',
                    'value': float(mock_state.get('pak_sulfur', 6.8)),
                    'source': 'PAK',
                    'age_min': float(mock_state.get('pak_age_minutes', 10.0)),
                    'timestamp': dates[-1]
                },
                {
                    'tag': 'D15',
                    'value': 832.0,
                    'source': 'LIMS',
                    'age_min': 45.0,
                    'timestamp': dates[-1]
                }
            ])

        return telemetry, quality_data

    async def run_cycle(
        self,
        telemetry: Optional[pd.DataFrame] = None,
        quality_data: Optional[pd.DataFrame] = None,
        scenario: str = 'normal'
    ) -> Recommendation:
        """
        Один цикл принятия решения.

        Args:
            telemetry: телеметрия (если None, загружается по сценарию)
            quality_data: данные о качестве (если None, загружается по сценарию)
            scenario: сценарий ('normal', 'risk', 'missing', 'no_solution')

        Returns:
            Recommendation
        """
        if telemetry is None or quality_data is None:
            sc_telemetry, sc_quality = self._load_scenario_data(scenario)
            if telemetry is None:
                telemetry = sc_telemetry
            if quality_data is None:
                quality_data = sc_quality

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
        cand = conflict_resolution.recommended_candidate
        if cand is None and optimization_result is not None:
            cand = getattr(optimization_result, 'recommended', None)

        if cand is None:
            return self._no_recommendation("Нет допустимых вариантов управления", cycle_id=cycle_id)

        t6_curr = float(telemetry['T6'].dropna().iloc[-1]) if 'T6' in telemetry and not telemetry['T6'].dropna().empty else 360.0
        f2_ratio_curr = float(telemetry['F2_F26_ratio'].dropna().iloc[-1]) if 'F2_F26_ratio' in telemetry and not telemetry['F2_F26_ratio'].dropna().empty else 0.85
        t55_curr = float(telemetry['T55'].dropna().iloc[-1]) if 'T55' in telemetry and not telemetry['T55'].dropna().empty else 380.0
        f9_curr = float(telemetry['F9'].dropna().iloc[-1]) if 'F9' in telemetry and not telemetry['F9'].dropna().empty else 215.0

        sulfur_row = quality_data[quality_data['tag'] == 'Sulfur'] if 'tag' in quality_data.columns else pd.DataFrame()
        sulfur_curr = float(sulfur_row['value'].iloc[-1]) if not sulfur_row.empty else 9.2
        sulfur_age = float(sulfur_row['age_min'].iloc[-1]) if not sulfur_row.empty and 'age_min' in sulfur_row.columns else 45.0

        state = {
            'T6': round(t6_curr, 2),
            'F2_F26_ratio': round(f2_ratio_curr, 4),
            'T55': round(t55_curr, 2),
            'F9': round(f9_curr, 2),
            'Sulfur_current': round(sulfur_curr, 2),
            'Sulfur_age_min': round(sulfur_age, 1)
        }

        # Кандидат
        cand_dict = cand if isinstance(cand, dict) else (cand.__dict__ if hasattr(cand, '__dict__') else {})
        cand_action = cand_dict.get('action', {})
        if hasattr(cand, 'candidate') and hasattr(cand.candidate, 'params'):
            cand_action = cand.candidate.params
        elif hasattr(cand, 'action') and isinstance(cand.action, dict):
            cand_action = cand.action
        elif 'params' in cand_dict:
            cand_action = cand_dict['params']

        actions = []
        if 'T6' in cand_action:
            to_t6 = float(cand_action['T6'])
            actions.append(create_action_item('T6', 'Температура реактора', t6_curr, to_t6, '°C'))
        if 'F2_F26_ratio' in cand_action:
            to_ratio = float(cand_action['F2_F26_ratio'])
            actions.append(create_action_item('F2_F26_ratio', 'Соотношение ВСГ / сырье', f2_ratio_curr, to_ratio, '-'))
        if 'T55' in cand_action:
            to_t55 = float(cand_action['T55'])
            actions.append(create_action_item('T55', 'Температура низа колонны', t55_curr, to_t55, '°C'))
        if 'F9' in cand_action:
            to_f9 = float(cand_action['F9'])
            actions.append(create_action_item('F9', 'Расход сырья', f9_curr, to_f9, 'м³/ч'))

        # Эффект
        pred_q = cand_dict.get('predicted_quality', {})
        s_60 = pred_q.get('Sulfur_60min', pred_q.get('Sulfur', 7.5))
        d15_60 = pred_q.get('D15_60min', pred_q.get('D15', 835.0))
        t_60 = pred_q.get('T95_60min', pred_q.get('T95', 355.0))
        cfpp_60 = pred_q.get('CFPP_60min', pred_q.get('CFPP', -20.0))

        throughput = float(cand_dict.get('throughput', f9_curr))
        throughput_delta = throughput - f9_curr
        energy_proxy = float(cand_dict.get('energy_proxy', 0.45))
        rel_risk = getattr(reliability_assessment, 'risk_index', 0.15)
        risk_idx = float(cand_dict.get('risk_index', rel_risk))

        expected_effect = ExpectedEffect(
            sulfur_60min=round(s_60, 2),
            sulfur_delta=round(s_60 - sulfur_curr, 2),
            d15_60min=round(d15_60, 1),
            t95_60min=round(t_60, 1),
            cfpp_60min=round(cfpp_60, 1),
            throughput=round(throughput, 2),
            throughput_delta=round(throughput_delta, 2),
            energy_proxy=round(energy_proxy, 4),
            risk_index=round(risk_idx, 3),
            risk_delta=round(risk_idx - rel_risk, 3)
        )

        # Ограничения
        constraints_checked = []
        for c in conflict_resolution.checked_constraints:
            try:
                thresh_float = float(c.get('threshold', 0.0))
            except (ValueError, TypeError):
                thresh_float = 0.0

            try:
                pred_float = float(c.get('predicted_value', c.get('value', 0.0)))
            except (ValueError, TypeError):
                pred_float = 0.0

            constraints_checked.append(
                create_constraint_check(
                    constraint_id=c.get('constraint_id', f"C_{len(constraints_checked)+1:03d}"),
                    constraint=c.get('constraint', c.get('name', 'Ограничение')),
                    predicted_value=pred_float,
                    threshold=thresh_float
                )
            )
        if not constraints_checked:
            constraints_checked.append(create_constraint_check('C001', 'Сера ≤ 10 мг/кг', s_60, 10.0))
            constraints_checked.append(create_constraint_check('C002', 'Риск оборудования < 0.70', risk_idx, 0.70))

        # Альтернативы
        alts = []
        ref_score = float(getattr(cand, 'score', cand_dict.get('score', 0.5)))
        for a in conflict_resolution.alternatives:
            a_dict = a if isinstance(a, dict) else (a.__dict__ if hasattr(a, '__dict__') else {})
            a_action = a_dict.get('action', {})
            if hasattr(a, 'candidate') and hasattr(a.candidate, 'params'):
                a_action = a.candidate.params
            elif 'params' in a_dict:
                a_action = a_dict['params']

            a_id = getattr(getattr(a, 'candidate', None), 'id', a_dict.get('id', len(alts) + 2))
            a_score = float(getattr(a, 'score', a_dict.get('score', 0.0)))
            a_throughput = float(getattr(a, 'throughput', a_dict.get('throughput', throughput)))
            a_energy = float(getattr(a, 'energy_proxy', a_dict.get('energy_proxy', energy_proxy)))
            a_risk = float(getattr(a, 'risk_index', a_dict.get('risk_index', risk_idx)))

            alts.append(
                create_alternative(
                    id=int(a_id) if isinstance(a_id, (int, float, str)) and str(a_id).isdigit() else len(alts) + 2,
                    action=a_action,
                    score=a_score,
                    throughput=a_throughput,
                    energy_proxy=a_energy,
                    risk_index=a_risk,
                    reference_score=ref_score,
                    reference_throughput=throughput
                )
            )

        # Объяснение
        action_parts = [f"{a.name}: {a.from_value:.1f} → {a.to_value:.1f} {a.unit}" for a in actions]
        action_str = "; ".join(action_parts) if action_parts else "сохранение текущего режима"
        rel_class = getattr(reliability_assessment, 'risk_class', 'LOW')
        explanation = (
            f"Рекомендовано: {action_str}. "
            f"Ожидаемая сера через 60 мин: {s_60:.2f} мг/кг (Δ={s_60 - sulfur_curr:+.2f} мг/кг). "
            f"Производительность: {throughput:.1f} м³/ч. "
            f"Индекс риска оборудования: {risk_idx:.2f} ({rel_class}). "
            f"Решение сбалансировано по качеству, производительности и ресурсу катализатора."
        )

        conf = getattr(quality_assessment, 'confidence', 0.85)

        return Recommendation(
            recommendation_id=f"rec_{datetime.now():%Y%m%d_%H%M%S}",
            timestamp=datetime.now().isoformat(),
            state=state,
            problem_type="RISK_SPEC_VIOLATION" if s_60 > 9.0 else "SUBOPTIMAL",
            action=actions,
            expected_effect=expected_effect,
            constraints_checked=constraints_checked,
            confidence=round(conf, 2),
            status="RECOMMENDED",
            alternatives=alts,
            explanation=explanation,
            metadata={
                'cycle_id': cycle_id,
                'candidate_id': cand_dict.get('id', 1),
                'score': cand_dict.get('score', 0.0)
            }
        )

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
            'T6': float(telemetry['T6'].iloc[-1]) if 'T6' in telemetry else 360.0,
            'F2_F26_ratio': float(telemetry['F2_F26_ratio'].iloc[-1]) if 'F2_F26_ratio' in telemetry else 0.85,
            'T55': float(telemetry['T55'].iloc[-1]) if 'T55' in telemetry else 380.0,
            'F9': float(telemetry['F9'].iloc[-1]) if 'F9' in telemetry else 215.0,
        }

        return self.optimization_agent.optimize(
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
        'T6': np.random.normal(360, 2, 100),
        'F9': np.random.normal(215, 10, 100),
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