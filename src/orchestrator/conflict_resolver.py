"""
Модуль разрешения конфликтов между агентами.

Приоритеты:
1. Качество (сера ≤ 10 мг/кг, другие спецификации) — жёсткое veto
2. Надёжность (риск оборудования) — жёсткое veto
3. Экономика (throughput, energy) — компромисс

Если качество и надёжность противоречат → отказ с объяснением.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Тип конфликта."""
    NONE = "none"
    QUALITY_VETO = "quality_veto"
    RELIABILITY_VETO = "reliability_veto"
    QUALITY_VS_RELIABILITY = "quality_vs_reliability"
    NO_FEASIBLE = "no_feasible"


@dataclass
class ConflictResolution:
    """Результат разрешения конфликта."""
    conflict_type: ConflictType
    is_resolved: bool
    recommended_candidate: Optional[Dict[str, Any]]
    alternatives: List[Dict[str, Any]]
    explanation: str
    veto_reasons: List[str]
    checked_constraints: List[Dict[str, Any]]


class ConflictResolver:
    """
    Разрешение конфликтов между агентами.

    Логика:
    1. Проверка качества (сера ≤ 10, другие спецификации)
    2. Проверка надёжности (risk_class != 'high')
    3. Выбор лучшего по экономике (score)

    Если 1 или 2 не проходят → veto.
    Если 1 и 2 противоречат → отказ.
    """

    def __init__(
            self,
            quality_risk_threshold: float = 0.1,
            reliability_risk_threshold: float = 0.7
    ):
        """
        Инициализация.

        Args:
            quality_risk_threshold: порог риска качества (P(S>10) > threshold → veto)
            reliability_risk_threshold: порог риска надёжности (risk_index > threshold → veto)
        """
        self.quality_risk_threshold = quality_risk_threshold
        self.reliability_risk_threshold = reliability_risk_threshold

        logger.info(
            f"ConflictResolver инициализирован: "
            f"quality_risk_threshold={quality_risk_threshold}, "
            f"reliability_risk_threshold={reliability_risk_threshold}"
        )

    def resolve(
            self,
            candidates: List[Dict[str, Any]],
            quality_assessment: Dict[str, Any],
            reliability_assessment: Dict[str, Any]
    ) -> ConflictResolution:
        """
        Разрешение конфликтов.

        Args:
            candidates: список кандидатов (с оценками)
            quality_assessment: оценка качества (от Quality Agent)
            reliability_assessment: оценка надёжности (от Reliability Agent)

        Returns:
            ConflictResolution
        """
        logger.info(f"Разрешение конфликтов: {len(candidates)} кандидатов")

        veto_reasons = []
        checked_constraints = []

        # ====================================================================
        # ШАГ 1: ПРОВЕРКА КАЧЕСТВА (жёсткое veto)
        # ====================================================================

        logger.info("Шаг 1: Проверка качества")

        quality_veto, quality_reasons, quality_constraints = self._check_quality_veto(
            candidates=candidates,
            quality_assessment=quality_assessment
        )

        checked_constraints.extend(quality_constraints)

        if quality_veto:
            veto_reasons.extend(quality_reasons)
            logger.warning(f"Veto по качеству: {quality_reasons}")

            return ConflictResolution(
                conflict_type=ConflictType.QUALITY_VETO,
                is_resolved=False,
                recommended_candidate=None,
                alternatives=[],
                explanation=f"Недопустимое качество: {'; '.join(quality_reasons)}",
                veto_reasons=veto_reasons,
                checked_constraints=checked_constraints
            )

        # ====================================================================
        # ШАГ 2: ПРОВЕРКА НАДЁЖНОСТИ (жёсткое veto)
        # ====================================================================

        logger.info("Шаг 2: Проверка надёжности")

        reliability_veto, reliability_reasons, reliability_constraints = self._check_reliability_veto(
            candidates=candidates,
            reliability_assessment=reliability_assessment
        )

        checked_constraints.extend(reliability_constraints)

        if reliability_veto:
            veto_reasons.extend(reliability_reasons)
            logger.warning(f"Veto по надёжности: {reliability_reasons}")

            return ConflictResolution(
                conflict_type=ConflictType.RELIABILITY_VETO,
                is_resolved=False,
                recommended_candidate=None,
                alternatives=[],
                explanation=f"Недопустимый риск оборудования: {'; '.join(reliability_reasons)}",
                veto_reasons=veto_reasons,
                checked_constraints=checked_constraints
            )

        # ====================================================================
        # ШАГ 3: ПРОВЕРКА НА ПРОТИВОРЕЧИЯ
        # ====================================================================

        logger.info("Шаг 3: Проверка на противоречия")

        # Если качество OK, но надёжность high → противоречие
        if not quality_veto and reliability_veto:
            logger.error("Противоречие: качество OK, но надёжность high")

            return ConflictResolution(
                conflict_type=ConflictType.QUALITY_VS_RELIABILITY,
                is_resolved=False,
                recommended_candidate=None,
                alternatives=[],
                explanation=(
                    "Противоречие: качество допустимо, но риск оборудования недопустим. "
                    "Рекомендация невозможна."
                ),
                veto_reasons=veto_reasons,
                checked_constraints=checked_constraints
            )

        # ====================================================================
        # ШАГ 4: ВЫБОР ЛУЧШЕГО ПО ЭКОНОМИКЕ
        # ====================================================================

        logger.info("Шаг 4: Выбор лучшего по экономике")

        if not candidates:
            logger.warning("Нет допустимых кандидатов")

            return ConflictResolution(
                conflict_type=ConflictType.NO_FEASIBLE,
                is_resolved=False,
                recommended_candidate=None,
                alternatives=[],
                explanation="Нет допустимых вариантов управления",
                veto_reasons=veto_reasons,
                checked_constraints=checked_constraints
            )

        # Сортировка по score (убывание)
        ranked = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)

        recommended = ranked[0]
        alternatives = ranked[1:4] if len(ranked) > 1 else []

        logger.info(f"Выбран кандидат #{recommended.get('id', 'N/A')} со score={recommended.get('score', 0):.3f}")

        return ConflictResolution(
            conflict_type=ConflictType.NONE,
            is_resolved=True,
            recommended_candidate=recommended,
            alternatives=alternatives,
            explanation=(
                f"Выбран вариант с score={recommended.get('score', 0):.3f}. "
                f"Доступно {len(alternatives)} альтернатив."
            ),
            veto_reasons=veto_reasons,
            checked_constraints=checked_constraints
        )

    def _check_quality_veto(
            self,
            candidates: List[Dict[str, Any]],
            quality_assessment: Dict[str, Any]
    ) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
        """
        Проверка veto по качеству.

        Args:
            candidates: список кандидатов
            quality_assessment: оценка качества

        Returns:
            (veto, список причин, список проверенных ограничений)
        """
        veto = False
        reasons = []
        constraints = []

        # 1. Проверка серы (сера ≤ 10 мг/кг)
        sulfur_risk = quality_assessment.get('risk_spec_violation', {}).get('P_S_gt_10', 0.0)

        constraints.append({
            'constraint': 'Сера ≤ 10 мг/кг',
            'metric': f'P(S>10) = {sulfur_risk:.3f}',
            'threshold': self.quality_risk_threshold,
            'status': 'PASS' if sulfur_risk <= self.quality_risk_threshold else 'FAIL'
        })

        if sulfur_risk > self.quality_risk_threshold:
            veto = True
            reasons.append(
                f"Риск выхода серы за спецификацию: P(S>10)={sulfur_risk:.3f} > {self.quality_risk_threshold}")

        # 2. Проверка других показателей (T95, CFPP, etc.)
        for metric, threshold in quality_assessment.get('risk_spec_violation', {}).items():
            if metric == 'P_S_gt_10':
                continue  # уже проверено

            # Порог для каждого показателя (можно вынести в config)
            metric_threshold = 0.15  # 15% риск

            constraints.append({
                'constraint': f'{metric} ≤ {metric_threshold:.2f}',
                'metric': f'{metric} = {threshold:.3f}',
                'threshold': metric_threshold,
                'status': 'PASS' if threshold <= metric_threshold else 'FAIL'
            })

            if threshold > metric_threshold:
                veto = True
                reasons.append(f"Риск выхода {metric} за спецификацию: {threshold:.3f} > {metric_threshold}")

        # 3. Проверка уверенности качества
        confidence = quality_assessment.get('confidence', 0.0)
        min_confidence = 0.3

        constraints.append({
            'constraint': f'Уверенность качества ≥ {min_confidence}',
            'metric': f'confidence = {confidence:.3f}',
            'threshold': min_confidence,
            'status': 'PASS' if confidence >= min_confidence else 'FAIL'
        })

        if confidence < min_confidence:
            veto = True
            reasons.append(f"Низкая уверенность качества: {confidence:.3f} < {min_confidence}")

        return veto, reasons, constraints

    def _check_reliability_veto(
            self,
            candidates: List[Dict[str, Any]],
            reliability_assessment: Dict[str, Any]
    ) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
        """
        Проверка veto по надёжности.

        Args:
            candidates: список кандидатов
            reliability_assessment: оценка надёжности

        Returns:
            (veto, список причин, список проверенных ограничений)
        """
        veto = False
        reasons = []
        constraints = []

        # 1. Проверка класса риска
        risk_class = reliability_assessment.get('risk_class', 'low')

        constraints.append({
            'constraint': 'Класс риска оборудования',
            'metric': f'risk_class = {risk_class}',
            'threshold': 'high',
            'status': 'PASS' if risk_class != 'high' else 'FAIL'
        })

        if risk_class == 'high':
            veto = True
            reasons.append(f"Недопустимый риск оборудования: {risk_class}")

        # 2. Проверка индекса риска
        risk_index = reliability_assessment.get('risk_index', 0.0)

        constraints.append({
            'constraint': f'Индекс риска ≤ {self.reliability_risk_threshold:.2f}',
            'metric': f'risk_index = {risk_index:.3f}',
            'threshold': self.reliability_risk_threshold,
            'status': 'PASS' if risk_index <= self.reliability_risk_threshold else 'FAIL'
        })

        if risk_index > self.reliability_risk_threshold:
            veto = True
            reasons.append(f"Высокий индекс риска: {risk_index:.3f} > {self.reliability_risk_threshold}")

        # 3. Проверка факторов риска
        risk_factors = reliability_assessment.get('risk_factors', [])

        critical_factors = [
            f for f in risk_factors
            if f.get('deviation', 0) > 2.0  # z-score > 2
        ]

        if critical_factors:
            veto = True
            factor_names = [f.get('tag', 'N/A') for f in critical_factors]
            reasons.append(f"Критические факторы риска: {', '.join(factor_names)}")

        return veto, reasons, constraints


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == '__main__':
    import json

    # Тест 1: Качество OK, надёжность OK → выбор лучшего
    resolver = ConflictResolver()

    candidates = [
        {'id': 1, 'score': 0.85, 'T6': 295.0},
        {'id': 2, 'score': 0.75, 'T6': 298.0},
        {'id': 3, 'score': 0.65, 'T6': 300.0},
    ]

    quality_assessment = {
        'risk_spec_violation': {'P_S_gt_10': 0.05},
        'confidence': 0.8
    }

    reliability_assessment = {
        'risk_class': 'low',
        'risk_index': 0.25,
        'risk_factors': []
    }

    result = resolver.resolve(candidates, quality_assessment, reliability_assessment)

    print("\n" + "=" * 80)
    print("Тест 1: Качество OK, надёжность OK")
    print("=" * 80)
    print(f"Конфликт: {result.conflict_type.value}")
    print(f"Разрешён: {result.is_resolved}")
    print(f"Рекомендация: {result.recommended_candidate}")
    print(f"Объяснение: {result.explanation}")
    print(f"Проверенные ограничения: {json.dumps(result.checked_constraints, indent=2, ensure_ascii=False)}")

    # Тест 2: Качество veto (сера > 10)
    quality_assessment_veto = {
        'risk_spec_violation': {'P_S_gt_10': 0.25},  # > 0.1
        'confidence': 0.8
    }

    result = resolver.resolve(candidates, quality_assessment_veto, reliability_assessment)

    print("\n" + "=" * 80)
    print("Тест 2: Качество veto (сера > 10)")
    print("=" * 80)
    print(f"Конфликт: {result.conflict_type.value}")
    print(f"Разрешён: {result.is_resolved}")
    print(f"Объяснение: {result.explanation}")
    print(f"Veto причины: {result.veto_reasons}")

    # Тест 3: Надёжность veto (risk_class = high)
    reliability_assessment_veto = {
        'risk_class': 'high',
        'risk_index': 0.85,
        'risk_factors': [{'tag': 'T6', 'deviation': 2.5}]
    }

    result = resolver.resolve(candidates, quality_assessment, reliability_assessment_veto)

    print("\n" + "=" * 80)
    print("Тест 3: Надёжность veto (risk_class = high)")
    print("=" * 80)
    print(f"Конфликт: {result.conflict_type.value}")
    print(f"Разрешён: {result.is_resolved}")
    print(f"Объяснение: {result.explanation}")
    print(f"Veto причины: {result.veto_reasons}")

    # Тест 4: Противоречие (качество OK, надёжность high)
    result = resolver.resolve(candidates, quality_assessment, reliability_assessment_veto)

    print("\n" + "=" * 80)
    print("Тест 4: Противоречие (качество OK, надёжность high)")
    print("=" * 80)
    print(f"Конфликт: {result.conflict_type.value}")
    print(f"Разрешён: {result.is_resolved}")
    print(f"Объяснение: {result.explanation}")