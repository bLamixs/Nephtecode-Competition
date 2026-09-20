"""
Модуль формирования рекомендации оператору.

Recommendation — итоговый ответ системы оператору.
Содержит:
- действие (какие параметры изменить)
- ожидаемый эффект (качество, выпуск, энергия, риск)
- проверенные ограничения
- уверенность
- альтернативы
- объяснение (почему выбран этот вариант)

Status:
- "RECOMMENDED": рекомендация есть
- "NO_RECOMMENDATION": отказ (нехватка данных, нет допустимых вариантов)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import json


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ActionItem:
    """
    Одно действие (изменение параметра).

    Пример:
    {
        "tag": "T6",
        "name": "Температура реактора",
        "from": 360.0,
        "to": 363.0,
        "unit": "°C",
        "delta": 3.0,
        "delta_percent": 0.83
    }
    """
    tag: str
    name: str
    from_value: float
    to_value: float
    unit: str
    delta: float
    delta_percent: float


@dataclass
class ExpectedEffect:
    """
    Ожидаемый эффект от рекомендации.

    Пример:
    {
        "Sulfur_60min": 7.5,
        "Sulfur_delta": -1.7,
        "throughput": 262.5,
        "throughput_delta": 2.5,
        "energy_proxy": 0.45,
        "risk_index": 0.20
    }
    """
    sulfur_60min: Optional[float] = None
    sulfur_delta: Optional[float] = None
    d15_60min: Optional[float] = None
    t95_60min: Optional[float] = None
    cfpp_60min: Optional[float] = None

    throughput: Optional[float] = None
    throughput_delta: Optional[float] = None

    energy_proxy: Optional[float] = None
    energy_delta: Optional[float] = None

    risk_index: Optional[float] = None
    risk_delta: Optional[float] = None


@dataclass
class ConstraintCheck:
    """
    Проверенное ограничение.

    Пример:
    {
        "constraint_id": "C001",
        "constraint": "Сера ≤ 10 мг/кг",
        "predicted_value": 7.5,
        "threshold": 10.0,
        "status": "PASS",  # или "FAIL"
        "margin": 2.5  # запас до нарушения
    }
    """
    constraint_id: str
    constraint: str
    predicted_value: float
    threshold: float
    status: str  # "PASS" или "FAIL"
    margin: float


@dataclass
class Alternative:
    """
    Альтернативный вариант.

    Пример:
    {
        "id": 312,
        "action": {"T6": 297.0, "F2_F26_ratio": 0.87},
        "score": 0.4412,
        "throughput": 260.0,
        "energy_proxy": 0.46,
        "risk_index": 0.21,
        "delta_score": -0.0111,
        "delta_throughput": -2.5
    }
    """
    id: int
    action: Dict[str, float]
    score: float
    throughput: float
    energy_proxy: float
    risk_index: float
    delta_score: float
    delta_throughput: float


@dataclass
class Recommendation:
    """
    Рекомендация оператору.

    Это итоговый ответ системы, который показывается в dashboard.

    Пример JSON:
    {
        "recommendation_id": "rec_20260917_134500",
        "timestamp": "2026-09-17T13:45:00",
        "state": {
            "T6": 360.0,
            "F2_F26_ratio": 0.85,
            "F9": 215.0,
            "Sulfur_current": 9.2,
            "Sulfur_age_min": 45
        },
        "problem_type": "RISK_SPEC_VIOLATION",
        "action": [
            {"tag": "T6", "name": "Температура реактора", "from": 360.0, "to": 363.0, "unit": "°C", "delta": 3.0, "delta_percent": 0.83}
        ],
        "expected_effect": {
            "Sulfur_60min": 7.5,
            "Sulfur_delta": -1.7,
            "throughput": 262.5,
            "throughput_delta": 2.5,
            "energy_proxy": 0.45,
            "risk_index": 0.20
        },
        "constraints_checked": [
            {"constraint_id": "C001", "constraint": "Сера ≤ 10 мг/кг", "predicted_value": 7.5, "threshold": 10.0, "status": "PASS", "margin": 2.5}
        ],
        "confidence": 0.78,
        "status": "RECOMMENDED",
        "alternatives": [
            {"id": 312, "action": {"T6": 297.0, "F2_F26_ratio": 0.87}, "score": 0.4412, "throughput": 260.0, "delta_score": -0.0111}
        ],
        "explanation": "Повышение T6 на 3°C снизит серу с 9.2 до 7.5 мг/кг. Выбран компромисс между качеством и производительностью."
    }
    """

    # Идентификаторы
    recommendation_id: str
    timestamp: str

    # Текущее состояние
    state: Dict[str, Any]

    # Проблема/риск
    problem_type: str  # "RISK_SPEC_VIOLATION", "SUBOPTIMAL", "NO_DATA", "NO_SOLUTION"

    # Действие
    action: List[ActionItem]

    # Ожидаемый эффект
    expected_effect: ExpectedEffect

    # Проверенные ограничения
    constraints_checked: List[ConstraintCheck]

    # Уверенность
    confidence: float

    # Статус
    status: str  # "RECOMMENDED" или "NO_RECOMMENDATION"

    # Альтернативы
    alternatives: List[Alternative]

    # Объяснение
    explanation: str

    # Дополнительные метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_recommended(self) -> bool:
        """
        Проверка: есть ли рекомендация.

        Returns:
            True, если статус "RECOMMENDED"
        """
        return self.status == "RECOMMENDED"

    def to_dict(self) -> Dict[str, Any]:
        """
        Конвертация в dict (для JSON/dashboard).

        Returns:
            Dict с полями рекомендации
        """
        return {
            'recommendation_id': self.recommendation_id,
            'timestamp': self.timestamp,
            'state': self.state,
            'problem_type': self.problem_type,
            'action': [
                {
                    'tag': a.tag,
                    'name': a.name,
                    'from': a.from_value,
                    'to': a.to_value,
                    'unit': a.unit,
                    'delta': a.delta,
                    'delta_percent': a.delta_percent
                }
                for a in self.action
            ],
            'expected_effect': {
                'Sulfur_60min': self.expected_effect.sulfur_60min,
                'Sulfur_delta': self.expected_effect.sulfur_delta,
                'D15_60min': self.expected_effect.d15_60min,
                'throughput': self.expected_effect.throughput,
                'throughput_delta': self.expected_effect.throughput_delta,
                'energy_proxy': self.expected_effect.energy_proxy,
                'risk_index': self.expected_effect.risk_index
            },
            'constraints_checked': [
                {
                    'constraint_id': c.constraint_id,
                    'constraint': c.constraint,
                    'predicted_value': c.predicted_value,
                    'threshold': c.threshold,
                    'status': c.status,
                    'margin': c.margin
                }
                for c in self.constraints_checked
            ],
            'confidence': self.confidence,
            'status': self.status,
            'alternatives': [
                {
                    'id': alt.id,
                    'action': alt.action,
                    'score': alt.score,
                    'throughput': alt.throughput,
                    'energy_proxy': alt.energy_proxy,
                    'risk_index': alt.risk_index,
                    'delta_score': alt.delta_score,
                    'delta_throughput': alt.delta_throughput
                }
                for alt in self.alternatives
            ],
            'explanation': self.explanation,
            'metadata': self.metadata,
            'is_recommended': self.is_recommended()
        }

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """
        Конвертация в JSON string.

        Args:
            indent: отступ
            ensure_ascii: ASCII-only

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii, default=str)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_action_item(
        tag: str,
        name: str,
        from_value: float,
        to_value: float,
        unit: str
) -> ActionItem:
    """
    Создание ActionItem.

    Args:
        tag: тег параметра
        name: название
        from_value: текущее значение
        to_value: рекомендуемое значение
        unit: единица измерения

    Returns:
        ActionItem
    """
    delta = to_value - from_value
    delta_percent = (delta / abs(from_value) * 100) if abs(from_value) > 1e-6 else 0.0

    return ActionItem(
        tag=tag,
        name=name,
        from_value=from_value,
        to_value=to_value,
        unit=unit,
        delta=delta,
        delta_percent=delta_percent
    )


def create_constraint_check(
        constraint_id: str,
        constraint: str,
        predicted_value: float,
        threshold: float
) -> ConstraintCheck:
    """
    Создание ConstraintCheck.

    Args:
        constraint_id: ID ограничения
        constraint: описание
        predicted_value: прогнозируемое значение
        threshold: порог

    Returns:
        ConstraintCheck
    """
    status = "PASS" if predicted_value <= threshold else "FAIL"
    margin = threshold - predicted_value

    return ConstraintCheck(
        constraint_id=constraint_id,
        constraint=constraint,
        predicted_value=predicted_value,
        threshold=threshold,
        status=status,
        margin=margin
    )


def create_alternative(
        id: int,
        action: Dict[str, float],
        score: float,
        throughput: float,
        energy_proxy: float,
        risk_index: float,
        reference_score: float,
        reference_throughput: float
) -> Alternative:
    """
    Создание Alternative.

    Args:
        id: ID кандидата
        action: действие
        score: score
        throughput: throughput
        energy_proxy: energy
        risk_index: risk
        reference_score: score топ-1
        reference_throughput: throughput топ-1

    Returns:
        Alternative
    """
    delta_score = score - reference_score
    delta_throughput = throughput - reference_throughput

    return Alternative(
        id=id,
        action=action,
        score=score,
        throughput=throughput,
        energy_proxy=energy_proxy,
        risk_index=risk_index,
        delta_score=delta_score,
        delta_throughput=delta_throughput
    )


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == '__main__':
    # Пример создания рекомендации
    recommendation = Recommendation(
        recommendation_id=f"rec_{datetime.now():%Y%m%d_%H%M%S}",
        timestamp=datetime.now().isoformat(),
        state={
            'T6': 360.0,
            'F2_F26_ratio': 0.85,
            'F9': 215.0,
            'Sulfur_current': 9.2,
            'Sulfur_age_min': 45
        },
        problem_type="RISK_SPEC_VIOLATION",
        action=[
            create_action_item('T6', 'Температура реактора', 360.0, 363.0, '°C'),
            create_action_item('F2_F26_ratio', 'ВСГ/сырьё', 0.85, 0.875, '-')
        ],
        expected_effect=ExpectedEffect(
            sulfur_60min=7.5,
            sulfur_delta=-1.7,
            throughput=220.0,
            throughput_delta=5.0,
            energy_proxy=0.45,
            risk_index=0.20
        ),
        constraints_checked=[
            create_constraint_check('C001', 'Сера ≤ 10 мг/кг', 7.5, 10.0),
            create_constraint_check('C002', 'T6 в диапазоне [345, 375]', 363.0, 375.0)
        ],
        confidence=0.78,
        status="RECOMMENDED",
        alternatives=[
            create_alternative(
                id=312,
                action={'T6': 363.0, 'F2_F26_ratio': 0.87},
                score=0.4412,
                throughput=260.0,
                energy_proxy=0.46,
                risk_index=0.21,
                reference_score=0.4523,
                reference_throughput=262.5
            )
        ],
        explanation="Повышение T6 на 3°C снизит серу с 9.2 до 7.5 мг/кг. Выбран компромисс между качеством и производительностью."
    )

    print("\n" + "=" * 80)
    print("ПРИМЕР РЕКОМЕНДАЦИИ")
    print("=" * 80)
    print(recommendation.to_json(indent=2, ensure_ascii=True))
    print("=" * 80 + "\n")

    # Пример отказа
    no_recommendation = Recommendation(
        recommendation_id=f"rec_{datetime.now():%Y%m%d_%H%M%S}",
        timestamp=datetime.now().isoformat(),
        state={},
        problem_type="NO_DATA",
        action=[],
        expected_effect=ExpectedEffect(),
        constraints_checked=[],
        confidence=0.0,
        status="NO_RECOMMENDATION",
        alternatives=[],
        explanation="Надёжной рекомендации нет: последнее лабораторное значение устарело (age_min=360 мин), а доступные варианты либо нарушают ограничение по сере, либо выходят за заданный модельный диапазон."
    )

    print("\n" + "=" * 80)
    print("ПРИМЕР ОТКАЗА")
    print("=" * 80)
    print(no_recommendation.to_json(indent=2, ensure_ascii=True))
    print("=" * 80 + "\n")