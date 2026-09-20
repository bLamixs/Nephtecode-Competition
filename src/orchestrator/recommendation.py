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


class ActionList(list):
    """
    Список действий с поддержкой доступа по тегу как в dict,
    а также конвертации в словарь.
    """
    def __getitem__(self, key):
        if isinstance(key, str):
            for item in self:
                item_tag = getattr(item, 'tag', None)
                if item_tag is None and isinstance(item, dict):
                    item_tag = item.get('tag')
                if item_tag == key:
                    return item
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for item in self:
            tag = getattr(item, 'tag', None) or (item.get('tag') if isinstance(item, dict) else None)
            if tag:
                if hasattr(item, 'to_dict'):
                    result[tag] = item.to_dict()
                elif isinstance(item, dict):
                    result[tag] = item
                else:
                    result[tag] = {
                        'from': getattr(item, 'from_value', 0.0),
                        'to': getattr(item, 'to_value', 0.0),
                        'unit': getattr(item, 'unit', '')
                    }
        return result

    def items(self):
        return self.to_dict().items()

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tag': self.tag,
            'name': self.name,
            'from': self.from_value,
            'to': self.to_value,
            'unit': self.unit,
            'delta': self.delta,
            'delta_percent': self.delta_percent
        }

    def __getitem__(self, key: str) -> Any:
        k = str(key).lower()
        if k in ('tag',): return self.tag
        if k in ('name',): return self.name
        if k in ('from', 'from_value'): return self.from_value
        if k in ('to', 'to_value'): return self.to_value
        if k in ('unit',): return self.unit
        if k in ('delta',): return self.delta
        if k in ('delta_percent', 'deltapercent'): return self.delta_percent
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'Sulfur_60min': self.sulfur_60min,
            'Sulfur_delta': self.sulfur_delta,
            'D15_60min': self.d15_60min,
            'T95_60min': self.t95_60min,
            'CFPP_60min': self.cfpp_60min,
            'throughput': self.throughput,
            'throughput_delta': self.throughput_delta,
            'throughput_change': self.throughput_delta,
            'energy_proxy': self.energy_proxy,
            'energy_delta': self.energy_delta,
            'risk_index': self.risk_index,
            'risk_delta': self.risk_delta
        }

    def __getitem__(self, key: str) -> Any:
        k = str(key).lower()
        field_map = {
            'sulfur_60min': self.sulfur_60min,
            'sulfur60min': self.sulfur_60min,
            'sulfur_delta': self.sulfur_delta,
            'd15_60min': self.d15_60min,
            't95_60min': self.t95_60min,
            'cfpp_60min': self.cfpp_60min,
            'throughput': self.throughput,
            'throughput_delta': self.throughput_delta,
            'throughput_change': self.throughput_delta,
            'energy_proxy': self.energy_proxy,
            'energy_delta': self.energy_delta,
            'risk_index': self.risk_index,
            'risk_delta': self.risk_delta
        }
        if k in field_map:
            return field_map[k]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            val = self[key]
            return val if val is not None else default
        except KeyError:
            return default

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'constraint_id': self.constraint_id,
            'constraint': self.constraint,
            'predicted_value': self.predicted_value,
            'threshold': self.threshold,
            'status': self.status,
            'margin': self.margin
        }

    def __getitem__(self, key: str) -> Any:
        k = str(key).lower()
        if k in ('constraint_id', 'id'): return self.constraint_id
        if k in ('constraint', 'name'): return self.constraint
        if k in ('predicted_value', 'value'): return self.predicted_value
        if k in ('threshold', 'limit'): return self.threshold
        if k in ('status',): return self.status
        if k in ('margin',): return self.margin
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'action': self.action,
            'score': self.score,
            'throughput': self.throughput,
            'energy_proxy': self.energy_proxy,
            'risk_index': self.risk_index,
            'delta_score': self.delta_score,
            'delta_throughput': self.delta_throughput
        }

    def __getitem__(self, key: str) -> Any:
        k = str(key).lower()
        if k in ('id',): return self.id
        if k in ('action', 'params'): return self.action
        if k in ('score',): return self.score
        if k in ('throughput',): return self.throughput
        if k in ('energy_proxy', 'energy'): return self.energy_proxy
        if k in ('risk_index', 'risk'): return self.risk_index
        if k in ('delta_score',): return self.delta_score
        if k in ('delta_throughput',): return self.delta_throughput
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __str__(self) -> str:
        if isinstance(self.action, dict):
            act_str = ", ".join(f"{k}: {v}" for k, v in self.action.items())
        else:
            act_str = str(self.action)
        return f"{act_str} (Score: {self.score:.4f}, Выпуск: {self.throughput:.1f} м³/ч, Риск: {self.risk_index:.2f})"


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

    # Действие: Dict[str, Any] или List[ActionItem]
    action: Any

    # Ожидаемый эффект: Dict[str, Any] или ExpectedEffect
    expected_effect: Any

    # Проверенные ограничения: List[Dict[str, Any]] или List[ConstraintCheck]
    constraints_checked: List[Any]

    # Уверенность
    confidence: float

    # Статус: "RECOMMENDED" или "NO_RECOMMENDATION"
    status: str

    # Альтернативы: List[Dict[str, Any]] или List[Alternative]
    alternatives: List[Any]

    # Объяснение
    explanation: str

    # Дополнительные метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Нормализация action
        if isinstance(self.action, dict):
            if not self.action:
                self.action = ActionList()
            else:
                al = ActionList()
                for tag, val in self.action.items():
                    if isinstance(val, dict):
                        from_val = float(val.get('from', val.get('from_value', 0.0)))
                        to_val = float(val.get('to', val.get('to_value', 0.0)))
                        unit = str(val.get('unit', '°C' if 'T' in tag else ('м³/ч' if 'F9' in tag else '-')))
                        name = str(val.get('name', tag))
                        al.append(create_action_item(tag, name, from_val, to_val, unit))
                    elif isinstance(val, (int, float)):
                        al.append(create_action_item(tag, tag, float(val), float(val), ''))
                self.action = al
        elif isinstance(self.action, list) and not isinstance(self.action, ActionList):
            self.action = ActionList(self.action)

        # Нормализация expected_effect
        if isinstance(self.expected_effect, dict) and not isinstance(self.expected_effect, ExpectedEffect):
            ee_dict = self.expected_effect
            self.expected_effect = ExpectedEffect(
                sulfur_60min=ee_dict.get('Sulfur_60min', ee_dict.get('sulfur_60min')),
                sulfur_delta=ee_dict.get('Sulfur_delta', ee_dict.get('sulfur_delta')),
                d15_60min=ee_dict.get('D15_60min', ee_dict.get('d15_60min')),
                t95_60min=ee_dict.get('T95_60min', ee_dict.get('t95_60min')),
                cfpp_60min=ee_dict.get('CFPP_60min', ee_dict.get('cfpp_60min')),
                throughput=ee_dict.get('throughput', ee_dict.get('F9')),
                throughput_delta=ee_dict.get('throughput_delta', ee_dict.get('throughput_change')),
                energy_proxy=ee_dict.get('energy_proxy'),
                energy_delta=ee_dict.get('energy_delta'),
                risk_index=ee_dict.get('risk_index'),
                risk_delta=ee_dict.get('risk_delta')
            )
        elif self.expected_effect is None:
            self.expected_effect = ExpectedEffect()

        # Нормализация constraints_checked
        if isinstance(self.constraints_checked, list):
            norm_constraints = []
            for c in self.constraints_checked:
                if isinstance(c, dict):
                    norm_constraints.append(
                        ConstraintCheck(
                            constraint_id=c.get('constraint_id', f"C_{len(norm_constraints)+1:03d}"),
                            constraint=c.get('constraint', c.get('name', 'Ограничение')),
                            predicted_value=float(c.get('predicted_value', c.get('value', 0.0))),
                            threshold=float(c.get('threshold', 0.0)),
                            status=str(c.get('status', 'PASS')),
                            margin=float(c.get('margin', 0.0))
                        )
                    )
                else:
                    norm_constraints.append(c)
            self.constraints_checked = norm_constraints

        # Нормализация alternatives
        if isinstance(self.alternatives, list):
            norm_alts = []
            for alt in self.alternatives:
                if isinstance(alt, dict):
                    norm_alts.append(
                        Alternative(
                            id=int(alt.get('id', len(norm_alts)+1)),
                            action=alt.get('action', alt.get('params', {})),
                            score=float(alt.get('score', 0.0)),
                            throughput=float(alt.get('throughput', 0.0)),
                            energy_proxy=float(alt.get('energy_proxy', 0.0)),
                            risk_index=float(alt.get('risk_index', 0.0)),
                            delta_score=float(alt.get('delta_score', 0.0)),
                            delta_throughput=float(alt.get('delta_throughput', 0.0))
                        )
                    )
                else:
                    norm_alts.append(alt)
            self.alternatives = norm_alts

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
        actions_serialized = [
            a.to_dict() if hasattr(a, 'to_dict') else a
            for a in self.action
        ] if isinstance(self.action, list) else (
            self.action.to_dict() if hasattr(self.action, 'to_dict') else self.action
        )

        ee_serialized = self.expected_effect.to_dict() if hasattr(self.expected_effect, 'to_dict') else (
            self.expected_effect if isinstance(self.expected_effect, dict) else {}
        )

        constraints_serialized = [
            c.to_dict() if hasattr(c, 'to_dict') else c
            for c in self.constraints_checked
        ]

        alts_serialized = [
            alt.to_dict() if hasattr(alt, 'to_dict') else alt
            for alt in self.alternatives
        ]

        return {
            'recommendation_id': self.recommendation_id,
            'timestamp': self.timestamp,
            'state': self.state,
            'problem_type': self.problem_type,
            'action': actions_serialized,
            'expected_effect': ee_serialized,
            'constraints_checked': constraints_serialized,
            'confidence': self.confidence,
            'status': self.status,
            'alternatives': alts_serialized,
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