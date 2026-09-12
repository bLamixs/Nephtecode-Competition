"""
Модуль: src/agents/interfaces.py
Назначение: Определение строго типизированных контрактов (dataclass) для межагентного обмена.

Архитектура взаимодействия в МАС:
В системе реализована схема взаимодействия через Оркестратор:
[Оркестратор] ---> Запрос состояния ---> [Агент Качества]
               <--- QualityAssessment ---
[Оркестратор] ---> Запрос надёжности ---> [Агент Надёжности]
               <--- ReliabilityAssessment ---
[Оркестратор] ---> Запрос вариантов  ---> [Агент Оптимизации]
               <--- OptimizationResult ---
[Оркестратор] ---> Veto / Ранжирование ---> Recommendation (Оператору)

Принцип безопасности:
Никакие поля оптимизации не могут переопределить жёсткие технологические ограничения.
Каждый объект содержит метаданные достоверности, возраста замеров и флаги валидности.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class QualityAssessment:
    """
    Контракт оценки качества продукта от Quality Agent (Агента качества).
    
    Содержит:
    - Прогнозы ключевых показателей качества (сера, плотность D15, фракционка T50, T90, T95, ПТФ).
    - Оценку риска P(S > 10 ppm) выхода за спецификацию ГОСТ / ТР ТС (Евро-5).
    - Метаданные доверия к данным: источник (LIMS / PAK / VAC / HYBRID) и возраст замеров.
    """
    timestamp: datetime
    sulfur_forecast_mg_kg: float                    # Прогноз содержания серы в товарном ДТ (жёсткий предел <= 10 мг/кг)
    d15_forecast_kg_m3: Optional[float] = None       # Прогноз плотности при 15°C (норма 820..845 кг/м3)
    t50_forecast_c: Optional[float] = None           # Прогноз температуры перегонки 50%
    t90_forecast_c: Optional[float] = None           # Прогноз температуры перегонки 90%
    t95_forecast_c: Optional[float] = None           # Прогноз температуры перегонки 95% (макс 360 °C)
    cfpp_forecast_c: Optional[float] = None          # Предельная температура фильтруемости (ПТФ)
    
    # Вероятностные оценки риска
    risk_sulfur_violation: float = 0.0               # Вероятность превышения серы > 10 мг/кг в диапазоне [0..1]
    risk_overall_quality: float = 0.0                # Интегральный риск нарушения любого показателя качества [0..1]
    
    # Метаданные качества входной информации
    confidence: float = 1.0                          # Степень уверенности модели в прогнозе [0..1]
    data_source: str = "HYBRID_VAC_ML"               # Приоритетный источник: "LIMS", "PAK", "VAC", "HYBRID"
    lims_age_hours: Optional[float] = None           # Время (в часах) с момента последнего отбора пробы ЛИМС
    pak_age_minutes: Optional[float] = None          # Время (в минутах) с последнего валидного показания ПАК
    warnings: List[str] = field(default_factory=list)# Предупреждения (например, "ЛИМС устарел > 48ч")


@dataclass
class ReliabilityAssessment:
    """
    Контракт оценки технологической надёжности от Reliability Agent (Агента надёжности).
    
    Содержит:
    - Индекс интегрального риска оборудования и катализатора [0..1].
    - Категорию тяжести режима (LOW, MEDIUM, HIGH, CRITICAL).
    - Перечень лимитирующих факторов и активных ограничений.
    """
    timestamp: datetime
    risk_index: float                                # Индекс тяжести режима (0 - оптимум, 1 - аварийный риск)
    risk_class: str                                  # Категория риска: "LOW", "MEDIUM", "HIGH", "CRITICAL"
    is_safe: bool = True                             # Флаг: допустим ли текущий режим для непрерывной работы
    risk_factors: List[str] = field(default_factory=list) # Расшифровка факторов (например: "Перепад давления на Р-201 высок")
    equipment_penalties: Dict[str, float] = field(default_factory=dict) # Штрафы по отдельным узлам (печи, реакторы, колонны)
    active_constraints: List[str] = field(default_factory=list)         # Активные коридоры безопасности КИП


@dataclass
class CandidateAction:
    """
    Контракт варианта управляющего воздействия, сформированного Optimization Agent.
    
    Описывает:
    - Конкретные изменения в тегах КИП (было -> стало -> дельта).
    - Ожидаемый эффект на серу, производительность (выпуск) и энергопотребление.
    - Флаг соблюдения жёстких технологических ограничений (Hard Validity).
    """
    action_id: str                                   # Уникальный идентификатор кандидата
    changes: Dict[str, Dict[str, float]]             # tag -> {"current": val, "target": val, "delta": val}
    expected_sulfur_effect: float = 0.0              # Ожидаемое изменение серы (мг/кг)
    expected_throughput_change: float = 0.0          # Ожидаемое изменение выпуска (т/ч)
    expected_energy_proxy_change: float = 0.0        # Прокси-изменение энергозатрат (топливо печей, перепад)
    equipment_risk_score: float = 0.0                # Прогнозируемый индекс риска оборудования
    is_hard_valid: bool = True                       # Прошло ли воздействие жёсткие фильтры (вето)
    validation_notes: List[str] = field(default_factory=list) # Примечания валидатора


@dataclass
class OptimizationResult:
    """
    Контракт результата генерации и селекции воздействий от Optimization Agent.
    
    Содержит:
    - Топ-1 рекомендуемое решение по совокупности критериев.
    - Допустимые Парето-альтернативы для оператора (2-3 варианта).
    - Статистику генерации и причину отказа, если ни одного допустимого сценария не найдено.
    """
    timestamp: datetime
    is_solution_found: bool                          # Найдено ли хотя бы одно допустимое решение
    top_recommendation: Optional[CandidateAction] = None # Лучший вариант
    alternatives: List[CandidateAction] = field(default_factory=list) # Альтернативные Парето-варианты
    evaluated_candidates_count: int = 0              # Всего сгенерировано кандидатов
    valid_candidates_count: int = 0                  # Число кандидатов, прошедших жёсткие ограничения
    refusal_reason: Optional[str] = None             # Причина невозможности оптимизации


@dataclass
class Recommendation:
    """
    Итоговая рекомендация оператору технологической установки от Orchestrator.
    
    Реализует принцип полной объяснимости (Explainable AI):
    [Время и состояние] -> [Проблема/Риск] -> [Действие] -> [Ожидаемый эффект] ->
    [Проверка ограничений] -> [Уверенность] -> [Объяснение и альтернативы].
    
    Если безопасного решения нет или данные недостоверны — формирует аргументированный отказ.
    """
    timestamp: datetime
    is_refusal: bool                                 # True, если система отказывается выдавать управляющее действие
    status: str                                      # "NORMAL", "ACTION_RECOMMENDED", "REFUSAL", "ALERT"
    
    # 1. Диагностика ситуации
    problem_detected: str                            # Что обнаружено (риск серы, норма, сбой датчиков)
    
    # 2. Рекомендуемое управление (только если не отказ)
    proposed_action: Optional[Dict[str, Any]] = None # Изменяемые параметры (тег, было, стало, направление)
    
    # 3. Прогноз последствий
    expected_effect: Optional[Dict[str, Any]] = None # Ожидаемый эффект на качество, выпуск, затраты
    
    # 4. Аудит ограничений и доверие
    verified_constraints: List[str] = field(default_factory=list) # Список подтвержденных ограничений
    confidence_score: float = 1.0                    # Уровень доверия к выданному совету [0..1]
    
    # 5. Обоснование выбора
    explanation: str = ""                            # Человекопонятный текст: почему именно это действие
    alternatives_summary: List[str] = field(default_factory=list) # Краткое описание альтернативных вариантов
    refusal_reason: Optional[str] = None             # Подробная причина отказа при is_refusal=True
