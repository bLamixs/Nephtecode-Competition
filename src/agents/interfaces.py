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
    timestamp: Any
    sulfur_forecast_mg_kg: float = 8.5              # Прогноз содержания серы в товарном ДТ (жёсткий предел <= 10 мг/кг)
    predictions: Dict[str, float] = field(default_factory=dict) # Словарь всех прогнозов {'Sulfur': ..., 'D15': ...}
    d15_forecast_kg_m3: Optional[float] = None       # Прогноз плотности при 15°C (норма 820..845 кг/м3)
    t50_forecast_c: Optional[float] = None           # Прогноз температуры перегонки 50%
    t90_forecast_c: Optional[float] = None           # Прогноз температуры перегонки 90%
    t95_forecast_c: Optional[float] = None           # Прогноз температуры перегонки 95% (макс 360 °C)
    cfpp_forecast_c: Optional[float] = None          # Предельная температура фильтруемости (ПТФ)
    flash_forecast_c: Optional[float] = None         # Температура вспышки
    
    # Вероятностные оценки риска
    risk_spec_violation: Dict[str, float] = field(default_factory=dict) # P(S>10), P(T95>spec)
    risk_sulfur_violation: float = 0.0               # Вероятность превышения серы > 10 мг/кг в диапазоне [0..1]
    risk_overall_quality: float = 0.0                # Интегральный риск нарушения любого показателя качества [0..1]
    
    # Метаданные качества входной информации
    confidence: float = 1.0                          # Степень уверенности модели в прогнозе [0..1]
    data_source: str = "HYBRID_VAC_ML"               # Приоритетный источник: "LIMS", "PAK", "VAC", "HYBRID"
    age_min: int = 0                                 # Возраст анализа в минутах
    lims_age_hours: Optional[float] = None           # Время (в часах) с момента последнего отбора пробы ЛИМС
    pak_age_minutes: Optional[float] = None          # Время (в минутах) с последнего валидного показания ПАК
    warnings: List[str] = field(default_factory=list)# Предупреждения (например, "ЛИМС устарел > 48ч")

    def __post_init__(self):
        # Синхронизация predictions и отдельных полей
        if not self.predictions:
            self.predictions = {
                'Sulfur': self.sulfur_forecast_mg_kg,
                'D15': self.d15_forecast_kg_m3 if self.d15_forecast_kg_m3 is not None else 835.0,
                'T50': self.t50_forecast_c if self.t50_forecast_c is not None else 280.0,
                'T95': self.t95_forecast_c if self.t95_forecast_c is not None else 350.0,
                'CFPP': self.cfpp_forecast_c if self.cfpp_forecast_c is not None else -10.0,
            }
            if self.t90_forecast_c is not None:
                self.predictions['T90'] = self.t90_forecast_c
            if self.flash_forecast_c is not None:
                self.predictions['flash'] = self.flash_forecast_c
        elif self.sulfur_forecast_mg_kg == 8.5 and 'Sulfur' in self.predictions:
            self.sulfur_forecast_mg_kg = float(self.predictions['Sulfur'])

        if not self.risk_spec_violation:
            self.risk_spec_violation = {
                'P_S_gt_10': self.risk_sulfur_violation,
                'P_T95_gt_spec': 0.05 if (self.t95_forecast_c or 350.0) <= 360.0 else 0.85
            }
        elif 'P_S_gt_10' in self.risk_spec_violation and self.risk_sulfur_violation == 0.0:
            self.risk_sulfur_violation = float(self.risk_spec_violation['P_S_gt_10'])

        if self.lims_age_hours is None and self.age_min > 0:
            self.lims_age_hours = self.age_min / 60.0

    @property
    def predicted_sulfur(self) -> float:
        return self.sulfur_forecast_mg_kg

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class ReliabilityAssessment:
    """
    Контракт оценки технологической надёжности от Reliability Agent (Агента надёжности).
    
    Содержит:
    - Индекс интегрального риска оборудования и катализатора [0..1].
    - Категорию тяжести режима (low, medium, high / LOW, MEDIUM, HIGH).
    - Перечень лимитирующих факторов и активных ограничений для оптимизатора.
    """
    timestamp: Any
    risk_index: float                                # Индекс тяжести режима (0 - оптимум, 1 - аварийный риск)
    risk_class: str                                  # Категория риска: "low", "medium", "high"
    is_safe: bool = True                             # Флаг: допустим ли текущий режим для непрерывной работы
    risk_factors: List[Any] = field(default_factory=list) # Топ-факторы риска [{"tag": "T6", "deviation": 1.8, "weight": 0.3}, ...]
    constraints_for_optimizer: List[Dict[str, Any]] = field(default_factory=list) # [{"tag": "T6", "min": 290, "max": 310}, ...]
    confidence: float = 1.0                          # Степень уверенности
    assumptions: List[str] = field(default_factory=list) # Допущения и гипотезы
    equipment_penalties: Dict[str, float] = field(default_factory=dict) # Штрафы по отдельным узлам
    active_constraints: List[str] = field(default_factory=list)         # Активные коридоры безопасности КИП

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


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
    Контракт результата генерации и селекции воздействий от Optimization Agent (OPT-05).
    
    Содержит:
    - Топ-1 рекомендуемое решение (recommended).
    - Допустимые Парето-альтернативы (alternatives).
    - Списки кандидатов (candidates, feasible, ranked).
    - Метрики оптимизации (throughput, energy, risk, best_score...).
    """
    timestamp: Any
    candidates: List[Any] = field(default_factory=list)
    feasible: List[Any] = field(default_factory=list)
    ranked: List[Any] = field(default_factory=list)
    recommended: Optional[Any] = None
    alternatives: List[Any] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Поля для обратной совместимости
    is_solution_found: bool = True
    top_recommendation: Optional[Any] = None
    evaluated_candidates_count: int = 0
    valid_candidates_count: int = 0
    refusal_reason: Optional[str] = None

    def __post_init__(self):
        if self.top_recommendation is None and self.recommended is not None:
            self.top_recommendation = self.recommended
        elif self.recommended is None and self.top_recommendation is not None:
            self.recommended = self.top_recommendation

        if not self.evaluated_candidates_count and self.candidates:
            self.evaluated_candidates_count = len(self.candidates)
        if not self.valid_candidates_count and self.feasible:
            self.valid_candidates_count = len(self.feasible)

    def __iter__(self):
        """Поддержка распаковки: recommended, alternatives = opt_result"""
        return iter((self.recommended, self.alternatives))


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
