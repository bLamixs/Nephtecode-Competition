"""
Интерфейсы и структуры данных для обмена сообщениями между агентами МАС.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class QualityAssessment:
    """Оценка качества продукта от Quality Agent."""
    timestamp: datetime
    sulfur_forecast_mg_kg: float
    d15_forecast_kg_m3: Optional[float] = None
    t50_forecast_c: Optional[float] = None
    t90_forecast_c: Optional[float] = None
    t95_forecast_c: Optional[float] = None
    cfpp_forecast_c: Optional[float] = None
    
    # Риски выхода за спецификацию
    risk_sulfur_violation: float = 0.0  # P(S > 10 мг/кг) [0..1]
    risk_overall_quality: float = 0.0   # [0..1]
    
    # Метаданные качества данных
    confidence: float = 1.0             # Уверенность модели [0..1]
    data_source: str = "HYBRID_VAC_ML"  # LIMS, PAK, VAC, HYBRID
    lims_age_hours: Optional[float] = None
    pak_age_minutes: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

@dataclass
class ReliabilityAssessment:
    """Оценка надёжности режима оборудования от Reliability Agent."""
    timestamp: datetime
    risk_index: float                   # Индекс риска [0..1]
    risk_class: str                     # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    is_safe: bool = True
    risk_factors: List[str] = field(default_factory=list)
    equipment_penalties: Dict[str, float] = field(default_factory=dict)
    active_constraints: List[str] = field(default_factory=list)

@dataclass
class CandidateAction:
    """Вариант управляющего воздействия."""
    action_id: str
    changes: Dict[str, Dict[str, float]] # tag: {"current": val, "target": val, "delta": val}
    expected_sulfur_effect: float = 0.0
    expected_throughput_change: float = 0.0
    expected_energy_proxy_change: float = 0.0
    equipment_risk_score: float = 0.0
    is_hard_valid: bool = True
    validation_notes: List[str] = field(default_factory=list)

@dataclass
class OptimizationResult:
    """Результат работы Optimization Agent."""
    timestamp: datetime
    is_solution_found: bool
    top_recommendation: Optional[CandidateAction] = None
    alternatives: List[CandidateAction] = field(default_factory=list)
    evaluated_candidates_count: int = 0
    valid_candidates_count: int = 0
    refusal_reason: Optional[str] = None

@dataclass
class Recommendation:
    """Итоговая рекомендация оператору от Orchestrator."""
    timestamp: datetime
    is_refusal: bool
    status: str                         # "NORMAL", "ACTION_RECOMMENDED", "REFUSAL", "ALERT"
    
    # Блоки объяснимости
    problem_detected: str               # Что обнаружено
    proposed_action: Optional[Dict[str, Any]] = None # Какие параметры изменить (было -> стало)
    expected_effect: Optional[Dict[str, Any]] = None # Ожидаемый эффект (качество, выпуск, энергия)
    verified_constraints: List[str] = field(default_factory=list) # Проверенные жёсткие ограничения
    confidence_score: float = 1.0       # Оценка уверенности
    explanation: str = ""               # Почему выбран этот вариант
    alternatives_summary: List[str] = field(default_factory=list)
    refusal_reason: Optional[str] = None
