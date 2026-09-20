"""
Агент оптимизации: генерация и сравнение вариантов управления.

Методы:
- generate_candidates: генерация вариантов (grid/Sobol/Dirichlet)
- apply_veto: отбраковка по жёстким ограничениям (OPT-03)
- score_candidates: оценка вариантов (throughput, energy, risk) — OPT-04
- rank_pareto: выбор топ-1 + альтернативы — OPT-05
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc
from itertools import product
import logging

logger = logging.getLogger(__name__)


@dataclass
class ControlledParam:
    """Описание управляемого параметра."""
    current: float
    min: float
    max: float
    unit: str
    description: str
    step: Optional[float] = None


@dataclass
class Candidate:
    """Кандидат на изменение режима."""
    params: Dict[str, float]
    blending: Dict[str, float]
    source: str
    id: int


@dataclass
class ScoredCandidate:
    """Кандидат с оценками."""
    candidate: Candidate
    throughput: float
    throughput_normalized: float
    energy_proxy: float
    energy_normalized: float
    risk_index: float
    risk_normalized: float
    score: float
    score_breakdown: Dict[str, float]


@dataclass
class OptimizationResult:
    """
    Результат оптимизации (OPT-05).

    Атрибуты:
    - timestamp: время оптимизации
    - candidates: все сгенерированные кандидаты
    - feasible: допустимые кандидаты (после veto)
    - ranked: оценённые и отсортированные кандидаты
    - recommended: топ-1 рекомендация
    - alternatives: 2-3 альтернативы
    - metrics: метрики оптимизации (throughput, energy, risk, best_score...)
    """
    timestamp: pd.Timestamp
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

    def __iter__(self):
        """Поддержка распаковки: recommended, alternatives = opt_result"""
        return iter((self.recommended, self.alternatives))


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

CONTROLLED_PARAMS: Dict[str, ControlledParam] = {
    'T6': ControlledParam(current=360.0, min=345.0, max=375.0, unit='°C', description='Температура реактора', step=1.0),
    'F2_F26_ratio': ControlledParam(current=0.85, min=0.80, max=0.95, unit='-', description='ВСГ/сырьё', step=0.025),
    'T55': ControlledParam(current=380.0, min=375.0, max=386.0, unit='°C', description='Температура печи', step=1.0),
    'F9': ControlledParam(current=215.0, min=160.0, max=280.0, unit='т/ч', description='Расход на гидроочистку', step=5.0),
}

BLENDING_FRACTIONS: Dict[str, ControlledParam] = {
    'F30': ControlledParam(current=0.23, min=0.18, max=0.28, unit='доля', description='Фракция 290-350°C'),
    'F32': ControlledParam(current=0.14, min=0.10, max=0.20, unit='доля', description='Фракция 240-290°C'),
    'F34': ControlledParam(current=0.15, min=0.10, max=0.20, unit='доля', description='Фракция 150-250°C'),
    'F56': ControlledParam(current=0.05, min=0.02, max=0.09, unit='доля', description='Лёгкий компонент'),
    'F57': ControlledParam(current=0.06, min=0.03, max=0.10, unit='доля', description='Тяжёлый компонент'),
    'F59': ControlledParam(current=0.36, min=0.30, max=0.42, unit='доля', description='Фракция 420-500°C'),
}


class OptimizationAgent:
    """
    Агент оптимизации: генерация и сравнение вариантов управления.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.controlled_params = CONTROLLED_PARAMS
        self.blending_fractions = BLENDING_FRACTIONS

        self.weights = {
            'throughput': 0.5,
            'energy': 0.3,
            'risk': 0.2
        }

        self.baseline = {
            'throughput': 215.0,
            'energy': 1.0,
            'risk': 1.0
        }

        self._candidate_id = 0

    def _next_id(self) -> int:
        self._candidate_id += 1
        return self._candidate_id

    # ========================================================================
    # ГЕНЕРАЦИЯ (grid, Sobol, Dirichlet)
    # ========================================================================

    def _grid_search(self, param_names: Optional[List[str]] = None) -> List[Candidate]:
        """Grid search для 2-3 параметров."""
        if param_names is None:
            param_names = list(self.controlled_params.keys())[:3]

        grids = {}
        for name in param_names:
            param = self.controlled_params[name]
            if param.step:
                values = np.arange(param.min, param.max + param.step, param.step)
            else:
                values = np.linspace(param.min, param.max, 5)
            values = np.round(values, 3)
            grids[name] = values

        candidates = []
        for values in product(*grids.values()):
            candidate = {name: float(val) for name, val in zip(grids.keys(), values)}
            candidates.append(Candidate(params=candidate, blending={}, source='grid', id=self._next_id()))

        return candidates

    def _sobol_sampling(self, num_candidates: int = 50, param_names: Optional[List[str]] = None) -> List[Candidate]:
        """Sobol sequence для >3 параметров."""
        if param_names is None:
            param_names = list(self.controlled_params.keys())

        d = len(param_names)
        m = int(np.ceil(np.log2(num_candidates)))

        sampler = qmc.Sobol(d=d, scramble=True, seed=42)
        samples = sampler.random_base2(m=m)

        candidates = []
        for i, sample in enumerate(samples):
            if i >= num_candidates:
                break

            candidate = {}
            for j, name in enumerate(param_names):
                param = self.controlled_params[name]
                value = param.min + sample[j] * (param.max - param.min)
                candidate[name] = float(np.round(value, 3))

            candidates.append(Candidate(params=candidate, blending={}, source='sobol', id=self._next_id()))

        return candidates

    def _generate_blending_candidates(self, num_candidates: int = 20) -> List[Candidate]:
        """Dirichlet для блендинга (сумма = 1.0) с концентрацией вокруг базового состава."""
        fraction_names = list(self.blending_fractions.keys())
        candidates = []
        
        # Концентрация вокруг текущего технологического рецепта
        base_weights = np.array([self.blending_fractions[name].current for name in fraction_names])
        alpha = base_weights * 60.0

        attempts = 0
        max_attempts = num_candidates * 50

        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1
            fractions = np.random.dirichlet(alpha)

            valid = True
            candidate = {}

            for i, name in enumerate(fraction_names):
                param = self.blending_fractions[name]
                value = fractions[i]
                if value < param.min or value > param.max:
                    valid = False
                    break
                candidate[name] = float(np.round(value, 4))

            if valid:
                total = sum(candidate.values())
                candidate = {k: float(np.round(v / total, 4)) for k, v in candidate.items()}
                diff = 1.0 - sum(candidate.values())
                max_k = max(candidate.keys(), key=lambda k: candidate[k])
                candidate[max_k] = float(np.round(candidate[max_k] + diff, 4))
                candidates.append(Candidate(params={}, blending=candidate, source='dirichlet', id=self._next_id()))

        # Гарантированное дополнение базовым рецептом, если сэмплов недостаточно
        while len(candidates) < num_candidates:
            base_recipe = {k: float(self.blending_fractions[k].current) for k in fraction_names}
            candidates.append(Candidate(params={}, blending=base_recipe, source='baseline', id=self._next_id()))

        return candidates

    def generate_candidates(
        self,
        num_mode_candidates: int = 50,
        num_blending_candidates: int = 20,
        use_grid: bool = True
    ) -> List[Candidate]:
        """Основной метод генерации."""
        num_params = len(self.controlled_params)

        if use_grid and num_params <= 3:
            mode_candidates = self._grid_search()
        else:
            mode_candidates = self._sobol_sampling(num_candidates=num_mode_candidates)

        blending_candidates = self._generate_blending_candidates(num_candidates=num_blending_candidates)

        # Комбинирование
        all_candidates = []
        for mode in mode_candidates:
            for blend in blending_candidates:
                all_candidates.append(Candidate(
                    params={**mode.params},
                    blending={**blend.blending},
                    source=f"{mode.source}+{blend.source}",
                    id=self._next_id()
                ))

        if not blending_candidates:
            all_candidates = mode_candidates

        return all_candidates

    # ========================================================================
    # VETO (OPT-03)
    # ========================================================================

    def apply_veto(
        self,
        candidates: List[Candidate],
        quality_assessment: Optional[Dict[str, Any]] = None,
        reliability_assessment: Optional[Dict[str, Any]] = None
    ) -> List[Candidate]:
        """Veto по жёстким ограничениям."""
        logger.info(f"Veto: {len(candidates)} кандидатов до отбраковки")

        feasible = []

        for candidate in candidates:
            veto_reasons = []

            # 1. Сера ≤ 10
            sulfur_veto, _ = self._check_sulfur_veto(candidate, quality_assessment)
            if sulfur_veto:
                veto_reasons.append('sulfur')

            # 2. Доли = 100%
            blending_veto, _ = self._check_blending_sum_veto(candidate)
            if blending_veto:
                veto_reasons.append('blending_sum')

            # 3. Диапазоны параметров
            param_veto, _ = self._check_param_ranges_veto(candidate)
            if param_veto:
                veto_reasons.append('param_range')

            # 4. Диапазоны долей
            blending_range_veto, _ = self._check_blending_ranges_veto(candidate)
            if blending_range_veto:
                veto_reasons.append('blending_range')

            # 5. Надёжность
            reliability_veto, _ = self._check_reliability_veto(candidate, reliability_assessment)
            if reliability_veto:
                veto_reasons.append('reliability')

            if not veto_reasons:
                feasible.append(candidate)

        logger.info(f"Veto: {len(feasible)}/{len(candidates)} кандидатов допустимы")
        return feasible

    def _check_sulfur_veto(self, candidate: Candidate, quality_assessment: Optional[Any]) -> Tuple[bool, Optional[str]]:
        if quality_assessment is not None:
            sulfur_forecast = getattr(quality_assessment, 'predicted_sulfur', None)
            if sulfur_forecast is None and isinstance(quality_assessment, dict):
                sulfur_forecast = quality_assessment.get('predictions', {}).get('Sulfur', 8.5)

            sulfur_risk = getattr(quality_assessment, 'risk_spec_violation', None)
            if isinstance(sulfur_risk, dict):
                sulfur_risk = sulfur_risk.get('P_S_gt_10', 0.0)
            elif sulfur_risk is None and isinstance(quality_assessment, dict):
                sulfur_risk = quality_assessment.get('risk_spec_violation', {}).get('P_S_gt_10', 0.0)
            elif not isinstance(sulfur_risk, (int, float)):
                sulfur_risk = 0.0

            if sulfur_risk > 0.1:
                return True, f"P(S>10)={sulfur_risk:.3f}"
            if sulfur_forecast is not None and sulfur_forecast > 10.0:
                return True, f"Сера={sulfur_forecast:.2f}"

            return False, None

        t6 = candidate.params.get('T6', 360.0)
        # При T6=360 (норма): сера ~8.5 <= 10. При T6=355 (холодный реактор): сера ~11.5 > 10
        sulfur_estimate = 8.5 - 0.6 * (t6 - 360.0)

        if sulfur_estimate > 10.0:
            return True, f"Оценка серы={sulfur_estimate:.2f}"

        return False, None

    def _check_blending_sum_veto(self, candidate: Candidate) -> Tuple[bool, Optional[str]]:
        if not candidate.blending:
            return False, None

        blend_sum = sum(candidate.blending.values())

        if not np.isclose(blend_sum, 1.0, atol=0.01):
            return True, f"Сумма={blend_sum:.4f}"

        return False, None

    def _check_param_ranges_veto(self, candidate: Candidate) -> Tuple[bool, Optional[str]]:
        for tag, value in candidate.params.items():
            if tag not in self.controlled_params:
                continue

            param = self.controlled_params[tag]

            if value < param.min or value > param.max:
                return True, f"{tag}={value} вне [{param.min}, {param.max}]"

        return False, None

    def _check_blending_ranges_veto(self, candidate: Candidate) -> Tuple[bool, Optional[str]]:
        if not candidate.blending:
            return False, None

        for tag, value in candidate.blending.items():
            if tag not in self.blending_fractions:
                continue

            param = self.blending_fractions[tag]

            if value < param.min or value > param.max:
                return True, f"{tag}={value} вне [{param.min}, {param.max}]"

        return False, None

    def _check_reliability_veto(self, candidate: Candidate, reliability_assessment: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        if not reliability_assessment:
            return False, None

        risk_class = reliability_assessment.get('risk_class', 'low')

        if risk_class == 'high':
            return True, f"Риск={risk_class}"

        return False, None

    # ========================================================================
    # ОЦЕНКА ВАРИАНТОВ (OPT-04)
    # ========================================================================

    def score_candidates(
        self,
        feasible: List[Candidate],
        current_state: Optional[Dict[str, float]] = None
    ) -> List[ScoredCandidate]:
        """Оценка вариантов по целевой функции."""
        logger.info(f"Оценка: {len(feasible)} допустимых вариантов")

        if current_state is None:
            current_state = {}

        scored = []

        for candidate in feasible:
            # Throughput
            throughput = self._estimate_throughput(candidate, current_state)
            throughput_normalized = throughput / self.baseline['throughput']

            # Energy proxy
            energy_proxy = self._estimate_energy_proxy(candidate, current_state)
            energy_normalized = min(1.0, max(0.0, energy_proxy))

            # Risk index
            risk_index = self._estimate_risk_index(candidate, current_state)
            risk_normalized = min(1.0, max(0.0, risk_index))

            # Score
            score = (
                self.weights['throughput'] * throughput_normalized
                - self.weights['energy'] * energy_normalized
                - self.weights['risk'] * risk_normalized
            )

            scored_candidate = ScoredCandidate(
                candidate=candidate,
                throughput=throughput,
                throughput_normalized=throughput_normalized,
                energy_proxy=energy_proxy,
                energy_normalized=energy_normalized,
                risk_index=risk_index,
                risk_normalized=risk_normalized,
                score=score,
                score_breakdown={
                    'throughput_component': self.weights['throughput'] * throughput_normalized,
                    'energy_component': -self.weights['energy'] * energy_normalized,
                    'risk_component': -self.weights['risk'] * risk_normalized
                }
            )

            scored.append(scored_candidate)

        scored.sort(key=lambda x: x.score, reverse=True)

        if scored:
            logger.info(f"Оценка: лучший score={scored[0].score:.4f}, худший score={scored[-1].score:.4f}")
        else:
            logger.info("Оценка: нет вариантов для ранжирования")

        return scored

    def _estimate_throughput(self, candidate: Candidate, current_state: Dict[str, float]) -> float:
        """Оценка throughput (т/ч)."""
        f9 = candidate.params.get('F9', current_state.get('F9', self.baseline['throughput']))

        blend_factor = 1.0
        if candidate.blending:
            light_fractions = candidate.blending.get('F30', 0.3) + candidate.blending.get('F32', 0.25)
            heavy_fractions = candidate.blending.get('F34', 0.2) + candidate.blending.get('F57', 0.1)
            blend_factor = 1.0 + 0.1 * (light_fractions - 0.55) - 0.1 * (heavy_fractions - 0.3)

        throughput = f9 * blend_factor

        return throughput

    def _estimate_energy_proxy(self, candidate: Candidate, current_state: Dict[str, float]) -> float:
        """Оценка энергозатрат (прокси, 0..1)."""
        t6 = candidate.params.get('T6', current_state.get('T6', 360.0))
        t6_min = self.controlled_params['T6'].min
        t6_range = self.controlled_params['T6'].max - t6_min
        t6_normalized = (t6 - t6_min) / t6_range if t6_range > 0 else 0.5
        t6_normalized = min(1.0, max(0.0, t6_normalized))

        t55 = candidate.params.get('T55', current_state.get('T55', 380.0))
        t55_min = self.controlled_params['T55'].min
        t55_range = self.controlled_params['T55'].max - t55_min
        t55_normalized = (t55 - t55_min) / t55_range if t55_range > 0 else 0.5
        t55_normalized = min(1.0, max(0.0, t55_normalized))

        f2_ratio = candidate.params.get('F2_F26_ratio', current_state.get('F2_F26_ratio', 0.85))
        f2_normalized = (f2_ratio - 0.80) / 0.15
        f2_normalized = min(1.0, max(0.0, f2_normalized))

        energy_proxy = 0.5 * t6_normalized + 0.3 * t55_normalized + 0.2 * f2_normalized

        return energy_proxy

    def _estimate_risk_index(self, candidate: Candidate, current_state: Dict[str, float]) -> float:
        """Оценка риска оборудования (0..1)."""
        risks = []

        # T6: отклонение от нормы
        t6 = candidate.params.get('T6', current_state.get('T6', 360.0))
        t6_norm = self.controlled_params['T6'].current
        t6_max = self.controlled_params['T6'].max
        t6_deviation = abs(t6 - t6_norm) / (t6_max - t6_norm) if (t6_max - t6_norm) > 0 else 0.0
        risks.append(t6_deviation)

        # T55: отклонение от нормы
        t55 = candidate.params.get('T55', current_state.get('T55', 380.0))
        t55_norm = self.controlled_params['T55'].current
        t55_max = self.controlled_params['T55'].max
        t55_deviation = abs(t55 - t55_norm) / (t55_max - t55_norm) if (t55_max - t55_norm) > 0 else 0.0
        risks.append(t55_deviation)

        # F2_F26_ratio: отклонение от нормы
        f2_ratio = candidate.params.get('F2_F26_ratio', current_state.get('F2_F26_ratio', 0.85))
        f2_norm = self.controlled_params['F2_F26_ratio'].current
        f2_range = self.controlled_params['F2_F26_ratio'].max - self.controlled_params['F2_F26_ratio'].min
        f2_deviation = abs(f2_ratio - f2_norm) / f2_range
        risks.append(f2_deviation)

        risk_index = np.mean(risks)

        return risk_index

    # ========================================================================
    # РАНЖИРОВАНИЕ (OPT-05): Pareto-фронт, топ-1 + альтернативы
    # ========================================================================

    def _rank_pareto(
        self,
        scored: List[Any],
        num_alternatives: int = 3,
        min_diversity: float = 0.1
    ) -> OptimizationResult:
        """
        Выбор топ-1 + альтернативы через Pareto-фронт (OPT-05).

        Логика:
        1. Сортировка по score (убывание).
        2. Топ-1 = лучший по score.
        3. Альтернативы = следующие 2-3 с проверкой diversity.
        4. Формирование метрик и возврат OptimizationResult.

        Args:
            scored: оценённые варианты (List[ScoredCandidate] или List[Dict])
            num_alternatives: количество альтернатив (2-3)
            min_diversity: минимальная разница между альтернативами (0..1)

        Returns:
            OptimizationResult (поддерживает распаковку как (recommended, alternatives))
        """
        logger.info(f"Pareto: выбор топ-1 + {num_alternatives} альтернатив из {len(scored)} вариантов")

        def _get_val(item, attr, default=0.0):
            if isinstance(item, dict):
                return item.get(attr, default)
            return getattr(item, attr, default)

        if not scored:
            empty = ScoredCandidate(
                candidate=Candidate(params={}, blending={}, source='none', id=0),
                throughput=0,
                throughput_normalized=0,
                energy_proxy=0,
                energy_normalized=0,
                risk_index=0,
                risk_normalized=0,
                score=0,
                score_breakdown={}
            )
            return OptimizationResult(
                timestamp=pd.Timestamp.now(),
                candidates=[],
                feasible=[],
                ranked=[],
                recommended=empty,
                alternatives=[],
                metrics={'throughput': 0.0, 'energy': 0.0, 'risk': 0.0}
            )

        # ================================================================
        # 1. СОРТИРОВКА ПО SCORE (убывание)
        # ================================================================
        ranked = sorted(scored, key=lambda x: _get_val(x, 'score', 0.0), reverse=True)

        # ================================================================
        # 2. ТОП-1 РЕКОМЕНДАЦИЯ
        # ================================================================
        recommended = ranked[0]

        logger.info(
            f"Топ-1: score={_get_val(recommended, 'score', 0.0):.4f}, "
            f"throughput={_get_val(recommended, 'throughput', 0.0):.2f}, "
            f"energy={_get_val(recommended, 'energy_proxy', _get_val(recommended, 'energy', 0.0)):.3f}, "
            f"risk={_get_val(recommended, 'risk_index', _get_val(recommended, 'risk', 0.0)):.3f}"
        )

        # ================================================================
        # 3. АЛЬТЕРНАТИВЫ С ПРОВЕРКОЙ DIVERSITY
        # ================================================================
        alternatives = []

        is_all_scored_candidates = all(isinstance(x, ScoredCandidate) for x in ranked)

        if is_all_scored_candidates:
            for i in range(1, len(ranked)):
                candidate = ranked[i]
                is_diverse = self._check_diversity(
                    candidate=candidate,
                    reference=recommended,
                    alternatives=alternatives,
                    min_diversity=min_diversity
                )
                if is_diverse:
                    alternatives.append(candidate)
                    if len(alternatives) >= num_alternatives:
                        break

            # Если не набрали diversity, берем просто следующие по score
            if len(alternatives) < num_alternatives:
                for i in range(1, len(ranked)):
                    candidate = ranked[i]
                    if candidate not in alternatives:
                        alternatives.append(candidate)
                    if len(alternatives) >= num_alternatives:
                        break
        else:
            # Для словарей: срез [1:1+num_alternatives]
            alternatives = ranked[1:1 + num_alternatives]

        logger.info(f"Альтернативы: {len(alternatives)} вариантов")

        # ================================================================
        # 4. МЕТРИКИ (throughput, energy, risk)
        # ================================================================
        rec_throughput = _get_val(recommended, 'throughput', 0.0)
        rec_energy = _get_val(recommended, 'energy_proxy', _get_val(recommended, 'energy', 0.0))
        rec_risk = _get_val(recommended, 'risk_index', _get_val(recommended, 'risk', 0.0))
        rec_score = _get_val(recommended, 'score', 0.0)

        metrics = {
            'throughput': rec_throughput,
            'energy': rec_energy,
            'risk': rec_risk,
            'best_score': rec_score,
            'best_throughput': rec_throughput,
            'best_energy': rec_energy,
            'best_risk': rec_risk,
            'num_candidates': len(scored),
            'num_alternatives': len(alternatives),
            'pareto_front': [
                {
                    'id': getattr(getattr(alt, 'candidate', None), 'id', i),
                    'score': _get_val(alt, 'score', 0.0),
                    'throughput': _get_val(alt, 'throughput', 0.0),
                    'energy': _get_val(alt, 'energy_proxy', _get_val(alt, 'energy', 0.0)),
                    'risk': _get_val(alt, 'risk_index', _get_val(alt, 'risk', 0.0))
                }
                for i, alt in enumerate([recommended] + alternatives)
            ]
        }

        return OptimizationResult(
            timestamp=pd.Timestamp.now(),
            candidates=scored,
            feasible=scored,
            ranked=ranked,
            recommended=recommended,
            alternatives=alternatives,
            metrics=metrics
        )

    def _check_diversity(
        self,
        candidate: ScoredCandidate,
        reference: ScoredCandidate,
        alternatives: List[ScoredCandidate],
        min_diversity: float = 0.1
    ) -> bool:
        """
        Проверка diversity кандидата относительно reference и alternatives.

        Критерии diversity:
        - T6: разница ≥ min_diversity * диапазон (15°C)
        - F2_F26_ratio: разница ≥ min_diversity * диапазон (0.15)
        - Throughput: разница ≥ min_diversity * baseline (250 т/ч)

        Args:
            candidate: проверяемый кандидат
            reference: топ-1 рекомендация
            alternatives: уже выбранные альтернативы
            min_diversity: порог diversity (0..1)

        Returns:
            True, если кандидат достаточно разнообразен
        """
        # Диапазоны для diversity
        t6_range = self.controlled_params['T6'].max - self.controlled_params['T6'].min  # 15°C
        f2_range = self.controlled_params['F2_F26_ratio'].max - self.controlled_params['F2_F26_ratio'].min  # 0.15
        throughput_range = self.controlled_params['F9'].max - self.controlled_params['F9'].min  # 50 т/ч
        throughput_threshold = min_diversity * throughput_range  # 0.1 * 50 = 5.0 т/ч

        t6_threshold = min_diversity * t6_range  # 0.1 * 15 = 1.5°C
        f2_threshold = min_diversity * f2_range  # 0.1 * 0.15 = 0.015

        # Проверка diversity с reference (топ-1)
        t6_diff_ref = abs(candidate.candidate.params.get('T6', 0) - reference.candidate.params.get('T6', 0))
        f2_diff_ref = abs(candidate.candidate.params.get('F2_F26_ratio', 0) - reference.candidate.params.get('F2_F26_ratio', 0))
        throughput_diff_ref = abs(candidate.throughput - reference.throughput)

        is_diverse_from_ref = (
            t6_diff_ref >= t6_threshold or
            f2_diff_ref >= f2_threshold or
            throughput_diff_ref >= throughput_threshold
        )

        if not is_diverse_from_ref:
            return False

        # Проверка diversity с уже выбранными альтернативами
        for alt in alternatives:
            t6_diff_alt = abs(candidate.candidate.params.get('T6', 0) - alt.candidate.params.get('T6', 0))
            f2_diff_alt = abs(candidate.candidate.params.get('F2_F26_ratio', 0) - alt.candidate.params.get('F2_F26_ratio', 0))
            throughput_diff_alt = abs(candidate.throughput - alt.throughput)

            is_diverse_from_alt = (
                t6_diff_alt >= t6_threshold or
                f2_diff_alt >= f2_threshold or
                throughput_diff_alt >= throughput_threshold
            )

            if not is_diverse_from_alt:
                return False

        return True

    def rank_pareto(
        self,
        scored: List[Any],
        num_alternatives: int = 3
    ) -> OptimizationResult:
        """
        Основной метод ранжирования: топ-1 + альтернативы.

        Args:
            scored: оценённые варианты
            num_alternatives: количество альтернатив

        Returns:
            OptimizationResult
        """
        return self._rank_pareto(scored, num_alternatives=num_alternatives)

    # ========================================================================
    # ОСНОВНОЙ МЕТОД ОПТИМИЗАЦИИ
    # ========================================================================

    def optimize(
        self,
        current_state: Dict[str, float],
        quality_assessment: Optional[Dict[str, Any]] = None,
        reliability_assessment: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Основной метод оптимизации.

        Args:
            current_state: текущее состояние
            quality_assessment: оценка качества
            reliability_assessment: оценка надёжности

        Returns:
            OptimizationResult
        """
        logger.info("Запуск оптимизации")

        # 1. Генерация кандидатов
        candidates = self.generate_candidates(
            num_mode_candidates=50,
            num_blending_candidates=20,
            use_grid=False
        )

        logger.info(f"Сгенерировано {len(candidates)} кандидатов")

        # 2. Veto
        feasible = self.apply_veto(candidates, quality_assessment, reliability_assessment)

        logger.info(f"Допустимо {len(feasible)} кандидатов после veto")

        # 3. Если нет допустимых → возврат пустого результата
        if not feasible:
            empty_result = OptimizationResult(
                timestamp=pd.Timestamp.now(),
                candidates=candidates,
                feasible=[],
                ranked=[],
                recommended=ScoredCandidate(
                    candidate=Candidate(params={}, blending={}, source='none', id=0),
                    throughput=0,
                    throughput_normalized=0,
                    energy_proxy=0,
                    energy_normalized=0,
                    risk_index=0,
                    risk_normalized=0,
                    score=0,
                    score_breakdown={}
                ),
                alternatives=[],
                metrics={
                    'num_candidates': len(candidates),
                    'num_feasible': 0,
                    'num_ranked': 0,
                    'veto_reason': 'no_feasible'
                }
            )
            return empty_result

        # 4. Оценка вариантов (OPT-04)
        scored = self.score_candidates(feasible, current_state)

        # 5. Ранжирование (OPT-05)
        result = self.rank_pareto(scored, num_alternatives=3)

        # Заполняем candidates и feasible
        result.candidates = candidates
        result.feasible = feasible
        result.metrics['num_candidates'] = len(candidates)
        result.metrics['num_feasible'] = len(feasible)
        result.metrics['num_ranked'] = len(scored)

        logger.info(f"Оптимизация завершена: лучший score={result.recommended.score:.4f}")

        return result

    # ========================================================================
    # CLI ДЛЯ ТЕСТИРОВАНИЯ
    # ========================================================================

    def print_pareto_report(self, result: OptimizationResult):
        """Отчёт по Pareto-фронту."""
        print("\n" + "="*80)
        print("ОТЧЁТ ПО PARETO-ФРОНТУ (OPT-05)")
        print("="*80)

        print(f"Всего оценено: {len(result.ranked)}")
        print(f"Альтернатив: {len(result.alternatives)}")

        print(f"\n{'='*80}")
        print("ТОП-1 РЕКОМЕНДАЦИЯ")
        print(f"{'='*80}")

        rec = result.recommended
        print(f"ID: {rec.candidate.id}")
        print(f"Score: {rec.score:.4f}")
        print(f"Throughput: {rec.throughput:.2f} т/ч")
        print(f"Energy: {rec.energy_proxy:.3f}")
        print(f"Risk: {rec.risk_index:.3f}")

        if rec.candidate.params:
            print(f"\nПараметры:")
            for tag, value in rec.candidate.params.items():
                param = self.controlled_params.get(tag)
                unit = param.unit if param else ''
                print(f"  {tag}: {value} {unit}")

        if rec.candidate.blending:
            print(f"\nБлендинг:")
            for tag, value in rec.candidate.blending.items():
                param = self.blending_fractions.get(tag)
                unit = param.unit if param else ''
                print(f"  {tag}: {value} {unit}")

        print(f"\n{'='*80}")
        print(f"АЛЬТЕРНАТИВЫ ({len(result.alternatives)})")
        print(f"{'='*80}")

        for i, alt in enumerate(result.alternatives):
            print(f"\nАльтернатива {i+1} (ID: {alt.candidate.id})")
            print(f"  Score: {alt.score:.4f}")
            print(f"  Throughput: {alt.throughput:.2f} т/ч")
            print(f"  Energy: {alt.energy_proxy:.3f}")
            print(f"  Risk: {alt.risk_index:.3f}")

            # Разница с топ-1
            delta_score = alt.score - rec.score
            delta_throughput = alt.throughput - rec.throughput
            delta_energy = alt.energy_proxy - rec.energy_proxy
            delta_risk = alt.risk_index - rec.risk_index

            print(f"  Δ Score: {delta_score:+.4f}")
            print(f"  Δ Throughput: {delta_throughput:+.2f} т/ч")
            print(f"  Δ Energy: {delta_energy:+.3f}")
            print(f"  Δ Risk: {delta_risk:+.3f}")

        print(f"\n{'='*80}")
        print("METRICS")
        print(f"{'='*80}")

        for key, value in result.metrics.items():
            if key == 'pareto_front':
                print(f"{key}:")
                for item in value:
                    print(f"  - ID {item['id']}: score={item['score']:.4f}, throughput={item['throughput']:.2f}")
            else:
                print(f"{key}: {value}")

        print("="*80 + "\n")


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='OPT-05: Pareto-фронт')
    parser.add_argument('--num-mode', type=int, default=50, help='Кандидатов по режиму')
    parser.add_argument('--num-blend', type=int, default=20, help='Кандидатов по блендингу')
    parser.add_argument('--num-alternatives', type=int, default=3, help='Количество альтернатив')
    args = parser.parse_args()

    agent = OptimizationAgent()

    # Текущее состояние (моки)
    current_state = {
        'T6': 360.0,
        'F2_F26_ratio': 0.85,
        'T55': 380.0,
        'F9': 215.0
    }

    # Полный цикл оптимизации
    result = agent.optimize(current_state)

    # Отчёт
    agent.print_pareto_report(result)