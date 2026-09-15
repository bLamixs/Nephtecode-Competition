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
    """Результат оптимизации."""
    timestamp: pd.Timestamp
    candidates: List[Candidate]
    feasible: List[Candidate]
    ranked: List[ScoredCandidate]
    recommended: ScoredCandidate
    alternatives: List[ScoredCandidate]
    metrics: Dict[str, Any]


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

CONTROLLED_PARAMS: Dict[str, ControlledParam] = {
    'T6': ControlledParam(current=295.0, min=290.0, max=305.0, unit='°C', description='Температура реактора', step=1.0),
    'F2_F26_ratio': ControlledParam(current=0.85, min=0.80, max=0.95, unit='-', description='ВСГ/сырьё', step=0.025),
    'T55': ControlledParam(current=320.0, min=315.0, max=330.0, unit='°C', description='Температура печи', step=2.5),
    'F9': ControlledParam(current=250.0, min=225.0, max=275.0, unit='т/ч', description='Расход на гидроочистку', step=5.0),
}

BLENDING_FRACTIONS: Dict[str, ControlledParam] = {
    'F30': ControlledParam(current=0.30, min=0.25, max=0.35, unit='доля', description='Фракция 240-290°C'),
    'F32': ControlledParam(current=0.25, min=0.20, max=0.30, unit='доля', description='Фракция 290-350°C'),
    'F34': ControlledParam(current=0.20, min=0.15, max=0.25, unit='доля', description='Фракция 350-500°C'),
    'F56': ControlledParam(current=0.10, min=0.05, max=0.15, unit='доля', description='Лёгкий компонент'),
    'F57': ControlledParam(current=0.10, min=0.05, max=0.15, unit='доля', description='Тяжёлый компонент'),
    'F59': ControlledParam(current=0.05, min=0.02, max=0.08, unit='доля', description='Присадка'),
}


class OptimizationAgent:
    """
    Агент оптимизации: генерация и сравнение вариантов управления.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.controlled_params = CONTROLLED_PARAMS
        self.blending_fractions = BLENDING_FRACTIONS

        # Веса целевой функции (из config.yaml)
        self.weights = {
            'throughput': 0.5,  # Производительность
            'energy': 0.3,      # Энергозатраты
            'risk': 0.2         # Риск оборудования
        }

        # Базовые значения для нормализации
        self.baseline = {
            'throughput': 250.0,  # т/ч (F9)
            'energy': 1.0,        # нормализовано
            'risk': 1.0           # нормализовано
        }

        self._candidate_id = 0

    def _next_id(self) -> int:
        self._candidate_id += 1
        return self._candidate_id

    # ========================================================================
    # ГЕНЕРАЦИЯ (grid, Sobol, Dirichlet) — из OPT-02
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
        """Dirichlet для блендинга (сумма = 1.0)."""
        fraction_names = list(self.blending_fractions.keys())
        n_fractions = len(fraction_names)

        candidates = []
        attempts = 0
        max_attempts = num_candidates * 10

        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1

            alpha = np.ones(n_fractions)
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
                candidates.append(Candidate(params={}, blending=candidate, source='dirichlet', id=self._next_id()))

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

    def _check_sulfur_veto(self, candidate: Candidate, quality_assessment: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        if quality_assessment:
            if hasattr(quality_assessment, 'sulfur_forecast_mg_kg'):
                sulfur_forecast = quality_assessment.sulfur_forecast_mg_kg
                sulfur_risk = getattr(quality_assessment, 'risk_sulfur_violation', 0.0)
            elif isinstance(quality_assessment, dict):
                sulfur_forecast = quality_assessment.get('predictions', {}).get('Sulfur', quality_assessment.get('sulfur_forecast', 8.5))
                sulfur_risk = quality_assessment.get('risk_spec_violation', {}).get('P_S_gt_10', quality_assessment.get('risk_sulfur_violation', 0.0))
            else:
                sulfur_forecast = 8.5
                sulfur_risk = 0.0

            if sulfur_risk > 0.1:
                return True, f"P(S>10)={sulfur_risk:.3f}"
            if sulfur_forecast > 10.0:
                return True, f"Сера={sulfur_forecast:.2f}"

            return False, None

        # Эвристическая оценка серы при отсутствии прогноза от Quality Agent:
        # Базовый уровень при T6=295°C равен 8.5 мг/кг.
        # Согласно регламенту (controlled_params.md), повышение T6 на 3°C снижает серу на ~1.8 мг/кг (0.6 мг/кг на °C).
        t6 = candidate.params.get('T6', 295.0) if hasattr(candidate, 'params') else candidate.get('T6', 295.0)
        sulfur_estimate = 8.5 - 0.6 * (t6 - 295.0)

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
        """
        Оценка вариантов по целевой функции.

        Целевая функция:
        J = w1 * throughput_normalized - w2 * energy_normalized - w3 * risk_normalized

        где:
        - throughput_normalized = F9 / baseline_throughput (нормализация)
        - energy_normalized = (T6 - 290) / 15 (нормализация 0..1)
        - risk_normalized = |T6 - T6_norm| / (T6_max - T6_norm) (нормализация 0..1)

        Args:
            feasible: допустимые варианты
            current_state: текущее состояние (для baseline)

        Returns:
            Список ScoredCandidate
        """
        logger.info(f"Оценка: {len(feasible)} допустимых вариантов")

        if current_state is None:
            current_state = {}

        scored = []

        for candidate in feasible:
            # ================================================================
            # 1. THROUGHPUT (производительность, т/ч)
            # ================================================================

            throughput = self._estimate_throughput(candidate, current_state)
            throughput_normalized = throughput / self.baseline['throughput']

            # ================================================================
            # 2. ENERGY PROXY (энергозатраты, нормализовано 0..1)
            # ================================================================

            energy_proxy = self._estimate_energy_proxy(candidate, current_state)
            energy_normalized = min(1.0, max(0.0, energy_proxy))

            # ================================================================
            # 3. RISK INDEX (риск оборудования, нормализовано 0..1)
            # ================================================================

            risk_index = self._estimate_risk_index(candidate, current_state)
            risk_normalized = min(1.0, max(0.0, risk_index))

            # ================================================================
            # 4. ЦЕЛЕВАЯ ФУНКЦИЯ (взвешенная сумма)
            # ================================================================

            score = (
                self.weights['throughput'] * throughput_normalized
                - self.weights['energy'] * energy_normalized
                - self.weights['risk'] * risk_normalized
            )

            # ================================================================
            # 5. СОЗДАНИЕ ScoredCandidate
            # ================================================================

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

        # Сортировка по score (убывание)
        scored.sort(key=lambda x: x.score, reverse=True)

        if scored:
            logger.info(f"Оценка: лучший score={scored[0].score:.4f}, худший score={scored[-1].score:.4f}")
        else:
            logger.warning("Оценка: нет допустимых кандидатов для скоринга")

        return scored

    def _estimate_throughput(self, candidate: Candidate, current_state: Dict[str, float]) -> float:
        """
        Оценка throughput (т/ч).

        Прокси: F9 (расход на гидроочистку).

        Args:
            candidate: кандидат
            current_state: текущее состояние

        Returns:
            Throughput (т/ч)
        """
        # F9 — основной драйвер throughput
        f9 = candidate.params.get('F9', current_state.get('F9', self.baseline['throughput']))

        # Влияние блендинга на выпуск ДТ
        # F30, F32 — лёгкие фракции → больше ДТ
        # F34, F57 — тяжёлые фракции → меньше ДТ
        blend_factor = 1.0

        if candidate.blending:
            light_fractions = candidate.blending.get('F30', 0.3) + candidate.blending.get('F32', 0.25)
            heavy_fractions = candidate.blending.get('F34', 0.2) + candidate.blending.get('F57', 0.1)

            # Больше лёгких → больше ДТ
            blend_factor = 1.0 + 0.1 * (light_fractions - 0.55) - 0.1 * (heavy_fractions - 0.3)

        throughput = f9 * blend_factor

        return throughput

    def _estimate_energy_proxy(self, candidate: Candidate, current_state: Dict[str, float]) -> float:
        """
        Оценка энергозатрат (прокси, нормализовано 0..1).

        Прокси:
        - T6: температура реактора (чем выше, тем больше энергии)
        - T55: температура печи (чем выше, тем больше энергии)

        Args:
            candidate: кандидат
            current_state: текущее состояние

        Returns:
            Energy proxy (0..1)
        """
        # T6: нормализация (290..305 → 0..1)
        t6 = candidate.params.get('T6', current_state.get('T6', 295.0))
        t6_normalized = (t6 - 290.0) / 15.0
        t6_normalized = min(1.0, max(0.0, t6_normalized))

        # T55: нормализация (315..330 → 0..1)
        t55 = candidate.params.get('T55', current_state.get('T55', 320.0))
        t55_normalized = (t55 - 315.0) / 15.0
        t55_normalized = min(1.0, max(0.0, t55_normalized))

        # F2_F26_ratio: влияние на компрессоры (0.80..0.95 → 0..1)
        f2_ratio = candidate.params.get('F2_F26_ratio', current_state.get('F2_F26_ratio', 0.85))
        f2_normalized = (f2_ratio - 0.80) / 0.15
        f2_normalized = min(1.0, max(0.0, f2_normalized))

        # Интегральный energy proxy (взвешенная сумма)
        energy_proxy = (
            0.5 * t6_normalized +      # T6 — 50%
            0.3 * t55_normalized +     # T55 — 30%
            0.2 * f2_normalized        # F2/F26 — 20%
        )

        return energy_proxy

    def _estimate_risk_index(self, candidate: Candidate, current_state: Dict[str, float]) -> float:
        """
        Оценка риска оборудования (0..1).

        Прокси:
        - Отклонение T6 от нормы (295°C)
        - Отклонение T55 от нормы (320°C)
        - Отклонение F2_F26_ratio от нормы (0.85)

        Args:
            candidate: кандидат
            current_state: текущее состояние

        Returns:
            Risk index (0..1)
        """
        risks = []

        # 1. T6: отклонение от нормы
        t6 = candidate.params.get('T6', current_state.get('T6', 295.0))
        t6_norm = self.controlled_params['T6'].current
        t6_max = self.controlled_params['T6'].max

        t6_deviation = abs(t6 - t6_norm) / (t6_max - t6_norm)
        risks.append(t6_deviation)

        # 2. T55: отклонение от нормы
        t55 = candidate.params.get('T55', current_state.get('T55', 320.0))
        t55_norm = self.controlled_params['T55'].current
        t55_max = self.controlled_params['T55'].max

        t55_deviation = abs(t55 - t55_norm) / (t55_max - t55_norm)
        risks.append(t55_deviation)

        # 3. F2_F26_ratio: отклонение от нормы
        f2_ratio = candidate.params.get('F2_F26_ratio', current_state.get('F2_F26_ratio', 0.85))
        f2_norm = self.controlled_params['F2_F26_ratio'].current
        f2_range = self.controlled_params['F2_F26_ratio'].max - self.controlled_params['F2_F26_ratio'].min

        f2_deviation = abs(f2_ratio - f2_norm) / f2_range
        risks.append(f2_deviation)

        # Интегральный risk index (среднее)
        risk_index = np.mean(risks)

        return risk_index

    # ========================================================================
    # РАНЖИРОВАНИЕ (OPT-05)
    # ========================================================================

    def rank_pareto(self, scored: List[ScoredCandidate]) -> Tuple[ScoredCandidate, List[ScoredCandidate]]:
        """
        Выбор топ-1 + альтернативы.

        Args:
            scored: оценённые варианты

        Returns:
            (топ-1, альтернативы)
        """
        if not scored:
            return ScoredCandidate(
                candidate=Candidate(params={}, blending={}, source='none', id=0),
                throughput=0,
                throughput_normalized=0,
                energy_proxy=0,
                energy_normalized=0,
                risk_index=0,
                risk_normalized=0,
                score=0,
                score_breakdown={}
            ), []

        # Сортировка уже выполнена в score_candidates
        recommended = scored[0]
        alternatives = scored[1:4] if len(scored) > 1 else []

        return recommended, alternatives

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
            use_grid=False  # Sobol для скорости
        )

        logger.info(f"Сгенерировано {len(candidates)} кандидатов")

        # 2. Veto
        feasible = self.apply_veto(candidates, quality_assessment, reliability_assessment)

        logger.info(f"Допустимо {len(feasible)} кандидатов после veto")

        # 3. Если нет допустимых → возврат пустого результата
        if not feasible:
            return OptimizationResult(
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

        # 4. Оценка вариантов (OPT-04)
        scored = self.score_candidates(feasible, current_state)

        # 5. Ранжирование (OPT-05)
        recommended, alternatives = self.rank_pareto(scored)

        logger.info(f"Оптимизация завершена: лучший score={recommended.score:.4f}")

        return OptimizationResult(
            timestamp=pd.Timestamp.now(),
            candidates=candidates,
            feasible=feasible,
            ranked=scored,
            recommended=recommended,
            alternatives=alternatives,
            metrics={
                'num_candidates': len(candidates),
                'num_feasible': len(feasible),
                'num_ranked': len(scored),
                'best_score': recommended.score,
                'best_throughput': recommended.throughput,
                'best_energy': recommended.energy_proxy,
                'best_risk': recommended.risk_index
            }
        )

    # ========================================================================
    # CLI ДЛЯ ТЕСТИРОВАНИЯ
    # ========================================================================

    def print_score_report(self, scored: List[ScoredCandidate], max_show: int = 10):
        """Отчёт по оценкам."""
        print("\n" + "="*80)
        print("ОТЧЁТ ПО ОЦЕНКЕ ВАРИАНТОВ (OPT-04)")
        print("="*80)
        print(f"Оценено вариантов: {len(scored)}")

        if scored:
            print(f"\nЛучший вариант:")
            best = scored[0]
            print(f"  Score: {best.score:.4f}")
            print(f"  Throughput: {best.throughput:.2f} т/ч (норм={best.throughput_normalized:.3f})")
            print(f"  Energy: {best.energy_proxy:.3f} (норм={best.energy_normalized:.3f})")
            print(f"  Risk: {best.risk_index:.3f} (норм={best.risk_normalized:.3f})")
            print(f"\n  Компоненты score:")
            for component, value in best.score_breakdown.items():
                print(f"    {component}: {value:+.4f}")

        print(f"\nТоп-{min(max_show, len(scored))} вариантов:")
        print(f"{'#':<4} {'Score':<8} {'Throughput':<12} {'Energy':<8} {'Risk':<8}")
        print("-" * 80)

        for i, s in enumerate(scored[:max_show]):
            print(f"{i+1:<4} {s.score:<8.4f} {s.throughput:<12.2f} {s.energy_proxy:<8.3f} {s.risk_index:<8.3f}")

        print("="*80 + "\n")


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='OPT-04: Оценка вариантов')
    parser.add_argument('--num-mode', type=int, default=50, help='Кандидатов по режиму')
    parser.add_argument('--num-blend', type=int, default=20, help='Кандидатов по блендингу')
    args = parser.parse_args()

    agent = OptimizationAgent()

    # Текущее состояние (моки)
    current_state = {
        'T6': 295.0,
        'F2_F26_ratio': 0.85,
        'T55': 320.0,
        'F9': 250.0
    }

    # Генерация
    candidates = agent.generate_candidates(
        num_mode_candidates=args.num_mode,
        num_blending_candidates=args.num_blend,
        use_grid=False
    )

    # Veto
    feasible = agent.apply_veto(candidates)

    # Оценка (OPT-04)
    scored = agent.score_candidates(feasible, current_state)

    # Отчёт
    agent.print_score_report(scored, max_show=10)

    # Оптимизация (полный цикл)
    result = agent.optimize(current_state)

    print(f"\nПолный цикл оптимизации:")
    print(f"  Кандидатов: {result.metrics['num_candidates']}")
    print(f"  Допустимо: {result.metrics['num_feasible']}")
    print(f"  Лучший score: {result.metrics['best_score']:.4f}")
    print(f"  Лучший throughput: {result.metrics['best_throughput']:.2f} т/ч")
    print(f"  Лучший energy: {result.metrics['best_energy']:.3f}")
    print(f"  Лучший risk: {result.metrics['best_risk']:.3f}")