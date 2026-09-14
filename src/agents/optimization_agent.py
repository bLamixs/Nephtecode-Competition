"""
Агент оптимизации: генерация и сравнение вариантов управления.

Методы:
- generate_candidates: генерация вариантов (grid/Sobol/Dirichlet)
- apply_veto: отбраковка по жёстким ограничениям (сера ≤ 10, доли = 100%, диапазоны)
- score_candidates: оценка вариантов (throughput, energy, risk)
- rank_pareto: выбор топ-1 + альтернативы
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
class OptimizationResult:
    """Результат оптимизации."""
    timestamp: pd.Timestamp
    candidates: List[Candidate]
    feasible: List[Candidate]
    ranked: List[Dict[str, Any]]
    recommended: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    metrics: Dict[str, Any]


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

CONTROLLED_PARAMS: Dict[str, ControlledParam] = {
    'T6': ControlledParam(
        current=295.0,
        min=290.0,
        max=305.0,
        unit='°C',
        description='Температура реактора гидроочистки (Р-201)',
        step=1.0
    ),
    'F2_F26_ratio': ControlledParam(
        current=0.85,
        min=0.80,
        max=0.95,
        unit='-',
        description='Соотношение ВСГ/сырьё',
        step=0.025
    ),
    'T55': ControlledParam(
        current=320.0,
        min=315.0,
        max=330.0,
        unit='°C',
        description='Температура печи П-1/1',
        step=2.5
    ),
    'F9': ControlledParam(
        current=250.0,
        min=225.0,
        max=275.0,
        unit='т/ч',
        description='Расход обессоленной нефти на гидроочистку',
        step=5.0
    ),
}

BLENDING_FRACTIONS: Dict[str, ControlledParam] = {
    'F30': ControlledParam(current=0.30, min=0.25, max=0.35, unit='доля', description='Доля фракции 240-290°C'),
    'F32': ControlledParam(current=0.25, min=0.20, max=0.30, unit='доля', description='Доля фракции 290-350°C'),
    'F34': ControlledParam(current=0.20, min=0.15, max=0.25, unit='доля', description='Доля фракции 350-500°C'),
    'F56': ControlledParam(current=0.10, min=0.05, max=0.15, unit='доля', description='Доля лёгкого компонента'),
    'F57': ControlledParam(current=0.10, min=0.05, max=0.15, unit='доля', description='Доля тяжёлого компонента'),
    'F59': ControlledParam(current=0.05, min=0.02, max=0.08, unit='доля', description='Доля присадки'),
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

        self._candidate_id = 0

        # Жёсткие ограничения (из ТЗ)
        self.hard_constraints = {
            'sulfur_max': 10.0,  # Сера ≤ 10 мг/кг
            'blending_sum_tolerance': 0.01,  # Доли = 100% ± 1%
        }

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
    # VETO ПО ЖЁСТКИМ ОГРАНИЧЕНИЯМ
    # ========================================================================

    def apply_veto(
        self,
        candidates: List[Candidate],
        quality_assessment: Optional[Dict[str, Any]] = None,
        reliability_assessment: Optional[Dict[str, Any]] = None
    ) -> List[Candidate]:
        """
        Отбраковка кандидатов по жёстким ограничениям.

        Жёсткие ограничения:
        1. Сера ≤ 10 мг/кг
        2. Доли блендинга = 100% (сумма = 1.0)
        3. Модельные диапазоны (T6, F2/F26, etc.)

        Args:
            candidates: список кандидатов
            quality_assessment: оценка качества (от Quality Agent)
            reliability_assessment: оценка надёжности (от Reliability Agent)

        Returns:
            Список допустимых кандидатов
        """
        logger.info(f"Veto: {len(candidates)} кандидатов до отбраковки")

        feasible = []
        veto_stats = {
            'sulfur': 0,
            'blending_sum': 0,
            'param_range': 0,
            'blending_range': 0,
            'quality_veto': 0,
            'reliability_veto': 0
        }

        for candidate in candidates:
            veto_reasons = []

            # ================================================================
            # 1. Проверка серы (сера ≤ 10 мг/кг)
            # ================================================================

            sulfur_veto, sulfur_reason = self._check_sulfur_veto(candidate, quality_assessment)
            if sulfur_veto:
                veto_reasons.append(sulfur_reason)
                veto_stats['sulfur'] += 1
                veto_stats['quality_veto'] += 1

            # ================================================================
            # 2. Проверка долей блендинга (сумма = 100%)
            # ================================================================

            blending_veto, blending_reason = self._check_blending_sum_veto(candidate)
            if blending_veto:
                veto_reasons.append(blending_reason)
                veto_stats['blending_sum'] += 1

            # ================================================================
            # 3. Проверка диапазонов параметров (T6, F2/F26, etc.)
            # ================================================================

            param_range_veto, param_range_reason = self._check_param_ranges_veto(candidate)
            if param_range_veto:
                veto_reasons.append(param_range_reason)
                veto_stats['param_range'] += 1

            # ================================================================
            # 4. Проверка диапазонов долей блендинга
            # ================================================================

            blending_range_veto, blending_range_reason = self._check_blending_ranges_veto(candidate)
            if blending_range_veto:
                veto_reasons.append(blending_range_reason)
                veto_stats['blending_range'] += 1

            # ================================================================
            # 5. Проверка надёжности (risk_class != 'high')
            # ================================================================

            reliability_veto, reliability_reason = self._check_reliability_veto(candidate, reliability_assessment)
            if reliability_veto:
                veto_reasons.append(reliability_reason)
                veto_stats['reliability_veto'] += 1

            # ================================================================
            # ИТОГ: если нет veto → допустимый кандидат
            # ================================================================

            if not veto_reasons:
                feasible.append(candidate)
            else:
                logger.debug(f"Кандидат #{candidate.id} отклонён: {veto_reasons}")

        logger.info(f"Veto: {len(feasible)}/{len(candidates)} кандидатов допустимы")
        logger.info(f"Veto статистика: {veto_stats}")

        return feasible

    def _check_sulfur_veto(
        self,
        candidate: Candidate,
        quality_assessment: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Проверка серы (сера ≤ 10 мг/кг).

        Если quality_assessment есть → используем прогноз серы.
        Иначе → упрощённая оценка по T6.

        Returns:
            (veto, причина)
        """
        # 1. Если есть quality_assessment → используем прогноз
        if quality_assessment:
            sulfur_forecast = quality_assessment.get('predictions', {}).get('Sulfur', 8.5)
            sulfur_risk = quality_assessment.get('risk_spec_violation', {}).get('P_S_gt_10', 0.0)

            # Veto если P(S>10) > 0.1
            if sulfur_risk > 0.1:
                return True, f"Риск серы: P(S>10)={sulfur_risk:.3f} > 0.1"

            # Veto если прогноз > 10
            if sulfur_forecast > self.hard_constraints['sulfur_max']:
                return True, f"Сера={sulfur_forecast:.2f} > {self.hard_constraints['sulfur_max']}"

        # 2. Если нет quality_assessment → упрощённая оценка по T6
        # (чем выше T6, тем ниже сера)
        t6 = candidate.params.get('T6', 295.0)

        # Упрощённая модель: сера = 15 - 0.02 * (T6 - 290)
        sulfur_estimate = 15.0 - 0.02 * (t6 - 290.0)

        if sulfur_estimate > self.hard_constraints['sulfur_max']:
            return True, f"Оценка серы={sulfur_estimate:.2f} > {self.hard_constraints['sulfur_max']} (T6={t6})"

        return False, None

    def _check_blending_sum_veto(self, candidate: Candidate) -> Tuple[bool, Optional[str]]:
        """
        Проверка долей блендинга (сумма = 100%).

        Returns:
            (veto, причина)
        """
        if not candidate.blending:
            return False, None

        blend_sum = sum(candidate.blending.values())
        tolerance = self.hard_constraints['blending_sum_tolerance']

        if not np.isclose(blend_sum, 1.0, atol=tolerance):
            return True, f"Сумма долей={blend_sum:.4f} ≠ 1.0 (tolerance={tolerance})"

        return False, None

    def _check_param_ranges_veto(self, candidate: Candidate) -> Tuple[bool, Optional[str]]:
        """
        Проверка диапазонов параметров (T6, F2/F26, etc.).

        Returns:
            (veto, причина)
        """
        for tag, value in candidate.params.items():
            if tag not in self.controlled_params:
                continue

            param = self.controlled_params[tag]

            if value < param.min or value > param.max:
                return True, f"{tag}={value} вне диапазона [{param.min}, {param.max}]"

        return False, None

    def _check_blending_ranges_veto(self, candidate: Candidate) -> Tuple[bool, Optional[str]]:
        """
        Проверка диапазонов долей блендинга.

        Returns:
            (veto, причина)
        """
        if not candidate.blending:
            return False, None

        for tag, value in candidate.blending.items():
            if tag not in self.blending_fractions:
                continue

            param = self.blending_fractions[tag]

            if value < param.min or value > param.max:
                return True, f"{tag}={value} вне диапазона [{param.min}, {param.max}]"

        return False, None

    def _check_reliability_veto(
        self,
        candidate: Candidate,
        reliability_assessment: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Проверка надёжности (risk_class != 'high').

        Returns:
            (veto, причина)
        """
        if not reliability_assessment:
            return False, None

        risk_class = reliability_assessment.get('risk_class', 'low')

        if risk_class == 'high':
            return True, f"Недопустимый риск оборудования: {risk_class}"

        return False, None

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
        # 1. Генерация кандидатов
        candidates = self.generate_candidates(
            num_mode_candidates=50,
            num_blending_candidates=20,
            use_grid=True
        )

        # 2. Veto
        feasible = self.apply_veto(candidates, quality_assessment, reliability_assessment)

        # 3. Если нет допустимых → возврат пустого результата
        if not feasible:
            return OptimizationResult(
                timestamp=pd.Timestamp.now(),
                candidates=candidates,
                feasible=[],
                ranked=[],
                recommended={},
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
        ranked = sorted(scored, key=lambda x: x.get('score', 0), reverse=True)

        # 6. Выбор топ-1 + альтернативы
        recommended = ranked[0] if ranked else {}
        alternatives = ranked[1:4] if len(ranked) > 1 else []

        return OptimizationResult(
            timestamp=pd.Timestamp.now(),
            candidates=candidates,
            feasible=feasible,
            ranked=ranked,
            recommended=recommended,
            alternatives=alternatives,
            metrics={
                'num_candidates': len(candidates),
                'num_feasible': len(feasible),
                'num_ranked': len(ranked)
            }
        )

    def score_candidates(
        self,
        feasible: List[Candidate],
        current_state: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Оценка вариантов (OPT-04).

        Args:
            feasible: допустимые варианты
            current_state: текущее состояние

        Returns:
            Список с метриками
        """
        scored = []

        for candidate in feasible:
            # Throughput: F9
            throughput = candidate.params.get('F9', current_state.get('F9', 250.0))

            # Energy proxy: T6 (чем выше, тем больше энергии)
            t6 = candidate.params.get('T6', 295.0)
            energy_proxy = (t6 - 290.0) / 15.0  # нормализация 0..1

            # Risk: отклонение T6 от нормы
            t6_norm = self.controlled_params['T6'].current
            risk = abs(t6 - t6_norm) / (self.controlled_params['T6'].max - t6_norm)
            risk = min(1.0, risk)

            # Score: J = w1*throughput - w2*energy - w3*risk
            score = (
                self.weights['throughput'] * (throughput / 250.0)  # нормализация
                - self.weights['energy'] * energy_proxy
                - self.weights['risk'] * risk
            )

            scored.append({
                'id': candidate.id,
                'params': candidate.params,
                'blending': candidate.blending,
                'throughput': throughput,
                'energy_proxy': energy_proxy,
                'risk': risk,
                'score': score
            })

        return scored

    def rank_pareto(self, scored: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Выбор топ-1 + альтернативы (OPT-05).

        Args:
            scored: оценённые варианты

        Returns:
            (топ-1, альтернативы)
        """
        if not scored:
            return {}, []

        # Сортировка по score
        ranked = sorted(scored, key=lambda x: x.get('score', 0), reverse=True)

        recommended = ranked[0]
        alternatives = ranked[1:4] if len(ranked) > 1 else []

        return recommended, alternatives

    # ========================================================================
    # CLI ДЛЯ ТЕСТИРОВАНИЯ
    # ========================================================================

    def print_veto_report(self, candidates: List[Candidate], feasible: List[Candidate]):
        """Отчёт по veto."""
        print("\n" + "="*80)
        print("ОТЧЁТ ПО VETO")
        print("="*80)
        print(f"Кандидатов до veto: {len(candidates)}")
        print(f"Кандидатов после veto: {len(feasible)}")
        print(f"Отбраковано: {len(candidates) - len(feasible)} ({100*(len(candidates)-len(feasible))/len(candidates):.1f}%)")
        print("="*80 + "\n")


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='OPT-03: Veto по жёстким ограничениям')
    parser.add_argument('--num-mode', type=int, default=50, help='Кандидатов по режиму')
    parser.add_argument('--num-blend', type=int, default=20, help='Кандидатов по блендингу')
    args = parser.parse_args()

    agent = OptimizationAgent()

    # Генерация
    candidates = agent.generate_candidates(
        num_mode_candidates=args.num_mode,
        num_blending_candidates=args.num_blend,
        use_grid=False  # Sobol для быстрого теста
    )

    # Veto (без quality/reliability assessment)
    feasible = agent.apply_veto(candidates)

    # Отчёт
    agent.print_veto_report(candidates, feasible)

    # Примеры допустимых
    print(f"Первые 5 допустимых кандидатов:")
    for i, candidate in enumerate(feasible[:5]):
        print(f"{i+1}. T6={candidate.params.get('T6', 'N/A')}, F2/F26={candidate.params.get('F2_F26_ratio', 'N/A')}")