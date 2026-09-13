"""
Агент оптимизации: генерация и сравнение вариантов управления.

Управляемые параметры:
- T6: температура реактора гидроочистки (°C)
- F2_F26_ratio: соотношение ВСГ/сырьё
- Доли блендинга: F30, F32, F34, F56, F57, F59 (сумма = 100%)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import qmc


@dataclass
class ControlledParam:
    """Описание управляемого параметра."""
    current: float
    min: float
    max: float
    unit: str
    description: str


@dataclass
class OptimizationResult:
    """Результат оптимизации."""
    timestamp: pd.Timestamp
    candidates: List[Dict[str, Any]]
    feasible: List[Dict[str, Any]]
    ranked: List[Dict[str, Any]]
    recommended: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    metrics: Dict[str, Any]


# ============================================================================
# КОНФИГУРАЦИЯ УПРАВЛЯЕМЫХ ПАРАМЕТРОВ
# ============================================================================

CONTROLLED_PARAMS: Dict[str, ControlledParam] = {
    'T6': ControlledParam(
        current=295.0,
        min=290.0,
        max=305.0,
        unit='°C',
        description='Температура реактора гидроочистки (Р-201)'
    ),

    'F2_F26_ratio': ControlledParam(
        current=0.85,
        min=0.80,
        max=0.95,
        unit='-',
        description='Соотношение ВСГ/сырьё (водородосодержащий газ / расход сырья)'
    ),

    'T55': ControlledParam(
        current=320.0,
        min=315.0,
        max=330.0,
        unit='°C',
        description='Температура печи П-1/1 (нагрев нефти)'
    ),

    'F9': ControlledParam(
        current=250.0,
        min=225.0,
        max=275.0,
        unit='т/ч',
        description='Расход обессоленной нефти на гидроочистку'
    ),
}

# Доли блендинга (сумма должна быть = 1.0)
BLENDING_FRACTIONS: Dict[str, ControlledParam] = {
    'F30': ControlledParam(
        current=0.30,
        min=0.25,
        max=0.35,
        unit='доля',
        description='Доля фракции 240-290°C в блендинге ДТ'
    ),

    'F32': ControlledParam(
        current=0.25,
        min=0.20,
        max=0.30,
        unit='доля',
        description='Доля фракции 290-350°C в блендинге ДТ'
    ),

    'F34': ControlledParam(
        current=0.20,
        min=0.15,
        max=0.25,
        unit='доля',
        description='Доля фракции 350-500°C в блендинге ДТ'
    ),

    'F56': ControlledParam(
        current=0.10,
        min=0.05,
        max=0.15,
        unit='доля',
        description='Доля лёгкого компонента в блендинге'
    ),

    'F57': ControlledParam(
        current=0.10,
        min=0.05,
        max=0.15,
        unit='доля',
        description='Доля тяжёлого компонента в блендинге'
    ),

    'F59': ControlledParam(
        current=0.05,
        min=0.02,
        max=0.08,
        unit='доля',
        description='Доля присадки в блендинге'
    ),
}


class OptimizationAgent:
    """
    Агент оптимизации: генерация и сравнение вариантов управления.

    Методы:
    - _generate_candidates: генерация вариантов (grid/Sobol)
    - _apply_veto: отбраковка по жёстким ограничениям
    - _score_candidates: оценка вариантов (throughput, energy, risk)
    - _rank_pareto: выбор топ-1 + альтернативы
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Инициализация агента оптимизации.

        Args:
            config_path: путь к файлу конфигурации (config.yaml)
        """
        self.config_path = config_path
        self.controlled_params = CONTROLLED_PARAMS
        self.blending_fractions = BLENDING_FRACTIONS

        # Веса целевой функции (из config.yaml)
        self.weights = {
            'throughput': 0.5,
            'energy': 0.3,
            'risk': 0.2
        }

    def _generate_candidates(
            self,
            num_params: int,
            num_candidates: int = 50
    ) -> List[Dict[str, float]]:
        """
        Генерация вариантов изменения режима.

        Args:
            num_params: количество управляемых параметров
            num_candidates: количество вариантов

        Returns:
            Список кандидатов (каждый — Dict[tag, value])
        """
        if num_params <= 3:
            # Grid search для 2-3 параметров
            return self._grid_search()
        else:
            # Sobol sequence для >3 параметров
            return self._sobol_sampling(num_candidates)

    def _grid_search(self) -> List[Dict[str, float]]:
        """
        Grid search для 2-3 параметров.

        Returns:
            Список кандидатов
        """
        import itertools

        # Значения для каждого параметра (3-5 точек)
        grids = {
            'T6': np.linspace(
                self.controlled_params['T6'].min,
                self.controlled_params['T6'].max,
                5
            ),
            'F2_F26_ratio': np.linspace(
                self.controlled_params['F2_F26_ratio'].min,
                self.controlled_params['F2_F26_ratio'].max,
                4
            ),
        }

        # Декартово произведение
        candidates = []
        for t6, ratio in itertools.product(grids['T6'], grids['F2_F26_ratio']):
            candidate = {
                'T6': float(t6),
                'F2_F26_ratio': float(ratio),
            }
            candidates.append(candidate)

        return candidates

    def _sobol_sampling(self, num_candidates: int) -> List[Dict[str, float]]:
        """
        Sobol sequence для >3 параметров.

        Args:
            num_candidates: количество вариантов

        Returns:
            Список кандидатов
        """
        # Параметры для сэмплирования
        param_names = list(self.controlled_params.keys())
        d = len(param_names)

        # Sobol sequence
        sampler = qmc.Sobol(d=d, scramble=True)
        samples = sampler.random_base2(m=int(np.log2(num_candidates)))

        # Масштабирование в диапазоны
        candidates = []
        for sample in samples:
            candidate = {}
            for i, name in enumerate(param_names):
                param = self.controlled_params[name]
                value = param.min + sample[i] * (param.max - param.min)
                candidate[name] = float(value)
            candidates.append(candidate)

        return candidates

    def _generate_blending_candidates(self, num_candidates: int = 20) -> List[Dict[str, float]]:
        """
        Генерация вариантов блендинга (сумма долей = 1.0).

        Args:
            num_candidates: количество вариантов

        Returns:
            Список кандидатов (каждый — Dict[фракция, доля])
        """
        # Dirichlet distribution (сумма = 1.0)
        candidates = []

        for _ in range(num_candidates):
            # Параметры Dirichlet (равномерное распределение)
            alpha = np.ones(len(self.blending_fractions))

            # Сэмплирование
            fractions = np.random.dirichlet(alpha)

            # Проверка диапазонов
            valid = True
            candidate = {}
            for i, (name, param) in enumerate(self.blending_fractions.items()):
                value = fractions[i]

                # Если выходит за диапазон — отбросить
                if value < param.min or value > param.max:
                    valid = False
                    break

                candidate[name] = float(value)

            # Нормализация (сумма = 1.0)
            if valid:
                total = sum(candidate.values())
                candidate = {k: v / total for k, v in candidate.items()}
                candidates.append(candidate)

        return candidates

    def optimize(
            self,
            current_state: Dict[str, float],
            quality_assessment: Any,
            reliability_assessment: Any
    ) -> OptimizationResult:
        """
        Основной метод оптимизации.

        Args:
            current_state: текущее состояние процесса
            quality_assessment: оценка качества (от Quality Agent)
            reliability_assessment: оценка надёжности (от Reliability Agent)

        Returns:
            OptimizationResult
        """
        # 1. Генерация кандидатов
        candidates = self._generate_candidates(
            num_params=len(self.controlled_params),
            num_candidates=50
        )

        # 2. Генерация кандидатов блендинга
        blending_candidates = self._generate_blending_candidates(num_candidates=20)

        # 3. Комбинирование
        all_candidates = []
        for candidate in candidates:
            for blend in blending_candidates:
                combined = {**candidate, **blend}
                all_candidates.append(combined)

        # 4. Veto (отбраковка по ограничениям)
        feasible = self._apply_veto(
            candidates=all_candidates,
            quality_assessment=quality_assessment,
            reliability_assessment=reliability_assessment
        )

        # 5. Оценка вариантов
        if feasible:
            scored = self._score_candidates(feasible)
            ranked = sorted(scored, key=lambda x: x['score'], reverse=True)

            # 6. Выбор топ-1 + альтернативы
            recommended = ranked[0]
            alternatives = ranked[1:4] if len(ranked) > 1 else []
        else:
            scored = []
            ranked = []
            recommended = {}
            alternatives = []

        return OptimizationResult(
            timestamp=pd.Timestamp.now(),
            candidates=all_candidates,
            feasible=feasible,
            ranked=ranked,
            recommended=recommended,
            alternatives=alternatives,
            metrics={
                'num_candidates': len(all_candidates),
                'num_feasible': len(feasible),
                'num_ranked': len(ranked)
            }
        )

    def _apply_veto(
            self,
            candidates: List[Dict[str, float]],
            quality_assessment: Any,
            reliability_assessment: Any
    ) -> List[Dict[str, float]]:
        """
        Отбраковка вариантов по жёстким ограничениям.

        Args:
            candidates: список кандидатов
            quality_assessment: оценка качества
            reliability_assessment: оценка надёжности

        Returns:
            Список допустимых вариантов
        """
        feasible = []

        for candidate in candidates:
            # 1. Проверка серы (сера ≤ 10 мг/кг)
            # TODO: вызвать Quality Agent для прогноза серы
            sulfur_forecast = 8.5  # заглушка
            if sulfur_forecast > 10.0:
                continue  # veto

            # 2. Проверка долей блендинга (сумма = 1.0)
            blend_tags = list(self.blending_fractions.keys())
            blend_sum = sum(candidate.get(tag, 0) for tag in blend_tags)

            if not np.isclose(blend_sum, 1.0, atol=0.01):
                continue  # veto

            # 3. Проверка диапазонов
            if not self._in_ranges(candidate):
                continue  # veto

            # 4. Проверка надёжности (risk_class != 'high')
            # TODO: вызвать Reliability Agent
            risk_class = 'low'  # заглушка
            if risk_class == 'high':
                continue  # veto

            feasible.append(candidate)

        return feasible

    def _in_ranges(self, candidate: Dict[str, float]) -> bool:
        """
        Проверка, что все параметры в диапазонах.

        Args:
            candidate: кандидат

        Returns:
            True, если все параметры в диапазонах
        """
        for tag, value in candidate.items():
            if tag in self.controlled_params:
                param = self.controlled_params[tag]
                if value < param.min or value > param.max:
                    return False

            if tag in self.blending_fractions:
                param = self.blending_fractions[tag]
                if value < param.min or value > param.max:
                    return False

        return True

    def _score_candidates(
            self,
            feasible: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Оценка вариантов по целевой функции.

        Args:
            feasible: допустимые варианты

        Returns:
            Список вариантов с метриками
        """
        scored = []

        for candidate in feasible:
            # Throughput: прогноз выпуска ДТ (т/ч)
            throughput = self._estimate_throughput(candidate)

            # Energy proxy: сумма расходов пара, электроэнергии, водорода
            energy_proxy = self._estimate_energy(candidate)

            # Risk index: от Reliability Agent
            risk_index = self._estimate_risk(candidate)

            # Целевая функция: J = w1*throughput - w2*energy - w3*risk
            score = (
                    self.weights['throughput'] * throughput
                    - self.weights['energy'] * energy_proxy
                    - self.weights['risk'] * risk_index
            )

            scored.append({
                **candidate,
                'throughput': throughput,
                'energy_proxy': energy_proxy,
                'risk_index': risk_index,
                'score': score
            })

        return scored

    def _estimate_throughput(self, candidate: Dict[str, float]) -> float:
        """
        Оценка выпуска ДТ (т/ч).

        Args:
            candidate: вариант

        Returns:
            Throughput (т/ч)
        """
        # Прокси: F9 (расход на гидроочистку)
        return candidate.get('F9', 250.0)

    def _estimate_energy(self, candidate: Dict[str, float]) -> float:
        """
        Оценка энергозатрат (прокси).

        Args:
            candidate: вариант

        Returns:
            Energy proxy
        """
        # Прокси: F5 (пар) + F2 (ВСГ)
        # Используем T6 как прокси для энергозатрат
        t6 = candidate.get('T6', 295.0)
        return (t6 - 290.0) / 15.0  # нормализация в 0..1

    def _estimate_risk(self, candidate: Dict[str, float]) -> float:
        """
        Оценка риска оборудования.

        Args:
            candidate: вариант

        Returns:
            Risk index (0..1)
        """
        # Прокси: отклонение T6 от нормы
        t6 = candidate.get('T6', 295.0)
        t6_norm = self.controlled_params['T6'].current

        risk = abs(t6 - t6_norm) / (self.controlled_params['T6'].max - t6_norm)
        return min(1.0, risk)


# ============================================================================
# CLI для тестирования
# ============================================================================

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Тестирование Optimization Agent')
    parser.add_argument('--num-candidates', type=int, default=50, help='Количество кандидатов')
    args = parser.parse_args()

    agent = OptimizationAgent()

    # Генерация кандидатов
    candidates = agent._generate_candidates(
        num_params=len(agent.controlled_params),
        num_candidates=args.num_candidates
    )

    print(f"Сгенерировано {len(candidates)} кандидатов:")
    for i, candidate in enumerate(candidates[:5]):
        print(f"{i + 1}. {candidate}")

    # Генерация блендинга
    blending = agent._generate_blending_candidates(num_candidates=5)

    print(f"\nСгенерировано {len(blending)} вариантов блендинга:")
    for i, blend in enumerate(blending):
        total = sum(blend.values())
        print(f"{i + 1}. {blend} (сумма={total:.3f})")