"""
Модуль проверки входных данных (полнота, актуальность, согласованность).

Используется Orchestrator перед вызовом агентов.
Если проверка не пройдена → отказ от рекомендации с объяснением.

Критерии:
1. Полнота: пропуски < 30% для ключевых тегов
2. Актуальность: age_min < 120 мин для ЛИМС/ПАК
3. Согласованность: ЛИМС и ПАК не противоречат (разница < 20%)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Результат проверки входа."""
    is_valid: bool
    checks: Dict[str, bool]
    reasons: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]


class InputValidator:
    """
    Проверка входных данных перед вызовом агентов.

    Проверки:
    1. Полнота (пропуски < 30%)
    2. Актуальность (age_min < 120 мин)
    3. Согласованность (ЛИМС/ПАК не противоречат)
    """

    def __init__(
            self,
            max_missing_ratio: float = 0.3,
            max_age_min: int = 120,
            consistency_threshold: float = 0.2,
            critical_tags: Optional[List[str]] = None
    ):
        """
        Инициализация.

        Args:
            max_missing_ratio: максимальная доля пропусков (30%)
            max_age_min: максимальный возраст анализа (120 мин)
            consistency_threshold: порог согласованности (20%)
            critical_tags: критические теги (если None, используются по умолчанию)
        """
        self.max_missing_ratio = max_missing_ratio
        self.max_age_min = max_age_min
        self.consistency_threshold = consistency_threshold

        # Критические теги телеметрии для проверки полноты
        if critical_tags is None:
            self.critical_tags = [
                'T6',  # Температура реактора
                'F2_F26_ratio',  # ВСГ/сырьё
                'F9',  # Расход на гидроочистку
            ]
        else:
            self.critical_tags = critical_tags

        logger.info(
            f"InputValidator инициализирован: "
            f"max_missing_ratio={max_missing_ratio}, "
            f"max_age_min={max_age_min}, "
            f"consistency_threshold={consistency_threshold}"
        )

    def validate(
            self,
            telemetry: pd.DataFrame,
            quality_data: pd.DataFrame
    ) -> ValidationResult:
        """
        Комплексная проверка входа.

        Args:
            telemetry: телеметрия (свежая, за последние 24 часа)
            quality_data: качество (ЛИМС/ПАК, с age_min)

        Returns:
            ValidationResult
        """
        logger.info("Проверка входных данных")

        checks = {}
        reasons = []
        warnings = []
        metrics = {}

        # ================================================================
        # 1. ПРОВЕРКА ПОЛНОТЫ (пропуски < 30%)
        # ================================================================

        logger.info("Проверка полноты данных")

        completeness_ok, completeness_reasons, completeness_metrics = self._check_completeness(telemetry)

        checks['completeness'] = completeness_ok
        reasons.extend(completeness_reasons)
        metrics.update(completeness_metrics)

        if not completeness_ok:
            logger.warning(f"Проверка полноты не пройдена: {completeness_reasons}")

        # ================================================================
        # 2. ПРОВЕРКА АКТУАЛЬНОСТИ (age_min < 120 мин)
        # ================================================================

        logger.info("Проверка актуальности данных")

        freshness_ok, freshness_reasons, freshness_metrics = self._check_freshness(quality_data)

        checks['freshness'] = freshness_ok
        reasons.extend(freshness_reasons)
        metrics.update(freshness_metrics)

        if not freshness_ok:
            logger.warning(f"Проверка актуальности не пройдена: {freshness_reasons}")

        # ================================================================
        # 3. ПРОВЕРКА СОГЛАСОВАННОСТИ (ЛИМС/ПАК не противоречат)
        # ================================================================

        logger.info("Проверка согласованности данных")

        consistency_ok, consistency_reasons, consistency_metrics = self._check_consistency(quality_data)

        checks['consistency'] = consistency_ok
        reasons.extend(consistency_reasons)
        metrics.update(consistency_metrics)

        if not consistency_ok:
            logger.warning(f"Проверка согласованности не пройдена: {consistency_reasons}")

        # ================================================================
        # 4. ДОПОЛНИТЕЛЬНЫЕ ПРОВЕРКИ (выбросы, стабильность)
        # ================================================================

        logger.info("Дополнительные проверки")

        outliers_ok, outliers_reasons, outliers_metrics = self._check_outliers(telemetry)

        checks['outliers'] = outliers_ok
        reasons.extend(outliers_reasons)
        metrics.update(outliers_metrics)

        if not outliers_ok:
            warnings.extend(outliers_reasons)
            logger.warning(f"Обнаружены выбросы: {outliers_reasons}")

        # ================================================================
        # ИТОГ: все проверки пройдены?
        # ================================================================

        is_valid = all(checks.values())

        logger.info(
            f"Проверка завершена: is_valid={is_valid}, "
            f"checks={checks}, "
            f"reasons={len(reasons)}, "
            f"warnings={len(warnings)}"
        )

        return ValidationResult(
            is_valid=is_valid,
            checks=checks,
            reasons=reasons,
            warnings=warnings,
            metrics=metrics
        )

    def _check_completeness(
            self,
            telemetry: pd.DataFrame
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Проверка полноты данных (пропуски < 30%).

        Args:
            telemetry: телеметрия

        Returns:
            (OK, список причин, метрики)
        """
        reasons = []
        metrics = {}

        if telemetry is None or len(telemetry) == 0:
            return False, ["Телеметрия пуста"], {'missing_ratio': 1.0}

        # 1. Общая доля пропусков
        total_missing = telemetry.isnull().sum().sum()
        total_cells = telemetry.size
        missing_ratio = total_missing / total_cells if total_cells > 0 else 1.0

        metrics['total_missing'] = int(total_missing)
        metrics['total_cells'] = int(total_cells)
        metrics['missing_ratio'] = float(missing_ratio)

        if missing_ratio > self.max_missing_ratio:
            reasons.append(f"Общая доля пропусков={missing_ratio:.2%} > {self.max_missing_ratio:.2%}")

        # 2. Проверка критических тегов
        critical_missing = []

        for tag in self.critical_tags:
            if tag not in telemetry.columns:
                critical_missing.append(f"Критический тег отсутствует: {tag}")
                continue

            tag_missing = telemetry[tag].isnull().sum()
            tag_total = len(telemetry)
            tag_missing_ratio = tag_missing / tag_total if tag_total > 0 else 1.0

            metrics[f'{tag}_missing_ratio'] = float(tag_missing_ratio)

            if tag_missing_ratio > self.max_missing_ratio:
                critical_missing.append(f"Пропуски по {tag}={tag_missing_ratio:.2%} > {self.max_missing_ratio:.2%}")

        if critical_missing:
            reasons.extend(critical_missing)

        is_ok = len(reasons) == 0

        return is_ok, reasons, metrics

    def _check_freshness(
            self,
            quality_data: pd.DataFrame
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Проверка актуальности данных (age_min < 120 мин).

        Args:
            quality_data: качество (с age_min)

        Returns:
            (OK, список причин, метрики)
        """
        reasons = []
        metrics = {}

        if quality_data is None or len(quality_data) == 0:
            return False, ["Данные о качестве отсутствуют"], {'max_age_min': None}

        # 1. Максимальный возраст анализа
        if 'age_min' not in quality_data.columns:
            return False, ["Колонка age_min отсутствует"], {'max_age_min': None}

        max_age = quality_data['age_min'].max()
        min_age = quality_data['age_min'].min()
        avg_age = quality_data['age_min'].mean()

        metrics['max_age_min'] = float(max_age) if pd.notna(max_age) else None
        metrics['min_age_min'] = float(min_age) if pd.notna(min_age) else None
        metrics['avg_age_min'] = float(avg_age) if pd.notna(avg_age) else None

        if pd.isna(max_age):
            reasons.append("age_min содержит только NaN")
        elif max_age > self.max_age_min:
            reasons.append(f"Максимальный возраст анализа={max_age:.0f} мин > {self.max_age_min} мин")

        # 2. Проверка по ключевым показателям
        critical_tags = ['Sulfur', 'D15', 'T95', 'CFPP']

        for tag in critical_tags:
            tag_data = quality_data[quality_data.get('tag', '') == tag]

            if len(tag_data) == 0:
                continue

            tag_max_age = tag_data['age_min'].max()

            if pd.notna(tag_max_age) and tag_max_age > self.max_age_min:
                reasons.append(f"Возраст {tag}={tag_max_age:.0f} мин > {self.max_age_min} мин")

        is_ok = len(reasons) == 0

        return is_ok, reasons, metrics

    def _check_consistency(
            self,
            quality_data: pd.DataFrame
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Проверка согласованности (ЛИМС/ПАК не противоречат).

        Args:
            quality_data: качество (с source='LIMS' или 'PAK')

        Returns:
            (OK, список причин, метрики)
        """
        reasons = []
        metrics = {}

        if quality_data is None or len(quality_data) == 0:
            return True, [], {}  # Нет данных → нет противоречий

        # 1. Проверка по тегам с ЛИМС и ПАК одновременно
        if 'tag' not in quality_data.columns or 'source' not in quality_data.columns:
            return True, [], {}  # Нет структуры для проверки

        # Группировка по tag
        inconsistencies = []

        for tag in quality_data['tag'].unique():
            tag_data = quality_data[quality_data['tag'] == tag]

            lims_data = tag_data[tag_data['source'] == 'LIMS']
            pak_data = tag_data[tag_data['source'] == 'PAK']

            if len(lims_data) == 0 or len(pak_data) == 0:
                continue  # Нет обоих источников

            # Сравнение последних значений
            lims_value = lims_data.sort_values('timestamp').iloc[-1]['value']
            pak_value = pak_data.sort_values('timestamp').iloc[-1]['value']

            # Относительная разница
            if abs(lims_value) > 1e-6:
                rel_diff = abs(lims_value - pak_value) / abs(lims_value)
            else:
                rel_diff = abs(lims_value - pak_value)

            metrics[f'{tag}_lims_value'] = float(lims_value)
            metrics[f'{tag}_pak_value'] = float(pak_value)
            metrics[f'{tag}_rel_diff'] = float(rel_diff)

            if rel_diff > self.consistency_threshold:
                inconsistencies.append(
                    f"{tag}: ЛИМС={lims_value:.2f}, ПАК={pak_value:.2f}, "
                    f"разница={rel_diff:.2%} > {self.consistency_threshold:.2%}"
                )

        if inconsistencies:
            reasons.extend(inconsistencies)

        metrics['num_inconsistencies'] = len(inconsistencies)

        is_ok = len(reasons) == 0

        return is_ok, reasons, metrics

    def _check_outliers(
            self,
            telemetry: pd.DataFrame
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Проверка на выбросы (z-score > 4).

        Args:
            telemetry: телеметрия

        Returns:
            (OK, список причин, метрики)
        """
        reasons = []
        metrics = {}

        if telemetry is None or len(telemetry) == 0:
            return True, [], {}

        # Проверка по критическим тегам
        outliers_found = []

        for tag in self.critical_tags:
            if tag not in telemetry.columns:
                continue

            values = telemetry[tag].dropna()

            if len(values) < 10:
                continue  # Мало данных

            mean = values.mean()
            std = values.std()

            if std < 1e-6:
                continue  # Нет разброса

            z_scores = np.abs((values - mean) / std)
            max_z = z_scores.max()
            num_outliers = (z_scores > 4.0).sum()

            metrics[f'{tag}_max_zscore'] = float(max_z)
            metrics[f'{tag}_num_outliers'] = int(num_outliers)

            if num_outliers > 0:
                outliers_found.append(f"{tag}: {num_outliers} выбросов (max z={max_z:.1f})")

        if outliers_found:
            reasons.extend(outliers_found)

        metrics['total_outliers'] = len(outliers_found)

        # Выбросы — не veto, а warning
        is_ok = True

        return is_ok, reasons, metrics


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    validator = InputValidator()

    # Тест 1: Все OK
    print("\n" + "=" * 80)
    print("Тест 1: Все OK")
    print("=" * 80)

    telemetry_ok = pd.DataFrame({
        'T6': np.random.normal(295, 2, 100),
        'F9': np.random.normal(250, 10, 100),
        'F2_F26_ratio': np.random.normal(0.85, 0.02, 100),
    })

    quality_ok = pd.DataFrame({
        'tag': ['Sulfur', 'D15', 'Sulfur'],
        'value': [8.5, 835.0, 8.7],
        'source': ['LIMS', 'PAK', 'PAK'],
        'age_min': [45, 30, 50],
        'timestamp': pd.date_range('2026-01-01', periods=3, freq='1h')
    })

    result = validator.validate(telemetry_ok, quality_ok)

    print(f"is_valid: {result.is_valid}")
    print(f"checks: {result.checks}")
    print(f"reasons: {result.reasons}")
    print(f"warnings: {result.warnings}")
    print(f"metrics: {result.metrics}")

    # Тест 2: Пропуски > 30%
    print("\n" + "=" * 80)
    print("Тест 2: Пропуски > 30%")
    print("=" * 80)

    telemetry_missing = pd.DataFrame({
        'T6': [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'F9': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    })

    result = validator.validate(telemetry_missing, quality_ok)

    print(f"is_valid: {result.is_valid}")
    print(f"checks: {result.checks}")
    print(f"reasons: {result.reasons}")

    # Тест 3: age_min > 120
    print("\n" + "=" * 80)
    print("Тест 3: age_min > 120")
    print("=" * 80)

    quality_stale = pd.DataFrame({
        'tag': ['Sulfur', 'D15'],
        'value': [8.5, 835.0],
        'source': ['LIMS', 'PAK'],
        'age_min': [180, 150],
        'timestamp': pd.date_range('2026-01-01', periods=2, freq='1h')
    })

    result = validator.validate(telemetry_ok, quality_stale)

    print(f"is_valid: {result.is_valid}")
    print(f"checks: {result.checks}")
    print(f"reasons: {result.reasons}")

    # Тест 4: ЛИМС/ПАК противоречат
    print("\n" + "=" * 80)
    print("Тест 4: ЛИМС/ПАК противоречат")
    print("=" * 80)

    quality_inconsistent = pd.DataFrame({
        'tag': ['Sulfur', 'Sulfur'],
        'value': [8.5, 12.0],  # Разница > 20%
        'source': ['LIMS', 'PAK'],
        'age_min': [45, 50],
        'timestamp': pd.date_range('2026-01-01', periods=2, freq='1h')
    })

    result = validator.validate(telemetry_ok, quality_inconsistent)

    print(f"is_valid: {result.is_valid}")
    print(f"checks: {result.checks}")
    print(f"reasons: {result.reasons}")