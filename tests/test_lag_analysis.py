"""
Модуль: tests/test_lag_analysis.py
Назначение: Автоматические тесты для блока DATA-07:
Анализ динамического запаздывания отклика качества и корректность отчёта lag_report.csv.
"""

import pytest
import os
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.analyze_lags import (
    load_datasets,
    compute_lag_correlations,
    extract_summary_recommendations,
    LAG_MINUTES,
)


@pytest.fixture
def lag_report_path():
    """Путь к артефакту lag_report.csv."""
    path = Path("data/external/lag_report.csv")
    if not path.exists():
        # Если файл еще не создан в окружении теста, запускаем расчёт
        from scripts.analyze_lags import main
        main()
    return path


class TestLagReportArtifact:
    """Тестирование структуры и данных итогового отчёта lag_report.csv."""

    def test_report_exists_and_non_empty(self, lag_report_path):
        assert lag_report_path.exists(), "Файл lag_report.csv не найден"
        assert lag_report_path.stat().st_size > 500, "Файл lag_report.csv пуст или слишком мал"

    def test_required_columns_present(self, lag_report_path):
        df = pd.read_csv(lag_report_path)
        required_cols = ['quality_tag', 'telemetry_tag', 'best_lag_min', 'correlation']
        for col in required_cols:
            assert col in df.columns, f"Обязательная колонка {col} отсутствует в lag_report.csv"

        for lm in LAG_MINUTES:
            assert f'corr_{lm}m' in df.columns, f"Колонка corr_{lm}m отсутствует в lag_report.csv"

    def test_all_dod_quality_targets_present(self, lag_report_path):
        """Проверка выполнения DoD: присутствие S, D15, T50, T90, T95, CFPP."""
        df = pd.read_csv(lag_report_path)
        quality_tags_in_report = set(df['quality_tag'].unique())

        # Сера (S): либо Mg.Sulfur, либо Mass.Sulfur
        assert ('Mg.Sulfur' in quality_tags_in_report) or ('Mass.Sulfur' in quality_tags_in_report), \
            "Показатель серы S отсутствует в lag_report.csv"

        # Плотность
        assert 'D15' in quality_tags_in_report, "Показатель D15 отсутствует в lag_report.csv"

        # Фракционный состав T50, T90, T95
        assert '50%.T' in quality_tags_in_report, "Показатель T50 (50%.T) отсутствует в lag_report.csv"
        assert '90%.T' in quality_tags_in_report, "Показатель T90 (90%.T) отсутствует в lag_report.csv"
        assert '95%.T' in quality_tags_in_report, "Показатель T95 (95%.T) отсутствует в lag_report.csv"

        # Фильтруемость
        assert 'CFPP' in quality_tags_in_report, "Показатель CFPP отсутствует в lag_report.csv"

    def test_lag_values_within_allowed_grid(self, lag_report_path):
        """Значения best_lag_min должны строго принадлежать сетке 10, 30, 60, 90, 120 мин."""
        df = pd.read_csv(lag_report_path)
        allowed_lags = {10, 30, 60, 90, 120}
        actual_lags = set(df['best_lag_min'].unique())
        assert actual_lags.issubset(allowed_lags), \
            f"Найдены некорректные лаги: {actual_lags - allowed_lags}"

    def test_correlation_ranges(self, lag_report_path):
        """Все коэффициенты корреляции должны лежать строго в диапазоне [-1.0, 1.0]."""
        df = pd.read_csv(lag_report_path)
        assert (df['correlation'] >= -1.0).all(), "Обнаружены значения корреляции < -1.0"
        assert (df['correlation'] <= 1.0).all(), "Обнаружены значения корреляции > 1.0"
        for lm in LAG_MINUTES:
            col = f'corr_{lm}m'
            assert (df[col] >= -1.0).all()
            assert (df[col] <= 1.0).all()


class TestLagAlgorithmLogic:
    """Тестирование математической корректности алгоритма поиска запаздывания."""

    def test_synthetic_exact_lag_detection(self):
        """
        Синтетический тест: создаём искусственный отклик с точной задержкой в 30 минут (3 шага)
        и проверяем, что алгоритм безошибочно находит best_lag_min = 30.
        """
        dates = pd.date_range('2023-01-01', periods=100, freq='10min', tz='UTC')
        np.random.seed(42)

        # Случайный процесс телеметрии
        telemetry_signal = np.random.randn(100)
        df_telem = pd.DataFrame({'T_fake': telemetry_signal}, index=dates)

        # Качество: чистый сигнал телеметрии, сдвинутый на 3 шага вперед (запаздывание 30 минут)
        # quality(t) = telemetry(t - 30m)
        quality_signal = np.roll(telemetry_signal, 3)
        quality_signal[:3] = np.nan  # первые точки неопределены

        df_qual = pd.DataFrame({
            'date': dates,
            'tag': 'Fake_Quality',
            'value_quality': quality_signal,
        })

        results = compute_lag_correlations(
            df_quality=df_qual,
            df_telemetry=df_telem,
            quality_tags=['Fake_Quality'],
            telemetry_tags=['T_fake']
        )

        assert len(results) == 1
        row = results.iloc[0]
        assert row['best_lag_min'] == 30, f"Ожидался оптимальный лаг 30 мин, получено {row['best_lag_min']}"
        assert row['correlation'] > 0.95, f"Корреляция должна быть близка к 1.0, получено {row['correlation']}"

    def test_summary_extraction(self):
        """Проверка работы функции extract_summary_recommendations."""
        dummy_data = pd.DataFrame([
            {'quality_tag': 'Q1', 'telemetry_tag': 'T1', 'best_lag_min': 60, 'correlation': 0.8},
            {'quality_tag': 'Q1', 'telemetry_tag': 'T2', 'best_lag_min': 30, 'correlation': 0.4},
            {'quality_tag': 'Q2', 'telemetry_tag': 'T1', 'best_lag_min': 120, 'correlation': -0.7},
        ])
        summary = extract_summary_recommendations(dummy_data)
        assert len(summary) == 2
        q1_row = summary[summary['quality_tag'] == 'Q1'].iloc[0]
        assert q1_row['recommended_lag_min'] == 60
        assert q1_row['primary_telemetry_tag'] == 'T1'


class TestLoadDatasets:
    """Тестирование функции загрузки данных load_datasets."""

    def test_load_datasets_real_files(self):
        q_path = Path("data/processed/telemetry_with_quality.parquet")
        t_path = Path("data/processed/telemetry_clean.parquet")
        if not (q_path.exists() and t_path.exists()):
            pytest.skip("Файлы витрин не найдены")

        # Проверяем вызов как с объектами Path, так и со строками
        df_q, df_t = load_datasets(q_path, str(t_path))

        assert isinstance(df_q, pd.DataFrame)
        assert isinstance(df_t, pd.DataFrame)
        assert len(df_q) > 0
        assert len(df_t) > 0

        # В телеметрии date должна стать DatetimeIndex
        assert isinstance(df_t.index, pd.DatetimeIndex)
        assert df_t.index.tz is not None, "Индекс телеметрии должен иметь таймзону UTC"

        # В качестве date должна быть колонка datetime с UTC
        assert 'date' in df_q.columns
        assert pd.api.types.is_datetime64_any_dtype(df_q['date'])
        assert df_q['date'].dt.tz is not None

        # Даты должны быть строго отсортированы по возрастанию
        assert df_t.index.is_monotonic_increasing, "Телеметрия должна быть строго отсортирована по времени"
        assert df_q['date'].is_monotonic_increasing, "Качество должно быть строго отсортировано по времени"

    def test_load_datasets_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_datasets("non_existent_quality.parquet", "data/processed/telemetry_clean.parquet")

        with pytest.raises(FileNotFoundError):
            load_datasets("data/processed/telemetry_with_quality.parquet", "non_existent_telemetry.parquet")

