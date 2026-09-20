"""
Тесты для блока DATA-02: Детекция выбросов Z-score и создание масок качества.
Файл: tests/test_clean_telemetry.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.etl.clean_telemetry import (
    detect_outliers_zscore,
    create_quality_masks,
    clean_telemetry_outliers
)


@pytest.fixture
def synthetic_telemetry() -> pd.DataFrame:
    """Генерирует 100 временных точек с нормальными данными, спайком, пропуском и константой."""
    dates = pd.date_range("2026-01-01", periods=100, freq="10min", tz="UTC")
    np.random.seed(42)

    # 1. Сигнал с известным выбросом (спайком) на 50-м шаге
    normal_signal = 100.0 + np.random.normal(0, 1.0, size=100)
    normal_signal[50] = 500.0  # Огромный спайк (Z-score > 50)
    normal_signal[70] = np.nan # Пропуск

    # 2. Константный сигнал (std = 0)
    const_signal = np.full(100, 42.0)

    df = pd.DataFrame({
        "date": dates,
        "source": ["AVT"] * 100,
        "tag_spike": normal_signal,
        "tag_const": const_signal
    })
    return df


def test_detect_outliers_known_spike(synthetic_telemetry: pd.DataFrame):
    """Проверка: алгоритм обнаруживает искусственный выброс на индексе 50."""
    outliers = detect_outliers_zscore(synthetic_telemetry, window=20, threshold=4.0)

    assert "tag_spike" in outliers.columns
    assert "date" not in outliers.columns
    assert "source" not in outliers.columns

    # Индекс 50 должен быть помечен как выброс
    assert outliers.loc[50, "tag_spike"] == True
    # Нормальные точки до выброса не должны быть выбросами
    assert outliers.loc[10:40, "tag_spike"].sum() == 0


def test_constant_signal_zero_std_handling(synthetic_telemetry: pd.DataFrame):
    """Проверка: для константного сигнала (std=0) деление на ноль не вызывает ошибок."""
    outliers = detect_outliers_zscore(synthetic_telemetry, window=20, threshold=4.0)

    assert "tag_const" in outliers.columns
    # На константном сигнале не должно быть ложных выбросов
    assert outliers["tag_const"].sum() == 0


def test_create_quality_masks(synthetic_telemetry: pd.DataFrame):
    """Проверка: маска равна 1 для нормы и 0 для выброса и NaN."""
    outliers = detect_outliers_zscore(synthetic_telemetry, window=20, threshold=4.0)
    masks = create_quality_masks(synthetic_telemetry, outliers, date_col="date")

    assert "date" in masks.columns
    assert "tag_spike_mask" in masks.columns
    assert "tag_const_mask" in masks.columns

    # Нормальная точка -> mask = 1
    assert masks.loc[10, "tag_spike_mask"] == 1
    # Спайк на индексе 50 -> mask = 0
    assert masks.loc[50, "tag_spike_mask"] == 0
    # Пропуск (NaN) на индексе 70 -> mask = 0
    assert masks.loc[70, "tag_spike_mask"] == 0
    # Константа -> все mask = 1
    assert (masks["tag_const_mask"] == 1).all()


def test_clean_telemetry_outliers_replaces_with_nan(synthetic_telemetry: pd.DataFrame):
    """Проверка: замена выбросов на NaN в очищенном датафрейме."""
    outliers = detect_outliers_zscore(synthetic_telemetry, window=20, threshold=4.0)
    cleaned = clean_telemetry_outliers(synthetic_telemetry, outliers)

    # Выброс заменен на NaN
    assert np.isnan(cleaned.loc[50, "tag_spike"])
    # Нормальные точки не повреждены
    assert cleaned.loc[10, "tag_spike"] == synthetic_telemetry.loc[10, "tag_spike"]
    # Колонка даты сохранена
    assert (cleaned["date"] == synthetic_telemetry["date"]).all()


def test_real_clean_parquet_dod():
    """Интеграционный тест: проверка критериев DoD на сгенерированных файлах проекта."""
    clean_path = Path("data/processed/telemetry_clean.parquet")
    masks_path = Path("data/processed/telemetry_masks.parquet")

    if not clean_path.exists() or not masks_path.exists():
        pytest.skip("Файлы data/processed/telemetry_clean*.parquet ещё не сгенерированы. Запустите scripts/clean_telemetry.py.")

    df_clean = pd.read_parquet(clean_path)
    df_masks = pd.read_parquet(masks_path)

    # 1. Проверка размерности
    assert len(df_clean) == 189_217
    assert len(df_masks) == 189_217

    # 2. Проверка колонок масок
    mask_cols = [c for c in df_masks.columns if c != "date"]
    assert len(mask_cols) > 0
    assert all(c.endswith("_mask") for c in mask_cols)

    # 3. Проверка критерия DoD: доля mask=0 не превышает 5%
    total_mask_points = df_masks.shape[0] * len(mask_cols)
    total_zero_masks = (df_masks[mask_cols] == 0).sum().sum()
    zero_ratio = (total_zero_masks / total_mask_points) * 100

    assert zero_ratio <= 5.0, f"Критерий DoD нарушен: доля mask=0 составляет {zero_ratio:.2f}% > 5.0%"
