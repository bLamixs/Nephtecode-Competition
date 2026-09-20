"""
Тесты для блока DATA-04: Временная синхронизация merge_asof, приоритеты качества и age_min.
Файл: tests/test_sync_data.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.etl.sync_data import sync_telemetry_quality, compute_age_min


@pytest.fixture
def mock_sync_inputs():
    """Создаёт тестовые датафреймы телеметрии и качества для проверки логики приоритетов и tolerance."""
    # Телеметрия: 4 точки с шагом 10 минут
    tel_dates = pd.date_range("2026-01-01 12:00:00", periods=4, freq="10min", tz="UTC")
    telemetry = pd.DataFrame({
        "date": tel_dates,
        "T6": [290.0, 291.0, 292.0, 293.0],
        "Mg.Sulfur": [np.nan, np.nan, np.nan, np.nan] # Тег качества в телеметрии
    })

    # Качество:
    # 1. ЛИМС на 12:00 со значением 7.5
    # 2. ПАК на 12:00 со значением 9.0 (коллизия с ЛИМС на 12:00)
    # 3. ПАК на 12:15 со значением 9.2 (попадает в телеметрию на 12:20, окно 15 мин)
    quality = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2026-01-01 12:00:00", tz="UTC"),
            pd.Timestamp("2026-01-01 12:00:00", tz="UTC"),
            pd.Timestamp("2026-01-01 12:15:00", tz="UTC"),
        ],
        "tag": ["Mg.Sulfur", "Mg.Sulfur", "Mg.Sulfur"],
        "value": [7.5, 9.0, 9.2],
        "source": ["LIMS", "PAK", "PAK"],
        "sample_point": ["24-2000", "24-2000", "24-2000"],
        "unit": ["мг/кг", "ppm", "ppm"]
    })

    return telemetry, quality


def test_priority_lims_over_pak(mock_sync_inputs):
    """Проверка: если на одну дату есть и ЛИМС, и ПАК, ЛИМС имеет абсолютный приоритет."""
    telemetry, quality = mock_sync_inputs
    synced = sync_telemetry_quality(telemetry, quality, tolerance_lims=60, tolerance_pak=15)

    row_1200 = synced[synced["date"] == pd.Timestamp("2026-01-01 12:00:00", tz="UTC")].iloc[0]
    
    # На 12:00 должен выбраться ЛИМС (7.5), а не ПАК (9.0)
    assert row_1200["source"] == "LIMS"
    assert row_1200["value_quality"] == 7.5
    assert row_1200["age_min"] == 0.0
    assert row_1200["stale"] == False


def test_pak_fallback_when_lims_absent(mock_sync_inputs):
    """Проверка: если свежего ЛИМС нет, система использует оперативный ПАК."""
    telemetry, quality = mock_sync_inputs
    # Исключаем ЛИМС
    quality_only_pak = quality[quality["source"] == "PAK"].copy()
    synced = sync_telemetry_quality(telemetry, quality_only_pak, tolerance_lims=60, tolerance_pak=15)

    row_1200 = synced[synced["date"] == pd.Timestamp("2026-01-01 12:00:00", tz="UTC")].iloc[0]
    assert row_1200["source"] == "PAK"
    assert row_1200["value_quality"] == 9.0


def test_tolerance_limits():
    """Проверка: замер вне окна допуска (tolerance) не подтягивается."""
    # Телеметрия в 13:00
    telemetry = pd.DataFrame({"date": [pd.Timestamp("2026-01-01 13:00:00", tz="UTC")]})
    
    # Замер ЛИМС был в 11:30 (90 минут назад > tolerance_lims=60 мин)
    quality = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01 11:30:00", tz="UTC")],
        "tag": ["Mg.Sulfur"],
        "value": [8.0],
        "source": ["LIMS"],
        "sample_point": ["24-2000"],
        "unit": ["мг/кг"]
    })

    synced = sync_telemetry_quality(telemetry, quality, tolerance_lims=60, tolerance_pak=15)
    row = synced.iloc[0]
    # Значение не должно подтянуться из-за превышения tolerance
    assert np.isnan(row["value_quality"])
    assert pd.isna(row["source"])
    assert row["stale"] == True


def test_compute_age_min_and_stale():
    """Проверка: расчет age_min и простановка флага stale при возрасте > 240 минут."""
    df = pd.DataFrame({
        "date": [
            pd.Timestamp("2026-01-01 12:00:00", tz="UTC"),
            pd.Timestamp("2026-01-01 17:00:00", tz="UTC"), # 5 часов позже (300 мин)
            pd.Timestamp("2026-01-01 18:00:00", tz="UTC")
        ],
        "timestamp": [
            pd.Timestamp("2026-01-01 11:30:00", tz="UTC"), # 30 мин назад
            pd.Timestamp("2026-01-01 12:00:00", tz="UTC"), # 300 мин назад
            pd.NaT                                         # нет замера
        ]
    })

    res = compute_age_min(df, date_col="date", ts_col="timestamp")

    # 1. 30 минут -> свежий анализ
    assert res.loc[0, "age_min"] == 30.0
    assert res.loc[0, "stale"] == False

    # 2. 300 минут (> 240) -> устаревший анализ
    assert res.loc[1, "age_min"] == 300.0
    assert res.loc[1, "stale"] == True

    # 3. Нет замера -> stale
    assert np.isnan(res.loc[2, "age_min"])
    assert res.loc[2, "stale"] == True


def test_real_telemetry_with_quality_dod():
    """Интеграционный тест: проверка критериев DoD на сгенерированном telemetry_with_quality.parquet."""
    parquet_path = Path("data/processed/telemetry_with_quality.parquet")
    if not parquet_path.exists():
        pytest.skip("Файл data/processed/telemetry_with_quality.parquet ещё не создан. Запустите scripts/sync_data.py.")

    df = pd.read_parquet(parquet_path)

    # 1. Проверка наличия обязательных колонок
    expected_cols = ["date", "tag", "value_telemetry", "value_quality", "source", "age_min", "stale"]
    assert list(df.columns) == expected_cols

    # 2. Проверка типов
    assert "UTC" in str(df["date"].dtype)
    assert pd.api.types.is_numeric_dtype(df["age_min"])
    assert pd.api.types.is_bool_dtype(df["stale"])

    # 3. Ключевой критерий DoD: нет строк с age_min > 240 без флага stale=True!
    invalid_stale_rows = (df["age_min"] > 240.0) & (~df["stale"])
    assert not invalid_stale_rows.any(), f"Найдено {invalid_stale_rows.sum()} строк с age_min > 240, где stale != True!"

    # 4. Проверка присутствия обоих источников качества
    valid_sources = df["source"].dropna().unique().tolist()
    assert "LIMS" in valid_sources
    assert "PAK" in valid_sources
