"""
Модуль тестирования ETL-пайплайна (Блок: Tests-01).

Проверяет:
- Загрузку телеметрии и удаление технических колонок Unnamed (DATA-01)
- Детекцию аномальных выбросов z-score и замену на NaN/маски (DATA-02)
- Временную синхронизацию merge_asof телеметрии и качества, расчет age_min и stale (DATA-04)
- Пограничные случаи: пустые данные, все NaN, приоритет LIMS над PAK.
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.etl.load_telemetry import load_telemetry_csv, save_telemetry_parquet
from src.etl.clean_telemetry import (
    detect_outliers_zscore,
    create_quality_masks,
    clean_telemetry_outliers
)
from src.etl.sync_data import sync_telemetry_quality, compute_age_min


# ==============================================================================
# 1. ТЕСТЫ ЗАГРУЗКИ ТЕЛЕМЕТРИИ (DATA-01)
# ==============================================================================

def test_load_telemetry_shape(tmp_path):
    """
    Проверка загрузки CSV телеметрии:
    - очистка от колонок 'Unnamed:*'
    - корректный размер строк (~189 тыс. строк на реальном датасете)
    """
    real_csv = Path('data/raw/avt_tags.csv')
    if real_csv.exists():
        df = load_telemetry_csv(str(real_csv), 'AVT')
        assert len(df) >= 189000 or len(df) == 189000
        assert 'Unnamed:' not in df.columns
        for c in df.columns:
            assert not str(c).startswith('Unnamed')
        assert 'date' in df.columns
        assert df['source'].iloc[0] == 'AVT'
    else:
        # Fallback мок при отсутствии сырого файла в CI/sandbox
        mock_file = tmp_path / "mock_avt.csv"
        mock_df = pd.DataFrame({
            'Unnamed: 0': range(10),
            'date': pd.date_range('2026-01-01', periods=10, freq='10min'),
            'T6': np.linspace(350, 360, 10),
            'Unnamed: 2': [np.nan] * 10
        })
        mock_df.to_csv(mock_file, index=False)
        df = load_telemetry_csv(str(mock_file), 'AVT')
        assert len(df) == 10
        assert 'Unnamed:' not in df.columns
        for c in df.columns:
            assert not str(c).startswith('Unnamed')


def test_load_telemetry_not_found():
    """Проверка ошибки FileNotFoundError при несуществующем пути."""
    with pytest.raises(FileNotFoundError):
        load_telemetry_csv('data/raw/non_existent_file_12345.csv', 'AVT')


def test_load_telemetry_missing_date_column(tmp_path):
    """Проверка ошибки при отсутствии колонки date."""
    f = tmp_path / "no_date.csv"
    pd.DataFrame({'T6': [300, 301]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="отсутствует обязательная колонка 'date'"):
        load_telemetry_csv(str(f), 'AVT')


def test_save_telemetry_parquet(tmp_path):
    """Проверка корректного сохранения в формат Parquet."""
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=5, freq='10min', tz='UTC'),
        'T6': [350.0, 351.0, 352.0, 353.0, 354.0],
        'source': ['AVT'] * 5
    })
    out_parquet = tmp_path / "telemetry.parquet"
    save_telemetry_parquet(df, str(out_parquet))
    assert out_parquet.exists()
    loaded = pd.read_parquet(out_parquet)
    assert len(loaded) == 5
    assert list(loaded.columns) == ['date', 'T6', 'source']


# ==============================================================================
# 2. ТЕСТЫ ОЧИСТКИ И ДЕТЕКЦИИ ВЫБРОСОВ (DATA-02)
# ==============================================================================

def test_clean_telemetry_no_outliers():
    """
    Проверка детекции выбросов:
    df = pd.DataFrame({'T6': [300, 301, 302, 1000]}) # 1000 — выброс
    cleaned = detect_outliers_zscore(df, window=3, threshold=4.0)
    assert cleaned['T6'].iloc[-1] != 1000 # заменено на NaN / идентифицирован выброс
    """
    df = pd.DataFrame({'T6': [300, 301, 302, 1000]})  # 1000 — выброс
    cleaned = detect_outliers_zscore(df, window=3, threshold=4.0)
    # Выброс зафиксирован (значение != 1000, в булевой маске True != 1000)
    assert cleaned['T6'].iloc[-1] != 1000
    assert bool(cleaned['T6'].iloc[-1]) is True
    assert bool(cleaned['T6'].iloc[0]) is False

    # Проверка очищенного датафрейма через clean_telemetry_outliers
    cleaned_df = clean_telemetry_outliers(df, cleaned)
    assert np.isnan(cleaned_df['T6'].iloc[-1])
    assert cleaned_df['T6'].iloc[0] == 300


def test_clean_telemetry_masks():
    """Проверка генерации бинарных масок качества 0/1."""
    df = pd.DataFrame({'T6': [300.0, 301.0, np.nan, 1000.0]})
    outliers = detect_outliers_zscore(df, window=3, threshold=3.0)
    masks = create_quality_masks(df, outliers)

    assert 'T6_mask' in masks.columns
    # Первые две точки качественные (1)
    assert masks['T6_mask'].iloc[0] == 1
    assert masks['T6_mask'].iloc[1] == 1
    # NaN и выброс должны получить маску 0
    assert masks['T6_mask'].iloc[2] == 0
    assert masks['T6_mask'].iloc[3] == 0


def test_clean_telemetry_all_nan():
    """Edge case: колонка целиком из NaN."""
    df = pd.DataFrame({'T6': [np.nan, np.nan, np.nan]})
    outliers = detect_outliers_zscore(df, window=3, threshold=4.0)
    # NaN не должен вызывать падения алгоритма
    assert len(outliers) == 3
    assert not outliers['T6'].any()


# ==============================================================================
# 3. ТЕСТЫ СИНХРОНИЗАЦИИ ТЕЛЕМЕТРИИ И КАЧЕСТВА (DATA-04)
# ==============================================================================

def test_sync_age_min():
    """
    Проверка расчета возраста замера качества:
    telemetry = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=10, freq='10min')})
    quality = pd.DataFrame({'timestamp': [pd.Timestamp('2026-01-01 00:00')]})
    synced = sync_telemetry_quality(telemetry, quality)
    assert 'age_min' in synced.columns
    assert synced['age_min'].iloc[0] == 0
    """
    telemetry = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=10, freq='10min')})
    quality = pd.DataFrame({'timestamp': [pd.Timestamp('2026-01-01 00:00')]})
    synced = sync_telemetry_quality(telemetry, quality)
    assert 'age_min' in synced.columns
    assert synced['age_min'].iloc[0] == 0
    assert synced['age_min'].iloc[1] == 10.0


def test_sync_quality_lims_priority():
    """
    Проверка приоритета источников: ЛИМС подавляет ПАК при одновременном наличии.
    """
    telemetry = pd.DataFrame({
        'date': pd.date_range('2026-01-01 00:00', periods=3, freq='10min'),
        'T6': [360, 360, 360]
    })
    quality = pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2026-01-01 00:00'),
            pd.Timestamp('2026-01-01 00:00')
        ],
        'tag': ['Sulfur', 'Sulfur'],
        'value': [7.5, 9.2],
        'source': ['LIMS', 'PAK']
    })
    synced = sync_telemetry_quality(telemetry, quality, tolerance_lims=60, tolerance_pak=15)
    # Для первой точки должно быть выбрано значение LIMS (7.5)
    row0 = synced.iloc[0]
    assert row0['value_quality'] == 7.5
    assert row0['source'] == 'LIMS'


def test_sync_quality_stale_flag():
    """Проверка выставления флага stale=True при age_min > 240 минут."""
    df = pd.DataFrame({
        'date': [pd.Timestamp('2026-01-01 05:00:00', tz='UTC')],
        'timestamp': [pd.Timestamp('2026-01-01 00:00:00', tz='UTC')]
    })
    res = compute_age_min(df)
    assert res['age_min'].iloc[0] == 300.0  # 5 часов = 300 мин
    assert bool(res['stale'].iloc[0]) is True


def test_sync_telemetry_missing_columns():
    """Проверка выброса исключения при отсутствии обязательных колонок даты."""
    with pytest.raises(ValueError, match="должна содержать колонку 'date'"):
        sync_telemetry_quality(pd.DataFrame({'T6': [360]}), pd.DataFrame({'timestamp': [pd.Timestamp.now()]}))

    with pytest.raises(ValueError, match="должна содержать колонку 'timestamp'"):
        sync_telemetry_quality(pd.DataFrame({'date': [pd.Timestamp.now()]}), pd.DataFrame({'value': [10]}))
