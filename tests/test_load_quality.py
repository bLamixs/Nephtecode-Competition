"""
Тесты для блока DATA-03: Загрузка данных качества ЛИМС и ПАК в long-format.
Файл: tests/test_load_quality.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.etl.load_quality import parse_lims_xlsx, parse_pak_xlsx, load_all_quality_data


@pytest.fixture
def mock_lims_excel(tmp_path: Path) -> Path:
    """Создаёт тестовый файл Excel со структурой ЛИМС."""
    file_path = tmp_path / "mock_lims.xlsx"
    
    # 2 блока колонок: CFPP (0, 1) и D15 (2, 3)
    data = [
        ["Точка 1", np.nan, "Точка 2", np.nan],                # Строка 0: Точки
        ["CFPP", np.nan, "D15", np.nan],                      # Строка 1: Теги
        ["°С", np.nan, "кг/м3", np.nan],                      # Строка 2: Единицы
        ["Кол-во: 2", np.nan, "Кол-во: 2", np.nan],           # Строка 3: Мета
        ["2026-01-01 08:00:00", 5.0, "2026-01-01 09:00:00", 835.2], # Строка 4: Замер 1
        ["2026-01-02 08:00:00", 4.5, "2026-01-02 09:00:00", 836.0], # Строка 5: Замер 2
    ]
    df = pd.DataFrame(data)
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Лист1", index=False, header=False)
    return file_path


@pytest.fixture
def mock_pak_excel(tmp_path: Path) -> Path:
    """Создаёт тестовый файл Excel со структурой ПАК."""
    file_path = tmp_path / "mock_pak.xlsx"
    
    # Колонки: Сера (0, 1), разделитель (2), Плотность (3, 4)
    data = {
        "24-2000:Mg.Sulfur": ["ppm", "2026-01-01 00:00:00", "2026-01-01 00:10:00"],
        "Unnamed: 1": [np.nan, 8.5, 9.1],
        "Unnamed: 2": [np.nan, np.nan, np.nan],
        "24-2000:D15": ["кг/м3", "2026-01-01 00:00:00", "2026-01-01 00:10:00"],
        "Unnamed: 4": [np.nan, 832.0, 831.8],
    }
    df = pd.DataFrame(data)
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Лист1", index=False)
    return file_path


def test_parse_lims_mock(mock_lims_excel: Path):
    """Проверка: парсинг ЛИМС преобразует блоки в long-format."""
    df = parse_lims_xlsx(str(mock_lims_excel))

    expected_cols = ["timestamp", "tag", "value", "source", "sample_point", "unit"]
    assert all(c in df.columns for c in expected_cols)
    assert len(df) == 4 # 2 точки * 2 замера
    assert (df["source"] == "LIMS").all()
    assert set(df["tag"]) == {"CFPP", "D15"}
    assert "UTC" in str(df["timestamp"].dtype)


def test_parse_pak_mock(mock_pak_excel: Path):
    """Проверка: парсинг ПАК извлекает серу и плотность в long-format."""
    df = parse_pak_xlsx(str(mock_pak_excel))

    expected_cols = ["timestamp", "tag", "value", "source", "sample_point", "unit"]
    assert all(c in df.columns for c in expected_cols)
    assert len(df) == 4 # 2 тега * 2 замера
    assert (df["source"] == "PAK").all()
    assert set(df["tag"]) == {"Mg.Sulfur", "D15"}
    assert "UTC" in str(df["timestamp"].dtype)


def test_no_duplicate_records(mock_lims_excel: Path, mock_pak_excel: Path):
    """Проверка: объединенная витрина не содержит дубликатов измерений."""
    combined = load_all_quality_data(str(mock_lims_excel), str(mock_pak_excel))
    
    # Проверка уникальности ключа (timestamp + tag + sample_point + source)
    duplicates = combined.duplicated(subset=["timestamp", "tag", "sample_point", "source"])
    assert not duplicates.any()


def test_real_quality_parquet_dod():
    """Интеграционный тест: проверка критериев DoD на сгенерированном quality_long.parquet."""
    parquet_path = Path("data/processed/quality_long.parquet")
    if not parquet_path.exists():
        pytest.skip("Файл data/processed/quality_long.parquet ещё не создан. Запустите scripts/load_quality.py.")

    df = pd.read_parquet(parquet_path)

    # 1. Проверка наличия обязательных колонок
    expected_cols = ["timestamp", "tag", "value", "source", "sample_point", "unit"]
    assert list(df.columns) == expected_cols

    # 2. Проверка источников
    assert set(df["source"].unique()).issubset({"LIMS", "PAK"})
    assert (df["source"] == "LIMS").sum() > 0
    assert (df["source"] == "PAK").sum() > 0

    # 3. Проверка типов
    assert "UTC" in str(df["timestamp"].dtype)
    assert pd.api.types.is_numeric_dtype(df["value"])

    # 4. Проверка отсутствия дубликатов
    duplicates = df.duplicated(subset=["timestamp", "tag", "sample_point", "source"])
    assert not duplicates.any(), f"Найдено {duplicates.sum()} дубликатов в quality_long.parquet"
