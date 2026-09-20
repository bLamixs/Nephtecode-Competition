"""
Тесты для блока DATA-01: Загрузка телеметрии и конвертация в Parquet.
Файл: tests/test_load_telemetry.py
"""

import os
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.etl.load_telemetry import load_telemetry_csv, save_telemetry_parquet


@pytest.fixture
def sample_csv_with_unnamed(tmp_path: Path) -> Path:
    """Создаёт временный CSV с техническими колонками Unnamed:*, разделителями и датами."""
    file_path = tmp_path / "mock_telemetry.csv"
    content = (
        "Unnamed: 0.1,Unnamed: 0,,date,T1,P2,F3\n"
        "0,0,dummy,2026-01-01 00:00:00,120.5,3.4,50.1\n"
        "1,1,dummy,2026-01-01 00:10:00,121.0,3.5,50.8\n"
        "2,2,dummy,2026-01-01 00:20:00,120.8,3.4,51.0\n"
    )
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_load_telemetry_removes_unnamed_and_empty_columns(sample_csv_with_unnamed: Path):
    """Проверка: удаление колонок Unnamed:* и пустых заголовков."""
    df = load_telemetry_csv(str(sample_csv_with_unnamed), source="AVT")
    
    # 1. Проверяем, что служебных колонок нет
    unnamed_cols = [c for c in df.columns if "Unnamed" in str(c) or str(c).strip() == ""]
    assert len(unnamed_cols) == 0, f"Обнаружены нежелательные колонки: {unnamed_cols}"
    
    # 2. Проверяем наличие целевых технологических колонок
    expected_cols = {"date", "T1", "P2", "F3", "source"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 3


def test_load_telemetry_date_utc_timezone(sample_csv_with_unnamed: Path):
    """Проверка: приведение даты к типу datetime с временной зоной UTC."""
    df = load_telemetry_csv(str(sample_csv_with_unnamed), source="AVT")
    
    # Тип должен быть datetime64 с UTC
    assert str(df["date"].dtype).startswith("datetime64")
    assert "UTC" in str(df["date"].dtype)
    assert df["date"].iloc[0] == pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def test_load_telemetry_source_column(sample_csv_with_unnamed: Path):
    """Проверка: добавление правильного значения источника данных."""
    df_avt = load_telemetry_csv(str(sample_csv_with_unnamed), source="AVT")
    assert (df_avt["source"] == "AVT").all()

    df_hydro = load_telemetry_csv(str(sample_csv_with_unnamed), source="24-2000")
    assert (df_hydro["source"] == "24-2000").all()


def test_save_and_read_parquet(tmp_path: Path, sample_csv_with_unnamed: Path):
    """Проверка: сохранение в Parquet и корректность обратного чтения."""
    df = load_telemetry_csv(str(sample_csv_with_unnamed), source="AVT")
    out_parquet = tmp_path / "test_telemetry.parquet"

    save_telemetry_parquet(df, str(out_parquet))
    assert out_parquet.exists()
    assert out_parquet.stat().st_size > 0

    # Читаем обратно и сравниваем
    df_loaded = pd.read_parquet(out_parquet)
    assert len(df_loaded) == len(df)
    assert list(df_loaded.columns) == list(df.columns)
    assert (df_loaded["T1"] == df["T1"]).all()


def test_real_processed_artifacts_dod():
    """Проверка критериев приемки (DoD) на сгенерированных файлах проекта, если они уже созданы."""
    avt_path = Path("data/processed/telemetry_avt.parquet")
    hydro_path = Path("data/processed/telemetry_242000.parquet")

    if not avt_path.exists() or not hydro_path.exists():
        pytest.skip("Файлы data/processed/telemetry_*.parquet ещё не сгенерированы. Пропуск интеграционного теста.")

    # Проверка АВТ
    df_avt = pd.read_parquet(avt_path)
    assert len(df_avt) == 189_217, f"Ожидалось 189,217 строк, получено: {len(df_avt)}"
    assert not any("Unnamed" in c for c in df_avt.columns)
    assert "UTC" in str(df_avt["date"].dtype)
    assert (df_avt["source"] == "AVT").all()

    # Проверка 24-2000
    df_hydro = pd.read_parquet(hydro_path)
    assert len(df_hydro) == 189_217, f"Ожидалось 189,217 строк, получено: {len(df_hydro)}"
    assert not any("Unnamed" in c for c in df_hydro.columns)
    assert "UTC" in str(df_hydro["date"].dtype)
    assert (df_hydro["source"] == "24-2000").all()
