"""
Скрипт: scripts/load_raw_data.py
Назначение: Консольная утилита (CLI) для загрузки сырой телеметрии АВТ и 24-2000,
первичной очистки структуры и сохранения в бинарный формат Parquet.

Блок: Data (DATA-01)

Пример запуска:
    python scripts/load_raw_data.py --avt data/raw/avt_tags.csv --hydro data/raw/242000_tags.csv --output data/processed/
"""

import argparse
import sys
import time
from pathlib import Path

# Добавляем корень проекта в путь поиска модулей для корректного импорта из src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.load_telemetry import load_telemetry_csv, save_telemetry_parquet


def main():
    parser = argparse.ArgumentParser(
        description="CLI утилита загрузки и конвертации телеметрии в Parquet."
    )
    parser.add_argument(
        "--avt",
        type=str,
        default="data/raw/avt_tags.csv",
        help="Путь к исходному CSV файлу телеметрии АВТ (по умолчанию: data/raw/avt_tags.csv)"
    )
    parser.add_argument(
        "--hydro",
        type=str,
        default="data/raw/242000_tags.csv",
        help="Путь к исходному CSV файлу телеметрии гидроочистки 24-2000 (по умолчанию: data/raw/242000_tags.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/",
        help="Директория для сохранения готовых Parquet-файлов (по умолчанию: data/processed/)"
    )

    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("НАЧАЛО ВЫПОЛНЕНИЯ ЗАДАЧИ DATA-01: ЗАГРУЗКА И КОНВЕРТАЦИЯ ТЕЛЕМЕТРИИ")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. ОБРАБОТКА ТЕЛЕМЕТРИИ АВТ
    # -------------------------------------------------------------------------
    print(f"\n[1/2] Загрузка телеметрии АВТ из: {args.avt}")
    t0 = time.time()
    df_avt = load_telemetry_csv(args.avt, source="AVT")
    load_time_avt = time.time() - t0

    print(f"  -> Загружено строк: {len(df_avt):,}")
    print(f"  -> Колонок после очистки: {df_avt.shape[1]}")
    print(f"  -> Временной диапазон: {df_avt['date'].min()} — {df_avt['date'].max()}")
    print(f"  -> Тип колонки date: {df_avt['date'].dtype}")
    print(f"  -> Использование памяти в RAM: {df_avt.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"  -> Время загрузки: {load_time_avt:.2f} сек")

    avt_parquet_path = out_dir / "telemetry_avt.parquet"
    print(f"  -> Сохранение в Parquet: {avt_parquet_path}")
    save_telemetry_parquet(df_avt, str(avt_parquet_path))
    avt_size_mb = avt_parquet_path.stat().st_size / (1024**2)
    print(f"  -> Размер сохранённого Parquet: {avt_size_mb:.2f} MB")

    # -------------------------------------------------------------------------
    # 2. ОБРАБОТКА ТЕЛЕМЕТРИИ ГИДРООЧИСТКИ 24-2000
    # -------------------------------------------------------------------------
    print(f"\n[2/2] Загрузка телеметрии 24-2000 из: {args.hydro}")
    t1 = time.time()
    df_hydro = load_telemetry_csv(args.hydro, source="24-2000")
    load_time_hydro = time.time() - t1

    print(f"  -> Загружено строк: {len(df_hydro):,}")
    print(f"  -> Колонок после очистки: {df_hydro.shape[1]}")
    print(f"  -> Временной диапазон: {df_hydro['date'].min()} — {df_hydro['date'].max()}")
    print(f"  -> Тип колонки date: {df_hydro['date'].dtype}")
    print(f"  -> Использование памяти в RAM: {df_hydro.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"  -> Время загрузки: {load_time_hydro:.2f} сек")

    hydro_parquet_path = out_dir / "telemetry_242000.parquet"
    print(f"  -> Сохранение в Parquet: {hydro_parquet_path}")
    save_telemetry_parquet(df_hydro, str(hydro_parquet_path))
    hydro_size_mb = hydro_parquet_path.stat().st_size / (1024**2)
    print(f"  -> Размер сохранённого Parquet: {hydro_size_mb:.2f} MB")

    print("\n" + "=" * 70)
    print("ВЫПОЛНЕНИЕ DATA-01 УСПЕШНО ЗАВЕРШЕНО!")
    print(f"ИТОГОВЫЕ АРТЕФАКТЫ В {out_dir}:")
    print(f"  - {avt_parquet_path.name} ({avt_size_mb:.2f} MB, {len(df_avt):,} строк)")
    print(f"  - {hydro_parquet_path.name} ({hydro_size_mb:.2f} MB, {len(df_hydro):,} строк)")
    print("=" * 70)


if __name__ == "__main__":
    main()
