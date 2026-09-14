"""
Скрипт: scripts/sync_data.py
Назначение: Консольная утилита (CLI) для временной синхронизации телеметрии
с показателями качества ЛИМС и ПАК через merge_asof с соблюдением приоритета ЛИМС > ПАК.

Блок: Data (DATA-04)

Пример запуска:
    python scripts/sync_data.py --telemetry data/processed/telemetry_clean.parquet --quality data/processed/quality_long.parquet --output data/processed/telemetry_with_quality.parquet
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.sync_data import sync_telemetry_quality


def main():
    parser = argparse.ArgumentParser(
        description="CLI утилита для синхронизации телеметрии и анализов качества через merge_asof."
    )
    parser.add_argument(
        "--telemetry",
        type=str,
        default="data/processed/telemetry_clean.parquet",
        help="Путь к очищенной телеметрии (по умолчанию: data/processed/telemetry_clean.parquet)"
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="data/processed/quality_long.parquet",
        help="Путь к витрине качества (по умолчанию: data/processed/quality_long.parquet)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/telemetry_with_quality.parquet",
        help="Путь для сохранения результата (по умолчанию: data/processed/telemetry_with_quality.parquet)"
    )
    parser.add_argument(
        "--tolerance-lims",
        type=int,
        default=60,
        help="Временной допуск (минуты) для ЛИМС (по умолчанию: 60)"
    )
    parser.add_argument(
        "--tolerance-pak",
        type=int,
        default=15,
        help="Временной допуск (минуты) для ПАК (по умолчанию: 15)"
    )

    args = parser.parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("НАЧАЛО ВЫПОЛНЕНИЯ ЗАДАЧИ DATA-04: ВРЕМЕННАЯ СИНХРОНИЗАЦИЯ (MERGE_ASOF)")
    print(f"Параметры: tolerance_lims={args.tolerance_lims} мин, tolerance_pak={args.tolerance_pak} мин")
    print("=" * 75)

    # 1. Загрузка витрин телеметрии и качества
    print(f"\n[1/3] Загрузка входных витрин...")
    t0 = time.time()
    df_telemetry = pd.read_parquet(args.telemetry)
    df_quality = pd.read_parquet(args.quality)
    print(f"  -> Телеметрия: {len(df_telemetry):,} строк, {df_telemetry.shape[1]} колонок")
    print(f"  -> Качество: {len(df_quality):,} записей, {df_quality['tag'].nunique()} уникальных тегов")

    # 2. Выполнение синхронизации
    print(f"\n[2/3] Синхронизация по времени (merge_asof с приоритетом ЛИМС > ПАК)...")
    t1 = time.time()
    synced_df = sync_telemetry_quality(
        telemetry=df_telemetry,
        quality=df_quality,
        tolerance_lims=args.tolerance_lims,
        tolerance_pak=args.tolerance_pak
    )
    t_sync = time.time() - t1
    print(f"  -> Синхронизация завершена за {t_sync:.2f} сек")
    print(f"  -> Сформировано строк витрины: {len(synced_df):,}")

    # 3. Анализ качества покрытия и источников
    print(f"\n[3/3] Сохранение в Parquet и аудит покрытия...")
    synced_df.to_parquet(out_path, engine='pyarrow', compression='snappy', index=False)
    file_size_mb = out_path.stat().st_size / (1024**2)

    # Статистика по источникам и свежести
    src_counts = synced_df['source'].value_counts(dropna=False).to_dict()
    stale_count = synced_df['stale'].sum()
    stale_ratio = (stale_count / len(synced_df)) * 100

    # Проверка приоритета ЛИМС > ПАК по сере
    sulfur_slice = synced_df[synced_df['tag'] == 'Mg.Sulfur']
    sulfur_src = sulfur_slice['source'].value_counts().to_dict()

    print("=" * 75)
    print("ВЫПОЛНЕНИЕ DATA-04 УСПЕШНО ЗАВЕРШЕНО!")
    print(f"ИТОГОВЫЙ АРТЕФАКТ: {out_path} ({file_size_mb:.2f} MB)")
    print(f"  - Всего строк: {len(synced_df):,}")
    print(f"  - Колонки: {synced_df.columns.tolist()}")
    print(f"  - Распределение источников качества:")
    for src, cnt in src_counts.items():
        print(f"    * {src if pd.notna(src) else 'Нет замера (NaN)'}: {cnt:,}")
    print(f"  - Покрытие по сере (Mg.Sulfur): {sulfur_src}")
    print(f"  - Замеров со статусом stale=True (возраст > 240 мин): {stale_count:,} ({stale_ratio:.2f}%)")
    print("=" * 75)


if __name__ == "__main__":
    main()
