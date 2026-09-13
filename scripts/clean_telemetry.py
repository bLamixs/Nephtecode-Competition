"""
Скрипт: scripts/clean_telemetry.py
Назначение: Консольная утилита (CLI) для детекции аномальных выбросов (Z-Score по окну 60)
и формирования очищенной телеметрии и масок качества.

Блок: Data (DATA-02)

Пример запуска:
    python scripts/clean_telemetry.py --avt data/processed/telemetry_avt.parquet --hydro data/processed/telemetry_242000.parquet --output data/processed/
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.clean_telemetry import (
    detect_outliers_zscore,
    create_quality_masks,
    clean_telemetry_outliers
)


def main():
    parser = argparse.ArgumentParser(
        description="CLI утилита для детекции выбросов и создания масок качества телеметрии."
    )
    parser.add_argument(
        "--avt",
        type=str,
        default="data/processed/telemetry_avt.parquet",
        help="Путь к Parquet телеметрии АВТ (по умолчанию: data/processed/telemetry_avt.parquet)"
    )
    parser.add_argument(
        "--hydro",
        type=str,
        default="data/processed/telemetry_242000.parquet",
        help="Путь к Parquet телеметрии 24-2000 (по умолчанию: data/processed/telemetry_242000.parquet)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/",
        help="Директория для сохранения результатов (по умолчанию: data/processed/)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Размер скользящего окна в точках (по умолчанию: 60 шагов = 10 часов)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Порог Z-score для детекции выброса (по умолчанию: 4.0 сигма)"
    )

    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("НАЧАЛО ВЫПОЛНЕНИЯ ЗАДАЧИ DATA-02: ДЕТЕКЦИЯ ВЫБРОСОВ И СОЗДАНИЕ МАСОК")
    print(f"Параметры: Окно = {args.window} точек (10 ч), Порог = {args.threshold} сигма")
    print("=" * 75)

    # 1. Загрузка исходных данных от DATA-01
    print(f"\n[1/3] Загрузка телеметрии...")
    df_avt = pd.read_parquet(args.avt)
    df_hydro = pd.read_parquet(args.hydro)
    print(f"  -> АВТ: {len(df_avt):,} строк, {df_avt.shape[1]} колонок")
    print(f"  -> 24-2000: {len(df_hydro):,} строк, {df_hydro.shape[1]} колонок")

    # Объединяем телеметрию по ключу 'date', разрешая коллизии имен суффиксами
    # Например, T6 -> T6_avt и T6_hydro
    c_avt = df_avt.drop(columns=['source'], errors='ignore')
    c_hydro = df_hydro.drop(columns=['source'], errors='ignore')
    telemetry_merged = pd.merge(
        c_avt,
        c_hydro,
        on='date',
        how='inner',
        suffixes=('_avt', '_hydro')
    )
    print(f"  -> Объединенная телеметрия: {len(telemetry_merged):,} строк, {telemetry_merged.shape[1]} колонок")

    # 2. Детекция выбросов Z-Score по скользящему окну
    print(f"\n[2/3] Детекция выбросов Z-Score для всех тегов...")
    t0 = time.time()
    outliers_df = detect_outliers_zscore(
        telemetry_merged,
        window=args.window,
        threshold=args.threshold,
        exclude_cols=['date']
    )
    t_detect = time.time() - t0

    # 3. Формирование масок качества и очищенного датасета
    print(f"\n[3/3] Формирование масок качества и очистка от выбросов...")
    telemetry_masks = create_quality_masks(telemetry_merged, outliers_df, date_col='date')
    telemetry_clean = clean_telemetry_outliers(telemetry_merged, outliers_df)

    # Сохранение в Parquet
    clean_path = out_dir / "telemetry_clean.parquet"
    masks_path = out_dir / "telemetry_masks.parquet"

    telemetry_clean.to_parquet(clean_path, engine='pyarrow', compression='snappy', index=False)
    telemetry_masks.to_parquet(masks_path, engine='pyarrow', compression='snappy', index=False)

    # Статистика и проверка DoD
    num_vals = outliers_df.shape[0] * outliers_df.shape[1]
    outlier_count = outliers_df.sum().sum()
    outlier_ratio = (outlier_count / num_vals) * 100

    mask_cols = [c for c in telemetry_masks.columns if c != 'date']
    total_mask_points = telemetry_masks.shape[0] * len(mask_cols)
    total_zero_masks = (telemetry_masks[mask_cols] == 0).sum().sum()
    zero_mask_ratio = (total_zero_masks / total_mask_points) * 100

    print("=" * 75)
    print("ВЫПОЛНЕНИЕ DATA-02 УСПЕШНО ЗАВЕРШЕНО!")
    print(f"Время расчёта z-score: {t_detect:.2f} сек")
    print(f"Выбросов обнаружено: {outlier_count:,} из {num_vals:,} ({outlier_ratio:.2f}%)")
    print(f"\nИТОГОВЫЕ АРТЕФАКТЫ В {out_dir}:")
    print(f"  - {clean_path.name}: {clean_path.stat().st_size / (1024**2):.2f} MB, {len(telemetry_clean):,} строк, {telemetry_clean.shape[1]} колонок")
    print(f"  - {masks_path.name}: {masks_path.stat().st_size / (1024**2):.2f} MB, {len(telemetry_masks):,} строк, {telemetry_masks.shape[1]} колонок")
    print(f"\nПРОВЕРКА DOD (Доля mask=0 <= 5%):")
    print(f"  - Доля mask=0: {zero_mask_ratio:.2f}% (Критерий DoD соблюден: {'OK' if zero_mask_ratio <= 5.0 else 'FAIL'})")
    print("=" * 75)


if __name__ == "__main__":
    main()
