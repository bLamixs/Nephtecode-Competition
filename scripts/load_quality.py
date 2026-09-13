"""
Скрипт: scripts/load_quality.py
Назначение: Консольная утилита (CLI) для загрузки редких данных ЛИМС и поточных ПАК
из таблиц Excel и сохранения в единый long-format Parquet.

Блок: Data (DATA-03)

Пример запуска:
    python scripts/load_quality.py --lims data/raw/ЛИМСы.xlsx --pak data/raw/ПАК.xlsx --output data/processed/quality_long.parquet
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.load_quality import load_all_quality_data, parse_lims_xlsx, parse_pak_xlsx


def find_file(candidates: list) -> str:
    """Возвращает первый существующий путь из списка кандидатов."""
    for p in candidates:
        if Path(p).exists():
            return str(p)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="CLI утилита для загрузки и объединения анализов ЛИМС и ПАК в long-format."
    )
    parser.add_argument(
        "--lims",
        type=str,
        default="data/raw/ЛИМСы.xlsx",
        help="Путь к файлу лабораторных анализов ЛИМС (по умолчанию: data/raw/ЛИМСы.xlsx)"
    )
    parser.add_argument(
        "--pak",
        type=str,
        default="data/raw/ПАК.xlsx",
        help="Путь к файлу поточных анализаторов ПАК (по умолчанию: data/raw/ПАК.xlsx)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/quality_long.parquet",
        help="Путь для сохранения результирующего файла Parquet"
    )

    args = parser.parse_args()

    # Разрешаем возможные варианты названий файлов
    lims_path = find_file([args.lims, "data/raw/ЛИМСы.xlsx", "ЛИМСы.xlsx"])
    pak_path = find_file([args.pak, "data/raw/ПАК.xlsx", "data/raw/Выгрузка ПАК.xlsx", "ПАК.xlsx"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("НАЧАЛО ВЫПОЛНЕНИЯ ЗАДАЧИ DATA-03: ЗАГРУЗКА ДАННЫХ КАЧЕСТВА (ЛИМС + ПАК)")
    print("=" * 75)

    # 1. Загрузка ЛИМС
    print(f"\n[1/3] Парсинг лабораторных анализов ЛИМС из: {lims_path}")
    t0 = time.time()
    df_lims = parse_lims_xlsx(lims_path)
    t_lims = time.time() - t0
    print(f"  -> Извлечено измерений ЛИМС: {len(df_lims):,}")
    print(f"  -> Уникальных показателей: {df_lims['tag'].nunique()} ({df_lims['tag'].unique().tolist()[:6]}...)")
    print(f"  -> Время обработки: {t_lims:.2f} сек")

    # 2. Загрузка ПАК
    print(f"\n[2/3] Парсинг поточных анализаторов ПАК из: {pak_path}")
    t1 = time.time()
    df_pak = parse_pak_xlsx(pak_path)
    t_pak = time.time() - t1
    print(f"  -> Извлечено измерений ПАК: {len(df_pak):,}")
    print(f"  -> Показатели ПАК: {df_pak['tag'].unique().tolist()}")
    print(f"  -> Время обработки: {t_pak:.2f} сек")

    # 3. Объединение в Long-Format
    print(f"\n[3/3] Объединение в единую витрину и сохранение в Parquet...")
    combined = pd.concat([df_lims, df_pak], ignore_index=True)
    # Дедупликация по ключам измерения
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=['timestamp', 'tag', 'sample_point', 'source'])
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    expected_cols = ['timestamp', 'tag', 'value', 'source', 'sample_point', 'unit']
    combined = combined[expected_cols]
    dedup_dropped = before_dedup - len(combined)

    # Сохраняем в Parquet со сжатием Snappy
    combined.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    file_size_mb = output_path.stat().st_size / (1024**2)

    print("=" * 75)
    print("ВЫПОЛНЕНИЕ DATA-03 УСПЕШНО ЗАВЕРШЕНО!")
    print(f"ИТОГОВЫЙ АРТЕФАКТ: {output_path}")
    print(f"  - Размер файла: {file_size_mb:.2f} MB")
    print(f"  - Всего строк измерений: {len(combined):,}")
    print(f"    * ЛИМС: {(combined['source'] == 'LIMS').sum():,} строк")
    print(f"    * ПАК:  {(combined['source'] == 'PAK').sum():,} строк")
    print(f"  - Дубликатов удалено: {dedup_dropped}")
    print(f"  - Колонки: {combined.columns.tolist()}")
    print(f"  - Диапазон дат: {combined['timestamp'].min()} — {combined['timestamp'].max()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
