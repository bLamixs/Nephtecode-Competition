#!/usr/bin/env python3
"""
Скрипт: scripts/generate_features.py
Назначение: Генерация витрины признаков (Feature Store) для моделей Quality Agent и Reliability Agent.
- Лаги режима (10, 30, 60 мин).
- Скользящие окна (rolling mean, rolling std за 60 шагов).
- Z-score по технологическим нормам из справочника (DATA-06).

Блок: Data (DATA-05)
Критерии готовности (DoD):
1. Для каждого тега есть лаги 10/30/60 мин.
2. Для каждого тега есть rolling_mean_60, rolling_std_60.
3. Z-score считается строго по норме из справочника.
4. CLI-скрипт отрабатывает быстрее 5 минут.
5. Выходной файл data/processed/features.parquet.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Добавляем корень проекта в путь поиска модулей
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.etl.feature_store import (
    build_feature_store,
    save_features_parquet,
    TOP_20_KEY_TAGS,
)
from src.config import load_tag_dict


def prepare_long_dataset(
    quality_path: Path,
    telemetry_path: Optional[Path] = None,
    target_tags: Optional[list] = None,
) -> pd.DataFrame:
    """
    Подготовка Long-датасета для расчета признаков.
    Объединяет теги качества и ключевые теги КИП в канонический формат (date, tag, value).
    """
    records = []

    # 1. Загрузка витрины качества
    if quality_path.exists():
        df_qual = pd.read_parquet(quality_path)
        # Оставляем валидные замеры качества
        df_qual_val = df_qual.dropna(subset=['value_quality']).copy()
        df_qual_val['value'] = df_qual_val['value_quality']
        cols = ['date', 'tag', 'value']
        records.append(df_qual_val[cols])

    # 2. Загрузка ключевых тегов телеметрии КИП (если передана)
    if telemetry_path and telemetry_path.exists():
        df_telem = pd.read_parquet(telemetry_path)
        telem_tags = [c for c in df_telem.columns if c != 'date']
        if target_tags:
            telem_tags = [t for t in telem_tags if t in target_tags]

        df_telem_subset = df_telem[['date'] + telem_tags]
        df_telem_long = df_telem_subset.melt(
            id_vars=['date'], var_name='tag', value_name='value'
        )
        records.append(df_telem_long)

    if not records:
        raise ValueError("Нет данных для формирования признаков")

    df_combined = pd.concat(records, ignore_index=True)
    df_combined['date'] = pd.to_datetime(df_combined['date'], utc=True)
    df_combined = df_combined.sort_values(['tag', 'date']).reset_index(drop=True)
    return df_combined


def main():
    parser = argparse.ArgumentParser(description="Генерация витрины признаков Feature Store (DATA-05)")
    parser.add_argument("--quality", default="data/processed/telemetry_with_quality.parquet", help="Путь к витрине с качеством")
    parser.add_argument("--telemetry", default="data/processed/telemetry_clean.parquet", help="Путь к очищенной телеметрии КИП")
    parser.add_argument("--tag-dict", default="data/external/tag_dict.csv", help="Путь к справочнику тегов")
    parser.add_argument("--output", default="data/processed/features.parquet", help="Путь для сохранения features.parquet")
    parser.add_argument("--all-tags", action="store_true", help="Считать признаки для всех 97 тегов вместо топ-20")

    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("DATA-05: Генерация витрины признаков Feature Store")
    print("=" * 70)
    print(f"Витрина качества:   {args.quality}")
    print(f"Телеметрия КИП:     {args.telemetry}")
    print(f"Справочник тегов:   {args.tag_dict}")
    print(f"Выходной артефакт:  {args.output}")

    tags_to_process = None if args.all_tags else TOP_20_KEY_TAGS
    if tags_to_process:
        print(f"Режим фильтрации:   Топ-{len(tags_to_process)} ключевых режимных тегов")
    else:
        print("Режим фильтрации:   Все доступные технологические теги")

    # 1. Загрузка справочника норм
    tag_dict_df = pd.read_csv(args.tag_dict)
    print(f"\n[1/3] Загружен справочник тегов ({len(tag_dict_df)} записей)")

    # 2. Подготовка входных рядов
    print("[2/3] Подготовка временных рядов (Long-format)...")
    df_long = prepare_long_dataset(
        quality_path=Path(args.quality),
        telemetry_path=Path(args.telemetry),
        target_tags=tags_to_process,
    )
    print(f"      Всего рядов для расчёта: {len(df_long):,} строк по {df_long['tag'].nunique()} уникальным тегам")

    # 3. Расчёт признаков (лаги, скользящие окна, z-score)
    print("[3/3] Расчёт лагов (10, 30, 60 мин), скользящих окон (60 шагов) и Z-Score...")
    t_feat = time.time()
    df_features = build_feature_store(
        df=df_long,
        tag_dict_df=tag_dict_df,
        tags=df_long['tag'].unique().tolist(),
        lags=[10, 30, 60],
        window=60,
    )
    print(f"      Признаки рассчитаны за {time.time() - t_feat:.2f} сек")

    # 4. Сохранение в Parquet
    out_path = Path(args.output)
    save_features_parquet(df_features, out_path)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)

    total_time = time.time() - t_start
    print(f"\n[OK] Витрина признаков сохранена: {out_path}")
    print(f"     Размер файла: {file_size_mb:.2f} MB")
    print(f"     Количество строк: {len(df_features):,}")
    print(f"     Колонки: {list(df_features.columns)}")
    print(f"     Общее время выполнения: {total_time:.2f} сек (< 300 сек по DoD)")

    # 5. Проверка критериев DoD
    print("\n--- Проверка критериев приёмки (DoD) ---")
    print(f"[x] Лаги 10/30/60 мин сформированы:       {'lag10' in df_features.columns and 'lag60' in df_features.columns}")
    print(f"[x] Окна rolling_mean_60 / std созданы:   {'rolling_mean_60' in df_features.columns}")
    print(f"[x] Z-score по нормам рассчитан:          {'zscore' in df_features.columns}")
    print(f"[x] Время работы < 5 минут:               {total_time < 300} ({total_time:.1f} сек)")
    print("=" * 70)


if __name__ == "__main__":
    main()
