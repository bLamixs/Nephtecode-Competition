#!/usr/bin/env python3
"""
Скрипт: scripts/calculate_vac.py
Назначение: Расчёт Виртуальных Анализаторов Качества (ВАК) по очищенной телеметрии КИП (AGENT-01).

Использование:
    python scripts/calculate_vac.py [--input PATH] [--output PATH]

Выходной артефакт:
    data/processed/vac_predictions.parquet
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Добавляем корень проекта в путь поиска модулей
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logging_config import setup_logger
from src.utils.vac_formulas import calculate_all_vac

logger = setup_logger('calculate_vac')


def parse_args():
    parser = argparse.ArgumentParser(description="Расчёт Виртуальных Анализаторов Качества (ВАК)")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/telemetry_clean.parquet",
        help="Путь к очищенной телеметрии (default: data/processed/telemetry_clean.parquet)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/vac_predictions.parquet",
        help="Путь для сохранения прогнозов ВАК (default: data/processed/vac_predictions.parquet)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info(f"Запуск расчёта ВАК. Чтение телеметрии из: {input_path}")
    if not input_path.exists():
        logger.error(f"Файл телеметрии не найден: {input_path}")
        sys.exit(1)

    df_telemetry = pd.read_parquet(input_path)
    logger.info(f"Загружено {len(df_telemetry):,} строк телеметрии, колонок: {len(df_telemetry.columns)}")

    # Расчёт всех формул ВАК
    logger.info("Расчёт формул ВАК для АВТ-6 и 24-2000...")
    vac_df = calculate_all_vac(df_telemetry)

    logger.info(f"Сформирован датасет ВАК: {len(vac_df):,} строк, колонки: {list(vac_df.columns)}")
    unique_tags = vac_df['tag'].unique().tolist()
    logger.info(f"Рассчитано {len(unique_tags)} виртуальных анализаторов: {unique_tags}")

    # Создание директории при необходимости и сохранение
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vac_df.to_parquet(output_path, index=False)
    logger.info(f"Файл успешно сохранён: {output_path} ({output_path.stat().st_size / (1024 * 1024):.2f} MB)")

    # Краткая статистика
    stats = vac_df.groupby('tag')['vac_value'].agg(['count', 'min', 'mean', 'max'])
    logger.info("Сводная статистика по ВАК:\n" + stats.to_string())
    logger.info("AGENT-01: Расчёт ВАК успешно завершён.")


if __name__ == '__main__':
    main()
