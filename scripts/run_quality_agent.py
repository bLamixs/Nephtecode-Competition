"""
Скрипт: scripts/run_quality_agent.py
Назначение: Запуск Quality Agent для мультигоризонтного прогноза качества (AGENT-03).
Горизонты прогнозирования: 30, 60, 120 минут.
Показатели: Sulfur, D15, T50, T90, T95, CFPP, flash с учётом запаздывания процессов.
Сохранение результата: data/processed/quality_forecast.parquet.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Добавляем корень репозитория в sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.logging_config import setup_logger
from src.agents.quality_agent import QualityAgent

logger = setup_logger('run_quality_agent')


def parse_args():
    parser = argparse.ArgumentParser(description="CLI для мультигоризонтного прогноза Quality Agent.")
    parser.add_argument(
        '--telemetry',
        type=str,
        default='data/processed/telemetry_clean.parquet',
        help='Путь к очищенной телеметрии'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='output/models/quality_lgbm.pkl',
        help='Путь к обученной модели остатков'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/quality_forecast.parquet',
        help='Путь для сохранения прогнозов'
    )
    parser.add_argument(
        '--horizons',
        nargs='+',
        type=int,
        default=[30, 60, 120],
        help='Горизонты прогноза в минутах (по умолчанию: 30 60 120)'
    )
    parser.add_argument(
        '--assess-risk',
        action='store_true',
        help='Также рассчитать риски нарушения спецификаций (AGENT-04)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    telemetry_path = Path(args.telemetry)
    if not telemetry_path.exists():
        # Fallback к telemetry_with_quality.parquet если clean не найден
        alt_path = Path('data/processed/telemetry_with_quality.parquet')
        if alt_path.exists():
            telemetry_path = alt_path
        else:
            logger.error(f"Файл телеметрии не найден: {args.telemetry}")
            sys.exit(1)

    logger.info(f"Загрузка телеметрии из: {telemetry_path}")
    telemetry = pd.read_parquet(telemetry_path)
    logger.info(f"Загружено записей: {len(telemetry):,}")

    logger.info(f"Инициализация QualityAgent с моделью: {args.model}")
    agent = QualityAgent(model_path=args.model)

    logger.info(f"Построение прогнозов на горизонты: {args.horizons} минут...")
    forecast_df = agent.predict_forecast(telemetry, horizons_min=args.horizons)

    if args.assess_risk:
        logger.info("Расчёт рисков выхода за спецификации (AGENT-04)...")
        forecast_df = agent.assess_risk(forecast_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_parquet(output_path, index=False)

    logger.info(f"Прогноз успешно сохранён в: {output_path}")
    logger.info(f"Размер прогноза: {forecast_df.shape} (строк: {len(forecast_df):,})")
    logger.info(f"Колонки: {list(forecast_df.columns)}")


if __name__ == '__main__':
    main()
