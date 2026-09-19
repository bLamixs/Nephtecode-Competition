#!/usr/bin/env python3
"""
Скрипт: scripts/train_quality_agent.py
Назначение: Обучение модели коррекции качества Quality Agent (AGENT-02).
ML-модель обучается на остатках (Residuals = LIMS - ВАК) по содержанию серы (Mg.Sulfur).

Использование:
    python scripts/train_quality_agent.py [--model-path output/models/quality_lgbm.pkl]

Критерий приёмки (DoD):
    MAE на тесте (2026 год) < 1.0 мг/кг.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Добавляем корень проекта в путь
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logging_config import setup_logger
from src.agents.quality_agent import QualityAgent

logger = setup_logger('train_quality_agent')


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение модели коррекции качества QualityAgent (AGENT-02)")
    parser.add_argument(
        "--telemetry",
        type=str,
        default="data/processed/telemetry_clean.parquet",
        help="Путь к чистой телеметрии КИП"
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="data/processed/telemetry_with_quality.parquet",
        help="Путь к телеметрии с замерами качества"
    )
    parser.add_argument(
        "--vac",
        type=str,
        default="data/processed/vac_predictions.parquet",
        help="Путь к прогнозам ВАК"
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="output/models/quality_lgbm.pkl",
        help="Путь для сохранения обученной модели"
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2026,
        help="Год для тестовой валидации без заглядывания в будущее"
    )
    return parser.parse_args()


def load_and_align_datasets(telemetry_path: Path, quality_path: Path, vac_path: Path):
    """
    Загрузка и выравнивание таймзон и дат между телеметрией, качеством и ВАК.
    """
    logger.info("Загрузка данных...")
    df_telemetry = pd.read_parquet(telemetry_path)
    df_quality = pd.read_parquet(quality_path)
    df_vac = pd.read_parquet(vac_path)

    # Приведение временных меток к tz-naive
    df_telemetry['date'] = pd.to_datetime(df_telemetry['date']).dt.tz_localize(None)
    df_quality['date'] = pd.to_datetime(df_quality['date']).dt.tz_localize(None)
    df_vac['date'] = pd.to_datetime(df_vac['date']).dt.tz_localize(None)

    # Фильтруем целевые ряды по сере
    sulfur_q = df_quality[df_quality['tag'] == 'Mg.Sulfur'].dropna(subset=['value_quality'])
    # Убираем аномальные выбросы датчиков (сера > 200 ppm при норме 10 ppm)
    sulfur_q = sulfur_q[sulfur_q['value_quality'] <= 150.0]

    sulfur_vac = df_vac[df_vac['tag'] == '24-2000:GODT:Sulfur']

    logger.info(f"Замеров серы в quality: {len(sulfur_q):,}, прогнозов ВАК: {len(sulfur_vac):,}")

    # Объединяем телеметрию, качество и ВАК по date
    merged = pd.merge(df_telemetry, sulfur_q[['date', 'value_quality', 'source', 'age_min']], on='date', how='inner')
    merged = pd.merge(merged, sulfur_vac[['date', 'vac_value']], on='date', how='inner')
    merged.sort_values('date', inplace=True)
    merged.reset_index(drop=True, inplace=True)

    logger.info(f"Объединённый датасет: {len(merged):,} точек с валидными замерами серы и ВАК.")
    return merged


def main():
    args = parse_args()
    model_path = Path(args.model_output)

    logger.info("=" * 80)
    logger.info("СТАРТ: Обучение модели QualityAgent (AGENT-02)")
    logger.info("=" * 80)

    dataset = load_and_align_datasets(
        Path(args.telemetry),
        Path(args.quality),
        Path(args.vac)
    )

    # Разделение по годам: Train (до test_year) и Test (test_year)
    train_mask = dataset['date'].dt.year < args.test_year
    test_mask = dataset['date'].dt.year >= args.test_year

    df_train = dataset[train_mask].copy()
    df_test = dataset[test_mask].copy()

    logger.info(f"Обучающая выборка (до {args.test_year}): {len(df_train):,} строк ({df_train['date'].min()} .. {df_train['date'].max()})")
    logger.info(f"Тестовая выборка ({args.test_year}+): {len(df_test):,} строк ({df_test['date'].min()} .. {df_test['date'].max()})")

    # Инициализация агента
    agent = QualityAgent(model_path=str(model_path))

    # Формирование обучающих матриц
    logger.info("Формирование признаков и остатков...")
    X_train = agent._prepare_features(df_train, df_train)
    y_train = agent._calculate_residuals(df_train['value_quality'], df_train['vac_value'])

    X_test = agent._prepare_features(df_test, df_test)
    y_test = agent._calculate_residuals(df_test['value_quality'], df_test['vac_value'])

    # Кросс-валидация TimeSeriesSplit
    logger.info("Запуск временной кросс-валидации (TimeSeriesSplit, 5 фолдов)...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []

    for fold, (trn_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_f_trn, y_f_trn = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_f_val, y_f_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        import lightgbm as lgb
        fold_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        fold_model.fit(X_f_trn, y_f_trn)
        preds_val = fold_model.predict(X_f_val)
        mae = mean_absolute_error(y_f_val, preds_val)
        cv_maes.append(mae)
        logger.info(f"  Фолд {fold}: MAE остатков = {mae:.4f}")

    logger.info(f"Средний CV MAE остатков: {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}")

    # Финальное обучение на всей обучающей выборке
    logger.info(f"Финальное обучение на {len(X_train):,} строках...")
    agent.train(
        telemetry=df_train,
        quality=df_train,
        vac=df_train,
        model_type='auto'
    )

    # Оценка на отложенном тесте (2026 год)
    pred_test = agent.predict(df_test, df_test, vac=df_test['vac_value'])
    actual_test = df_test['value_quality']
    vac_test = df_test['vac_value']

    mae_vac = mean_absolute_error(actual_test, vac_test)
    mae_agent = mean_absolute_error(actual_test, pred_test)
    rmse_agent = np.sqrt(mean_squared_error(actual_test, pred_test))
    r2_agent = r2_score(actual_test, pred_test)

    logger.info("=" * 80)
    logger.info("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ НА ДАННЫХ 2026 ГОДА:")
    logger.info(f"  Baseline ВАК MAE: {mae_vac:.4f} мг/кг")
    logger.info(f"  QualityAgent MAE: {mae_agent:.4f} мг/кг (улучшение на {max(0.0, (mae_vac - mae_agent)/mae_vac):.1%})")
    logger.info(f"  QualityAgent RMSE: {rmse_agent:.4f} мг/кг")
    logger.info(f"  QualityAgent R2: {r2_agent:.4f}")
    logger.info("=" * 80)

    # Проверка DoD
    if mae_agent < 1.0:
        logger.info(f"✓ DoD ВЫПОЛНЕН: MAE ({mae_agent:.4f}) < 1.0 мг/кг")
    else:
        logger.warning(f"⚠ Внимание: MAE ({mae_agent:.4f}) >= 1.0 мг/кг.")

    logger.info(f"Файл модели сохранён: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    logger.info("AGENT-02 успешно завершён.")


if __name__ == '__main__':
    main()
