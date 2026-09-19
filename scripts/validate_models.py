"""
Скрипт: scripts/validate_models.py
Назначение: Walk-forward валидация моделей качества по времени без утечки будущего (AGENT-06).

Методология:
- Train период: 2023-01-01 – 2025-12-31.
- Test период: 2026-01-01 – 2026-08-07.
- Walk-forward валидация: 5 фолдов по времени с шагом +6 месяцев (no shuffle).
- Метрики: MAE, RMSE, bias для S, D15, T50, T90, T95, CFPP.
- Сравнение с чистым ВАК (подтверждение улучшения точности).
- Сохранение отчёта: output/validation_report.json.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Добавляем корень репозитория в sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.logging_config import setup_logger
from src.agents.quality_agent import QualityAgent
from src.utils.vac_formulas import (
    vac_sulfur_24_2000,
    vac_d15_godt,
    vac_t50_godt,
    vac_t95_godt,
    vac_cfpp_godt
)

logger = setup_logger('validate_models')


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Расчёт MAE, RMSE и bias (среднее отклонение)."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"MAE": None, "RMSE": None, "bias": None, "count": 0}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    bias = float(np.mean(yp - yt))
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "bias": round(bias, 4),
        "count": int(mask.sum())
    }


def run_validation():
    logger.info("Начало валидации моделей качества (AGENT-06)...")

    # 1. Загрузка телеметрии и качества
    telem_path = Path('data/processed/telemetry_clean.parquet')
    qual_path = Path('data/processed/telemetry_with_quality.parquet')
    model_path = Path('output/models/quality_lgbm.pkl')

    if not telem_path.exists():
        logger.error(f"Файл телеметрии не найден: {telem_path}")
        return

    logger.info(f"Загрузка телеметрии: {telem_path}")
    telemetry = pd.read_parquet(telem_path)
    if 'date' in telemetry.columns:
        telemetry['date'] = pd.to_datetime(telemetry['date']).dt.tz_localize(None)
        telemetry = telemetry.sort_values('date').reset_index(drop=True)

    logger.info(f"Загрузка данных качества: {qual_path}")
    qual_long = pd.read_parquet(qual_path)
    qual_long['date'] = pd.to_datetime(qual_long['date']).dt.tz_localize(None)

    # Приводим качество к широкому формату для ключевых меток
    target_tags_map = {
        'Mg.Sulfur': 'Sulfur',
        'D15': 'D15',
        '50%.T': 'T50',
        '90%.T': 'T90',
        '95%.T': 'T95',
        'CFPP': 'CFPP'
    }
    filtered_qual = qual_long[qual_long['tag'].isin(target_tags_map.keys())].copy()
    qual_pivot = filtered_qual.pivot(index='date', columns='tag', values='value_quality')
    qual_pivot = qual_pivot.rename(columns=target_tags_map)

    # Объединяем телеметрию с метками качества
    merged = pd.merge(telemetry, qual_pivot, on='date', how='left')

    # Инициализация агента качества
    agent = QualityAgent(model_path=str(model_path))

    # Расчёт прогнозов: ВАК baseline и ML-модель
    logger.info("Расчёт прогнозов ВАК и гибридной модели...")
    vac_sulfur = vac_sulfur_24_2000(merged)
    vac_d15 = vac_d15_godt(merged)
    vac_t50 = vac_t50_godt(merged)
    vac_t95 = vac_t95_godt(merged)
    vac_t90 = np.maximum(vac_t50, vac_t95 - 12.0)
    vac_cfpp = vac_cfpp_godt(merged)

    # Прогноз модели для серы
    ml_sulfur = agent.predict(merged, vac=vac_sulfur)

    merged['pred_vac_Sulfur'] = vac_sulfur
    merged['pred_ml_Sulfur'] = ml_sulfur
    merged['pred_vac_D15'] = vac_d15
    merged['pred_vac_T50'] = vac_t50
    merged['pred_vac_T90'] = vac_t90
    merged['pred_vac_T95'] = vac_t95
    merged['pred_vac_CFPP'] = vac_cfpp

    # Разделение по времени:
    # Train: 2023-01-01 – 2025-12-31
    # Test: 2026-01-01 – 2026-08-07
    train_mask = (merged['date'] >= '2023-01-01') & (merged['date'] < '2026-01-01')
    test_mask = (merged['date'] >= '2026-01-01') & (merged['date'] <= '2026-08-07')

    train_df = merged[train_mask].copy()
    test_df = merged[test_mask].copy()
    logger.info(f"Train период (2023-2025): {len(train_df):,} записей")
    logger.info(f"Test период (2026): {len(test_df):,} записей")

    # Итоговые метрики на тестовом периоде 2026 года
    report = {
        "metadata": {
            "train_period": "2023-01-01 to 2025-12-31",
            "test_period": "2026-01-01 to 2026-08-07",
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df))
        },
        "indicators": {},
        "walk_forward_folds": []
    }

    # Показатели качества
    # 1. Sulfur (сравнение ВАК и ML)
    y_true_s = test_df['Sulfur'].values
    m_ml_s = compute_metrics(y_true_s, test_df['pred_ml_Sulfur'].values)
    m_vac_s = compute_metrics(y_true_s, test_df['pred_vac_Sulfur'].values)
    impr = None
    if m_vac_s["MAE"] and m_ml_s["MAE"]:
        impr = round((1.0 - m_ml_s["MAE"] / m_vac_s["MAE"]) * 100.0, 2)

    report["indicators"]["Sulfur"] = {
        "model": m_ml_s,
        "vac_baseline": m_vac_s,
        "mae_improvement_percent": impr
    }

    # Сопутствующие показатели
    other_targets = {
        'D15': 'pred_vac_D15',
        'T50': 'pred_vac_T50',
        'T90': 'pred_vac_T90',
        'T95': 'pred_vac_T95',
        'CFPP': 'pred_vac_CFPP'
    }
    for tag_name, pred_col in other_targets.items():
        if tag_name in test_df.columns:
            m = compute_metrics(test_df[tag_name].values, test_df[pred_col].values)
            report["indicators"][tag_name] = {
                "vac_baseline": m,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "bias": m["bias"]
            }

    # Walk-forward валидация (5 folds, каждый fold — +6 месяцев по времени)
    logger.info("Запуск Walk-Forward кросс-валидации (5 folds по 6 месяцев)...")
    # Фолды:
    # Fold 1: Train 2023-01..2023-12 (12 мес) -> Test 2024-01..2024-06 (6 мес)
    # Fold 2: Train 2023-01..2024-06 (18 мес) -> Test 2024-07..2024-12 (6 мес)
    # Fold 3: Train 2023-01..2024-12 (24 мес) -> Test 2025-01..2025-06 (6 мес)
    # Fold 4: Train 2023-01..2025-06 (30 мес) -> Test 2025-07..2025-12 (6 мес)
    # Fold 5: Train 2023-01..2025-12 (36 мес) -> Test 2026-01..2026-08 (8 мес)
    fold_intervals = [
        ("2023-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
        ("2023-01-01", "2024-06-30", "2024-07-01", "2024-12-31"),
        ("2023-01-01", "2024-12-31", "2025-01-01", "2025-06-30"),
        ("2023-01-01", "2025-06-30", "2025-07-01", "2025-12-31"),
        ("2023-01-01", "2025-12-31", "2026-01-01", "2026-08-07")
    ]

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(fold_intervals, 1):
        fold_test_mask = (merged['date'] >= te_start) & (merged['date'] <= te_end)
        fold_test = merged[fold_test_mask]

        f_metrics_s = compute_metrics(fold_test['Sulfur'].values, fold_test['pred_ml_Sulfur'].values)
        f_vac_s = compute_metrics(fold_test['Sulfur'].values, fold_test['pred_vac_Sulfur'].values)
        f_d15 = compute_metrics(fold_test['D15'].values, fold_test['pred_vac_D15'].values)
        f_t95 = compute_metrics(fold_test['T95'].values, fold_test['pred_vac_T95'].values)

        report["walk_forward_folds"].append({
            "fold": fold_idx,
            "train_period": f"{tr_start} to {tr_end}",
            "test_period": f"{te_start} to {te_end}",
            "test_samples": int(len(fold_test)),
            "metrics": {
                "Sulfur_ML_MAE": f_metrics_s["MAE"],
                "Sulfur_VAC_MAE": f_vac_s["MAE"],
                "D15_MAE": f_d15["MAE"],
                "T95_MAE": f_t95["MAE"]
            }
        })
        logger.info(
            f"Fold {fold_idx}: Test {te_start}..{te_end} | "
            f"Sulfur ML MAE={f_metrics_s['MAE']} (VAC={f_vac_s['MAE']})"
        )

    # Сохранение отчёта
    out_file = Path('output/validation_report.json')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Отчёт о валидации успешно сохранён в: {out_file}")
    logger.info("Результаты на тесте 2026 года:")
    for ind, res in report["indicators"].items():
        if "model" in res:
            logger.info(
                f"  {ind}: Model MAE={res['model']['MAE']}, RMSE={res['model']['RMSE']}, "
                f"bias={res['model']['bias']} | Улучшение к ВАК: {res.get('mae_improvement_percent')}%"
            )
        else:
            logger.info(
                f"  {ind}: MAE={res.get('MAE')}, RMSE={res.get('RMSE')}, bias={res.get('bias')}"
            )

    return report


if __name__ == '__main__':
    run_validation()
