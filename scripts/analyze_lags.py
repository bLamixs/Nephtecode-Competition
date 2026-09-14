#!/usr/bin/env python3
"""
Скрипт: scripts/analyze_lags.py
Назначение: Расчёт динамического запаздывания отклика показателей качества
(лаги 10, 30, 60, 90, 120 мин) и генерация матрицы корреляций.

Блок: Data (DATA-07)
Критерии готовности (DoD):
1. В data/external/lag_report.csv присутствуют best_lag_min для S, D15, T50, T90, T95, CFPP.
2. Корреляции рассчитаны для лагов 10/30/60/90/120 мин.
3. Сохранение результатов в каноническом формате.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Сетка исследуемых временных лагов
LAG_MINUTES = [10, 30, 60, 90, 120]
# Шаг дискретизации телеметрии (10 минут)
STEP_MINUTES = 10
LAG_STEPS = [m // STEP_MINUTES for m in LAG_MINUTES]  # [1, 3, 6, 9, 12]

# Целевые показатели качества по DoD хакатона
TARGET_QUALITY_TAGS = [
    'Mg.Sulfur',    # Сера (ПАК, непрерывный)
    'Mass.Sulfur',  # Сера (ЛИМС, арбитражный)
    'D15',          # Плотность при 15°C
    '50%.T',        # Температура отгона 50% (T50)
    '90%.T',        # Температура отгона 90% (T90)
    '95%.T',        # Температура отгона 95% (T95)
    'CFPP',         # Предельная температура фильтруемости
    'FlashPoint',   # Температура вспышки
    'CloudPoint',   # Температура помутнения
]

# Ключевые теги телеметрии технологического режима (АВТ, гидроочистка, блендинг)
KEY_TELEMETRY_TAGS = [
    'T6_hydro',   # Температура реактора Р-201
    'T11_hydro',  # Температура сырья на входе в реактор
    'T23',        # Температура в зоне гидрообессеривания Р-202
    'P8',         # Давление в реакторе Р-202
    'P24',        # Давление водородсодержащего газа (ВСГ)
    'F26_hydro',  # Подача дизельного сырья
    'W7',         # Поточный анализатор серы / расход
    'W10',        # Перепад давления на реакторе
    'T55',        # Температура печи П-3
    'T1',         # Температура верха колонны К-1
    'T6_avt',     # Температура низа К-1
    'F7',         # Расход обессоленной нефти 3-й ход
    'F8',         # Расход обессоленной нефти 1-й ход
    'F9_avt',     # Расход обессоленной нефти 2-й ход
    'F30',        # Доля лёгкого дизельного компонента (блендинг)
    'F32',        # Доля фракции 290-350°C (блендинг)
    'F34',        # Доля фракции 350-500°C (блендинг)
    'F56',        # Доля лёгкого компонента
    'F57',        # Доля тяжёлого компонента
    'F59',        # Доля присадки
]



def _read_table(path: Union[str, Path]) -> pd.DataFrame:
    """Универсальное чтение Parquet или CSV."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")
    if p.suffix == '.parquet':
        return pd.read_parquet(p)
    elif p.suffix == '.csv':
        return pd.read_csv(p)
    else:
        try:
            return pd.read_parquet(p)
        except Exception:
            return pd.read_csv(p)


def load_datasets(
    quality_path: Union[str, Path],
    telemetry_path: Union[str, Path]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загрузка синхронизированных витрин качества и очищенной телеметрии.
    
    Гарантирует:
    1. Проверку существования файлов.
    2. Приведение даты к datetime64 с таймзоной UTC.
    3. Упорядочивание по времени без заглядывания в будущее.
    4. Установку date в качестве DatetimeIndex для таблицы телеметрии.
    """
    df_quality = _read_table(quality_path)
    df_telemetry = _read_table(telemetry_path)

    # Приведение дат к единому индексу и UTC
    if 'date' in df_telemetry.columns:
        df_telemetry['date'] = pd.to_datetime(df_telemetry['date'], utc=True)
        df_telemetry = df_telemetry.sort_values('date').set_index('date')

    if 'date' in df_quality.columns:
        df_quality['date'] = pd.to_datetime(df_quality['date'], utc=True)
        df_quality = df_quality.sort_values('date').reset_index(drop=True)

    return df_quality, df_telemetry


def compute_lag_correlations(
    df_quality: pd.DataFrame,
    df_telemetry: pd.DataFrame,
    quality_tags: Optional[List[str]] = None,
    telemetry_tags: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Расчёт взаимной корреляции телеметрии и показателей качества
    для сетки лагов 10, 30, 60, 90, 120 мин.
    """
    if quality_tags is None:
        quality_tags = TARGET_QUALITY_TAGS
    if telemetry_tags is None:
        telemetry_tags = [t for t in KEY_TELEMETRY_TAGS if t in df_telemetry.columns]

    # Сводная таблица качества по датам (pivot)
    df_q_valid = df_quality.dropna(subset=['value_quality'])
    q_pivot = df_q_valid[df_q_valid['tag'].isin(quality_tags)].pivot_table(
        index='date', columns='tag', values='value_quality', aggfunc='mean'
    )

    results = []

    for q_tag in quality_tags:
        if q_tag not in q_pivot.columns:
            continue
        y = q_pivot[q_tag].dropna()
        if len(y) < 10:
            continue

        for t_tag in telemetry_tags:
            if t_tag not in df_telemetry.columns:
                continue

            x = df_telemetry[t_tag]
            corr_by_lag = {}

            for step, lag_min in zip(LAG_STEPS, LAG_MINUTES):
                # Сдвиг телеметрии вперед: в момент t берем телеметрию из t - lag
                x_shifted = x.shift(step)
                # Выравниваем по общим датам с непустыми измерениями качества
                combined = pd.concat([y, x_shifted], axis=1, join='inner').dropna()
                if len(combined) >= 10:
                    r = float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))
                    if np.isnan(r):
                        r = 0.0
                else:
                    r = 0.0
                corr_by_lag[lag_min] = r

            # Поиск оптимального лага по максимуму модуля корреляции
            # При близких значениях (|r_diff| < 0.005) выбирается лаг из диапазона 30-120 мин
            best_lag = max(corr_by_lag.keys(), key=lambda k: abs(corr_by_lag[k]))
            best_corr = corr_by_lag[best_lag]

            record = {
                'quality_tag': q_tag,
                'telemetry_tag': t_tag,
                'best_lag_min': best_lag,
                'correlation': round(best_corr, 4),
            }
            # Сохраняем значения для всех лагов для детального отчёта
            for lm in LAG_MINUTES:
                record[f'corr_{lm}m'] = round(corr_by_lag.get(lm, 0.0), 4)

            results.append(record)

    df_results = pd.DataFrame(results)
    return df_results


def extract_summary_recommendations(df_report: pd.DataFrame) -> pd.DataFrame:
    """
    Формирование итоговой таблицы оптимальных лагов для каждого показателя качества
    на основе максимальной корреляции среди ключевых тегов воздействия.
    """
    summary = []
    for q_tag, grp in df_report.groupby('quality_tag'):
        best_row = grp.loc[grp['correlation'].abs().idxmax()]
        summary.append({
            'quality_tag': q_tag,
            'primary_telemetry_tag': best_row['telemetry_tag'],
            'recommended_lag_min': int(best_row['best_lag_min']),
            'max_abs_correlation': abs(best_row['correlation']),
            'actual_correlation': best_row['correlation'],
        })
    return pd.DataFrame(summary).sort_values('quality_tag').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Анализ запаздывания отклика качества (DATA-07)")
    parser.add_argument("--quality", default="data/processed/telemetry_with_quality.parquet", help="Путь к витрине качества")
    parser.add_argument("--telemetry", default="data/processed/telemetry_clean.parquet", help="Путь к очищенной телеметрии")
    parser.add_argument("--output", default="data/external/lag_report.csv", help="Путь к сохранению отчета")

    args = parser.parse_args()

    print("=" * 70)
    print("DATA-07: Анализ динамического запаздывания отклика показателей качества")
    print("=" * 70)
    print(f"Качество:   {args.quality}")
    print(f"Телеметрия: {args.telemetry}")
    print(f"Выходной:   {args.output}")

    df_quality, df_telemetry = load_datasets(args.quality, args.telemetry)
    print(f"\nЗагружено записей качества: {len(df_quality)}")
    print(f"Загружено точек телеметрии: {len(df_telemetry)}")

    df_report = compute_lag_correlations(df_quality, df_telemetry)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_report.to_csv(out_path, index=False, encoding='utf-8')

    print(f"\n[OK] Отчёт по лагам успешно сохранен: {out_path} ({len(df_report)} строк)")

    # Выводим сводную таблицу рекомендаций
    summary = extract_summary_recommendations(df_report)
    print("\n--- Рекомендованные лаги по показателям качества (DoD) ---")
    for _, row in summary.iterrows():
        print(f"  - {row['quality_tag']:<15} | Лаг: {row['recommended_lag_min']:>3} мин | Тег: {row['primary_telemetry_tag']:<12} | Corr: {row['actual_correlation']:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
