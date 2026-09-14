"""
Модуль: src/etl/sync_data.py
Назначение: Временная синхронизация непрерывной телеметрии (10-мин срезы) 
с асинхронными показателями качества (ЛИМС и ПАК) через merge_asof без заглядывания в будущее.

Блок: Data (DATA-04)

Контекст и правила работы с данными:
1. Синхронизация строго по времени (merge_asof, direction='backward').
   Запрещено заглядывать в будущее или сопоставлять данные по номеру строки.
2. Иерархия приоритетов источников качества:
   ЛИМС (лабораторный факт) > ПАК (поточный замер).
   Если в допустимом окне присутствуют оба источника, лабораторный результат
   подавляет показания поточного прибора.
3. Окна допустимого запаздывания (tolerance):
   - ЛИМС: tolerance = 60 минут (редкие замеры).
   - ПАК: tolerance = 15 минут (высокочастотные измерения).
4. Расчёт свежести данных (Data Freshness / age_min):
   age_min = (date_telemetry - timestamp_quality).total_seconds() / 60.
5. Флаг устаревания (stale):
   Если age_min > 240 минут (4 часа) или данные отсутствуют -> stale = True.
"""

from typing import Optional, List
import pandas as pd
import numpy as np


def compute_age_min(df: pd.DataFrame, date_col: str = 'date', ts_col: str = 'timestamp') -> pd.DataFrame:
    """
    Рассчитывает возраст замера качества в минутах относительно момента телеметрии
    и выставляет флаг устаревания данных (stale).

    Параметры:
        df (pd.DataFrame): Датафрейм, содержащий дату телеметрии и временную метку замера качества.
        date_col (str): Имя колонки даты телеметрии.
        ts_col (str): Имя колонки даты замера качества.

    Возвращает:
        pd.DataFrame: Датафрейм с добавленными колонками 'age_min' (float) и 'stale' (bool).
    """
    result = df.copy()

    if ts_col in result.columns and date_col in result.columns:
        date_series = pd.to_datetime(result[date_col], errors='coerce', utc=True)
        ts_series = pd.to_datetime(result[ts_col], errors='coerce', utc=True)
        # Расчет разницы во времени в минутах
        time_diff = (date_series - ts_series).dt.total_seconds() / 60.0
        result['age_min'] = time_diff

        # Флаг устаревания: если возраст > 240 минут (4 часа) или замер отсутствует (NaN)
        result['stale'] = (result['age_min'] > 240.0) | result['age_min'].isna()
    else:
        result['age_min'] = np.nan
        result['stale'] = True

    return result


def sync_telemetry_quality(
    telemetry: pd.DataFrame,
    quality: pd.DataFrame,
    tolerance_lims: int = 60,
    tolerance_pak: int = 15,
    tags: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Синхронизирует телеметрию с анализами качества ЛИМС и ПАК по временной шкале
    с соблюдением приоритета ЛИМС > ПАК.

    Параметры:
        telemetry (pd.DataFrame): Датафрейм телеметрии (содержит 'date' и теги).
        quality (pd.DataFrame): Long-format датафрейм качества (timestamp, tag, value, source, sample_point, unit).
        tolerance_lims (int): Максимальное окно поиска замера ЛИМС назад во времени (в минутах, по умолчанию 60).
        tolerance_pak (int): Максимальное окно поиска замера ПАК назад во времени (в минутах, по умолчанию 15).
        tags (Optional[List[str]]): Список показателей качества для синхронизации (по умолчанию все из quality).

    Возвращает:
        pd.DataFrame: Синхронизированный long-format датафрейм со столбцами:
                     ['date', 'tag', 'value_telemetry', 'value_quality', 'source', 'age_min', 'stale']
    """
    # 1. Проверка и сортировка временных шкал
    if 'date' not in telemetry.columns:
        raise ValueError("Телеметрия должна содержать колонку 'date'!")
    if 'timestamp' not in quality.columns:
        raise ValueError("Таблица качества должна содержать колонку 'timestamp'!")

    # Извлекаем временную сетку телеметрии
    timeline = pd.DataFrame({'date': pd.to_datetime(telemetry['date'], utc=True).drop_duplicates().sort_values()})

    # Список целевых показателей качества
    target_tags = tags if tags is not None else quality['tag'].unique().tolist()

    # Временные допуски merge_asof
    tol_lims_td = pd.Timedelta(minutes=tolerance_lims)
    tol_pak_td = pd.Timedelta(minutes=tolerance_pak)

    results = []

    # 2. Потековая синхронизация показателей качества
    for tag in target_tags:
        q_subset = quality[quality['tag'] == tag]
        if q_subset.empty:
            continue

        lims_df = q_subset[q_subset['source'] == 'LIMS'].sort_values('timestamp')
        pak_df = q_subset[q_subset['source'] == 'PAK'].sort_values('timestamp')

        # Слияние ЛИМС (tolerance=60 min)
        if not lims_df.empty:
            m_lims = pd.merge_asof(
                timeline,
                lims_df[['timestamp', 'value', 'source']],
                left_on='date',
                right_on='timestamp',
                tolerance=tol_lims_td,
                direction='backward'
            )
        else:
            m_lims = timeline.copy()
            m_lims['timestamp'] = pd.NaT
            m_lims['value'] = np.nan
            m_lims['source'] = np.nan

        # Слияние ПАК (tolerance=15 min)
        if not pak_df.empty:
            m_pak = pd.merge_asof(
                timeline,
                pak_df[['timestamp', 'value', 'source']],
                left_on='date',
                right_on='timestamp',
                tolerance=tol_pak_td,
                direction='backward'
            )
        else:
            m_pak = timeline.copy()
            m_pak['timestamp'] = pd.NaT
            m_pak['value'] = np.nan
            m_pak['source'] = np.nan

        # 3. Приоритет источников: ЛИМС > ПАК
        # combine_first берёт значение из первого датасета, а если там NaN — из второго
        val_quality = m_lims['value'].combine_first(m_pak['value'])
        src_quality = m_lims['source'].combine_first(m_pak['source'])
        ts_quality = m_lims['timestamp'].combine_first(m_pak['timestamp'])

        # Значение из телеметрии (если существует совпадающий тег в телеметрии)
        val_telemetry = telemetry[tag] if tag in telemetry.columns else np.nan

        tag_synced = pd.DataFrame({
            'date': timeline['date'],
            'tag': tag,
            'value_telemetry': val_telemetry,
            'value_quality': val_quality,
            'source': src_quality,
            'timestamp': ts_quality
        })

        results.append(tag_synced)

    if not results:
        empty_cols = ['date', 'tag', 'value_telemetry', 'value_quality', 'source', 'age_min', 'stale']
        return pd.DataFrame(columns=empty_cols)

    combined_df = pd.concat(results, ignore_index=True)

    # 4. Расчёт свежести и флага stale
    combined_df = compute_age_min(combined_df, date_col='date', ts_col='timestamp')
    combined_df = combined_df.drop(columns=['timestamp'])

    # Финальный порядок колонок
    cols_order = ['date', 'tag', 'value_telemetry', 'value_quality', 'source', 'age_min', 'stale']
    return combined_df[cols_order]
