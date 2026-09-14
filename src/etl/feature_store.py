"""
Модуль: src/etl/feature_store.py
Назначение: Генерация витрины признаков (Feature Store) для ML-агентов:
- Лаги технологического режима (10, 30, 60 мин) без заглядывания в будущее.
- Скользящие статистические окна (rolling mean, rolling std за 60 шагов).
- Z-Score относительно технологических норм из справочника (DATA-06).

Блок: Data (DATA-05)
Зависимости: DATA-04 (telemetry_with_quality), DATA-06 (tag_dict).
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd

from src.config import load_tag_dict


# Ключевые режимные теги (Топ-20 параметров АВТ, гидроочистки и блендинга)
TOP_20_KEY_TAGS = [
    'T6_hydro',   # Температура реактора гидроочистки Р-201
    'T11_hydro',  # Температура газосырьевой смеси на входе
    'T23',        # Температура катализаторного слоя Р-202
    'P8',         # Давление в реакторе Р-202
    'P24',        # Давление водородсодержащего газа (ВСГ)
    'F26_hydro',  # Подача дизельного сырья (объемная скорость WHSV)
    'W7',         # Поточный анализатор серы / расход
    'W10',        # Перепад давления на реакторе Р-202
    'T55',        # Температура печи П-3 (блок АВТ)
    'T1',         # Температура верха колонны К-1
    'T6_avt',     # Температура низа колонны К-1
    'F7',         # Расход обессоленной нефти (3-й ход)
    'F8',         # Расход обессоленной нефти (1-й ход)
    'F9_avt',     # Расход обессоленной нефти (2-й ход)
    'F30',        # Доля фракции 240-290°C (блендинг)
    'F32',        # Доля фракции 290-350°C (блендинг)
    'F34',        # Доля фракции 350-500°C (блендинг)
    'F56',        # Доля лёгкого компонента (блендинг)
    'F57',        # Доля тяжёлого компонента (блендинг)
    'F59',        # Доля присадки в блендинге
]


def _lag_to_step(lag: int, step_min: int = 10) -> int:
    """Преобразование минутного лага в количество шагов дискретизации."""
    return lag // step_min if lag >= step_min else lag


def generate_lag_features(
    df: pd.DataFrame,
    tags: Optional[List[str]] = None,
    lags: List[int] = [10, 30, 60],
    value_col: str = 'value',
    step_min: int = 10,
) -> pd.DataFrame:
    """
    Генерация лаговых признаков (сдвиг назад во времени) без заглядывания в будущее.
    
    Поддерживает:
    - Long-format (колонки: date, tag, value)
    - Wide-format (колонки: date, tag1, tag2, ...)
    """
    res = df.copy()

    # Long-format случай (колонки date, tag, value)
    if 'tag' in res.columns and value_col in res.columns:
        if tags is not None:
            res = res[res['tag'].isin(tags)].copy()

        # Сортировка по дате внутри каждого тега
        res = res.sort_values(['tag', 'date']).reset_index(drop=True)

        grouped = res.groupby('tag')[value_col]
        for lag in lags:
            step = _lag_to_step(lag, step_min)
            # Добавляем стандартную колонку lag{min}
            res[f'lag{lag}'] = grouped.shift(step)
            # Также формируем колонку {tag}_lag{min} для полного соответствия DoD
            # в Long-формате это дублируется для удобства конкатенации
        return res

    # Wide-format случай
    if tags is None:
        tags = [c for c in res.columns if c != 'date']

    for tag in tags:
        if tag in res.columns:
            for lag in lags:
                step = _lag_to_step(lag, step_min)
                res[f'{tag}_lag{lag}'] = res[tag].shift(step)

    return res


def generate_rolling_features(
    df: pd.DataFrame,
    tags: Optional[List[str]] = None,
    window: int = 60,
    value_col: str = 'value',
) -> pd.DataFrame:
    """
    Генерация скользящих статистических признаков (rolling mean, rolling std)
    за окно в window шагов (по умолчанию 60 шагов = 10 часов).
    """
    res = df.copy()

    # Long-format случай
    if 'tag' in res.columns and value_col in res.columns:
        if tags is not None:
            res = res[res['tag'].isin(tags)].copy()

        res = res.sort_values(['tag', 'date']).reset_index(drop=True)
        grouped = res.groupby('tag')[value_col]

        # Скользящее среднее
        res[f'rolling_mean_{window}'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        # Скользящее стандартное отклонение с защитой от 0
        res[f'rolling_std_{window}'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(1e-6)
        )
        return res

    # Wide-format случай
    if tags is None:
        tags = [c for c in res.columns if c != 'date']

    for tag in tags:
        if tag in res.columns:
            res[f'{tag}_rolling_mean_{window}'] = res[tag].rolling(window, min_periods=1).mean()
            res[f'{tag}_rolling_std_{window}'] = res[tag].rolling(window, min_periods=1).std().fillna(1e-6)

    return res


def generate_zscore_features(
    df: pd.DataFrame,
    tags: Optional[List[str]] = None,
    norm_mean: Optional[float] = None,
    norm_std: Optional[float] = None,
    tag_dict_df: Optional[pd.DataFrame] = None,
    value_col: str = 'value',
) -> pd.DataFrame:
    """
    Генерация признаков стандартизации Z-score:
    zscore = (value - norm_mean) / norm_std
    
    ВНИМАНИЕ (DoD): norm_mean и norm_std берутся строго из справочника tag_dict (DATA-06),
    а не по всему текущему датасету!
    """
    res = df.copy()

    if tag_dict_df is None and (norm_mean is None or norm_std is None):
        tag_dict_df = load_tag_dict()

    # Быстрый словарь норм из tag_dict: tag -> (norm_mean, norm_std)
    norms_map: Dict[str, Tuple[float, float]] = {}
    if tag_dict_df is not None:
        for _, row in tag_dict_df.iterrows():
            t = row['tag']
            m = row.get('norm_mean', np.nan)
            s = row.get('norm_std', np.nan)
            if pd.notna(m) and pd.notna(s) and s > 0:
                norms_map[t] = (float(m), float(s))

    # Long-format случай
    if 'tag' in res.columns and value_col in res.columns:
        if tags is not None:
            res = res[res['tag'].isin(tags)].copy()

        def _calc_z(row):
            t = row['tag']
            val = row[value_col]
            if pd.isna(val):
                return np.nan
            if norm_mean is not None and norm_std is not None:
                std_val = max(norm_std, 1e-6)
                return (val - norm_mean) / std_val
            if t in norms_map:
                m, s = norms_map[t]
                return (val - m) / max(s, 1e-6)
            return 0.0

        res['zscore'] = res.apply(_calc_z, axis=1)
        return res

    # Wide-format случай
    if tags is None:
        tags = [c for c in res.columns if c != 'date']

    for tag in tags:
        if tag in res.columns:
            if norm_mean is not None and norm_std is not None:
                m, s = norm_mean, max(norm_std, 1e-6)
            elif tag in norms_map:
                m, s = norms_map[tag]
            else:
                m, s = 0.0, 1.0
            res[f'{tag}_zscore'] = (res[tag] - m) / max(s, 1e-6)

    return res


def build_feature_store(
    df: pd.DataFrame,
    tag_dict_df: Optional[pd.DataFrame] = None,
    tags: Optional[List[str]] = None,
    lags: List[int] = [10, 30, 60],
    window: int = 60,
    value_col: str = 'value',
) -> pd.DataFrame:
    """
    Комплексный конвейер генерации витрины признаков:
    1. Расчёт лагов (10, 30, 60 мин).
    2. Скользящие окна (rolling mean, rolling std).
    3. Z-score по нормам из справочника DATA-06.
    """
    if tag_dict_df is None:
        tag_dict_df = load_tag_dict()

    if tags is None:
        tags = TOP_20_KEY_TAGS

    # Проверка формата: если есть value_telemetry или value_quality, создаем колонку value
    df_work = df.copy()
    if 'value' not in df_work.columns:
        if 'value_telemetry' in df_work.columns and 'value_quality' in df_work.columns:
            df_work['value'] = df_work['value_telemetry'].combine_first(df_work['value_quality'])
        elif 'value_telemetry' in df_work.columns:
            df_work['value'] = df_work['value_telemetry']
        elif 'value_quality' in df_work.columns:
            df_work['value'] = df_work['value_quality']

    # 1. Лаги
    res = generate_lag_features(df_work, tags=tags, lags=lags, value_col=value_col)

    # 2. Скользящие окна
    res = generate_rolling_features(res, tags=tags, window=window, value_col=value_col)

    # 3. Z-Score
    res = generate_zscore_features(res, tags=tags, tag_dict_df=tag_dict_df, value_col=value_col)

    # В Long-формате добавляем колонки с префиксом тега ({tag}_lag10, {tag}_rolling_mean_60, {tag}_zscore)
    if 'tag' in res.columns:
        res['tag_lag10'] = res['lag10'] if 'lag10' in res.columns else np.nan
        res['tag_lag30'] = res['lag30'] if 'lag30' in res.columns else np.nan
        res['tag_rolling_mean_60'] = res[f'rolling_mean_{window}'] if f'rolling_mean_{window}' in res.columns else np.nan
        res['tag_zscore'] = res['zscore'] if 'zscore' in res.columns else np.nan

    return res


def save_features_parquet(df: pd.DataFrame, output_path: Union[str, Path]):
    """Сохранение витрины признаков в формате Parquet (Snappy compression)."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, compression='snappy', index=False)
