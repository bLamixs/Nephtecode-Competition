"""
Модуль: src/etl/clean_telemetry.py
Назначение: Детекция аномалий/выбросов на основе локального скользящего Z-Score
и формирование бинарных масок качества технологических данных.

Блок: Data (DATA-02)

Контекст задачи:
Телеметрия нефтеперерабатывающих установок содержит сбои датчиков КИП:
- Резкие ложные импульсы (спайки) из-за наводок или перепадов питания.
- Зависания и провалы в ноль при отключении приборов.
- Пропуски данных (NaN).

Алгоритм:
1. Для каждого числового тега рассчитывается скользящее среднее (rolling_mean)
   и стандартное отклонение (rolling_std) по окну в 60 шагов (10 часов).
2. Защита от нулевого отклонения: если датчик константен, std заменяется на 1e-6.
3. Локальный Z-Score: Z = (value - rolling_mean) / rolling_std.
4. Выбросом считается точка с |Z| > threshold (по умолчанию threshold = 4.0).
5. Формируются маски качества ({tag}_mask): 1 — значение достоверно, 0 — выброс или NaN.
6. В очищенном датасете (telemetry_clean) все выбросы заменяются на NaN.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def detect_outliers_zscore(
    df: pd.DataFrame,
    window: int = 60,
    threshold: float = 4.0,
    exclude_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Вычисляет флаги выбросов на основе локального скользящего Z-score.

    Параметры:
        df (pd.DataFrame): Исходный датафрейм телеметрии.
        window (int): Размер скользящего окна в шагах (60 шагов * 10 мин = 600 мин / 10 часов).
        threshold (float): Порог z-score (по умолчанию 4.0 сигма).
        exclude_cols (List[str]): Список колонок, исключаемых из детекции (например, 'date', 'source').

    Возвращает:
        pd.DataFrame: Булев датафрейм той же размерности для числовых колонок,
                     где True означает, что точка является выбросом (аномалией).
    """
    if exclude_cols is None:
        exclude_cols = ['date', 'source']

    # Отбираем только числовые колонки за вычетом служебных
    numeric_cols = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    outliers_df = pd.DataFrame(index=df.index)

    # Обрабатываем колонки (пакетно для контроля памяти)
    for col in numeric_cols:
        series = df[col].astype(float)
        
        # Скользящее среднее и стандартное отклонение по времени
        rolling_obj = series.rolling(window=window, min_periods=1)
        r_mean = rolling_obj.mean()
        r_std = rolling_obj.std().fillna(0.0)

        # Защита от деления на ноль при константных сигналах
        r_std = r_std.apply(lambda s: 1e-6 if s < 1e-6 else s)

        # Локальный z-score
        z_score = (series - r_mean) / r_std

        # Выброс: абсолютное значение z-score превышает порог
        # NaN в исходных данных не считается z-выбросом (он будет обработан в маске)
        is_outlier = (z_score.abs() > threshold) & series.notna()
        outliers_df[col] = is_outlier

    return outliers_df


def create_quality_masks(
    df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Формирует датафрейм бинарных масок качества для каждого тега.
    
    Правило маски:
    - 1: данные качественные и пригодны для обучения/рекомендаций.
    - 0: значение отсутствует (NaN) или идентифицировано как выброс (|Z| > threshold).

    Параметры:
        df (pd.DataFrame): Исходный датафрейм телеметрии.
        outliers_df (pd.DataFrame): Булев датафрейм выбросов от detect_outliers_zscore.
        date_col (str): Имя колонки даты для сохранения временного ключа.

    Возвращает:
        pd.DataFrame: Датафрейм с колонкой даты и колонками {tag}_mask (int8: 0 или 1).
    """
    masks_df = pd.DataFrame(index=df.index)
    if date_col in df.columns:
        masks_df[date_col] = df[date_col]

    for col in outliers_df.columns:
        # Качественные данные = НЕ пропуск И НЕ выброс
        valid_mask = df[col].notna() & (~outliers_df[col])
        masks_df[f"{col}_mask"] = valid_mask.astype(np.int8)

    return masks_df


def clean_telemetry_outliers(
    df: pd.DataFrame,
    outliers_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Создает очищенную копию датафрейма телеметрии, в которой все аномальные выбросы
    заменяются на NaN (для последующей обработки или интерполяции в DATA-04).

    Параметры:
        df (pd.DataFrame): Исходный датафрейм телеметрии.
        outliers_df (pd.DataFrame): Булев датафрейм выбросов.

    Возвращает:
        pd.DataFrame: Очищенный датафрейм с NaN на месте выбросов.
    """
    cleaned_df = df.copy()

    for col in outliers_df.columns:
        if col in cleaned_df.columns:
            # Заменяем точки-выбросы на NaN
            outlier_indices = outliers_df[col]
            cleaned_df.loc[outlier_indices, col] = np.nan

    return cleaned_df
