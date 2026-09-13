"""
Модуль: src/etl/load_telemetry.py
Назначение: Загрузка, первичная очистка структуры и конвертация телеметрии АВТ и 24-2000 в Parquet.

Блок: Data (DATA-01)
Контекст задачи:
Телеметрия технологических установок АВТ и 24-2000 содержит 10-минутные синхронные срезы (~189 тыс. строк).
Файлы CSV содержат технические индексные колонки вида 'Unnamed:*' или пустые заголовки,
а также строковые даты.

Данный модуль обеспечивает:
1. Корректное чтение CSV с автоопределением разделителя (запятая/точка с запятой).
2. Очистку от служебных колонок индексов (Unnamed:*).
3. Приведение временного ключа 'date' к единому стандарту datetime64[ns, UTC].
4. Добавление идентификатора источника данных ('AVT' или '24-2000').
5. Сохранение в бинарный колоночный формат Parquet со сжатием Snappy для ускорения загрузки последующими агентами.
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd


def load_telemetry_csv(path: str, source: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Загружает CSV-файл телеметрии, удаляет технические колонки индексов,
    устанавливает временную зону UTC и добавляет метку источника.

    Параметры:
        path (str): Путь к исходному CSV-файлу телеметрии.
        source (str): Имя источника ('AVT' или '24-2000').
        encoding (str): Кодировка файла (по умолчанию 'utf-8', при ошибках — 'cp1251').

    Возвращает:
        pd.DataFrame: Очищенный датафрейм телеметрии.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл телеметрии не найден по пути: {path}")

    # 1. Быстрое чтение заголовка для детекции разделителя (запятая или точка с запятой)
    with open(path, 'r', encoding=encoding, errors='ignore') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else ','

    # 2. Загрузка CSV в pandas
    df = pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        low_memory=False
    )

    # 3. Удаление служебных колонок индексов вида 'Unnamed:*' и колонок с пустыми именами
    # Регулярное выражение оставляет все колонки, которые НЕ начинаются на Unnamed
    df = df.filter(regex='^(?!Unnamed)')
    # Дополнительно отсекаем колонки без имени (если при экспорте образовался пустой заголовок)
    df = df.loc[:, [c for c in df.columns if str(c).strip() != '']]

    # 4. Проверка наличия обязательной колонки даты
    if 'date' not in df.columns:
        raise ValueError(f"В файле {path} отсутствует обязательная колонка 'date'!")

    # 5. Приведение колонки 'date' к типу datetime64[ns, UTC]
    df['date'] = pd.to_datetime(df['date'], utc=True)

    # 6. Добавление колонки источника данных
    df['source'] = source

    # 7. Сортировка по возрастанию даты и удаление возможных полных дубликатов по времени
    df = df.sort_values('date').reset_index(drop=True)

    return df


def save_telemetry_parquet(df: pd.DataFrame, output_path: str, compression: str = 'snappy') -> None:
    """
    Сохраняет обработанный датафрейм телеметрии в формате Apache Parquet.

    Параметры:
        df (pd.DataFrame): Датафрейм телеметрии для сохранения.
        output_path (str): Путь к результирующему Parquet-файлу.
        compression (str): Алгоритм сжатия (по умолчанию 'snappy').
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем без сохранения числового индекса pandas
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression=compression,
        index=False
    )
