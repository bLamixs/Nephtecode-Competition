"""
Модуль: src/etl/load_quality.py
Назначение: Загрузка и приведение асинхронных данных качества (ЛИМС и ПАК)
из многоколоночных отчётов Excel в канонический формат Long-Format.

Блок: Data (DATA-03)

Контекст задачи:
1. ЛИМС (лабораторная информационная система): контрольные эталонные замеры качества
   (редкие, раз в смену или сутки, по разным точкам отбора: АВТ, гидроочистка).
2. ПАК (поточные анализаторы качества): оперативные измерения серы и плотности
   (частые, каждые 10 минут, но могут сбоить).

Формат Long-Format (длинный формат) необходим для:
- Унифицированного хранения разнородных показателей.
- Корректного последующего слияния с 10-минутными рядами телеметрии через pd.merge_asof.
- Расчёта свежести замера (age_min) для каждого конкретного показателя качества.

Выходная структура колонок:
[timestamp, tag, value, source, sample_point, unit]
"""

import os
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import numpy as np


def parse_lims_xlsx(path: str) -> pd.DataFrame:
    """
    Парсит файл отчёта ЛИМС (ЛИМСы.xlsx) по всем листам и точкам отбора.
    Преобразует блочную структуру (пара колонок 'Дата' + 'Значение') в длинный формат.

    Параметры:
        path (str): Путь к файлу ЛИМСы.xlsx.

    Возвращает:
        pd.DataFrame: Long-format датафрейм со столбцами:
                     ['timestamp', 'tag', 'value', 'source', 'sample_point', 'unit']
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл ЛИМС не найден: {path}")

    # Читаем все листы книги Excel
    excel_file = pd.ExcelFile(path, engine='openpyxl')
    all_records = []

    for sheet_name in excel_file.sheet_names:
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        if df_sheet.shape[0] < 5 or df_sheet.shape[1] < 2:
            continue

        # Строка 0: Точка отбора (может быть объединена над несколькими тегами)
        row_point = df_sheet.iloc[0].ffill()
        # Строка 1: Имя показателя качества (тег)
        row_tag = df_sheet.iloc[1]
        # Строка 2: Единица измерения (°C, мг/кг, кг/м3 и т.д.)
        row_unit = df_sheet.iloc[2]

        # Каждое измерение состоит из двух колонок: 2*i (дата) и 2*i + 1 (значение)
        for col_idx in range(0, df_sheet.shape[1], 2):
            point = str(row_point[col_idx]).strip() if pd.notna(row_point[col_idx]) else "Unknown"
            tag = str(row_tag[col_idx]).strip() if pd.notna(row_tag[col_idx]) else None
            unit = str(row_unit[col_idx]).strip() if pd.notna(row_unit[col_idx]) else ""

            if not tag:
                continue

            # Данные измерений начинаются с 4-й строки (индекс 4)
            sub_df = df_sheet.iloc[4:, [col_idx, col_idx + 1]].dropna()
            if sub_df.empty:
                continue

            sub_df.columns = ['timestamp', 'value']
            # Конвертируем значение в число, отсекая текстовые пометки
            sub_df['value'] = pd.to_numeric(sub_df['value'], errors='coerce')
            sub_df = sub_df.dropna(subset=['value'])

            # Конвертируем дату в datetime64[ns, UTC]
            sub_df['timestamp'] = pd.to_datetime(sub_df['timestamp'], errors='coerce', utc=True)
            sub_df = sub_df.dropna(subset=['timestamp'])

            if sub_df.empty:
                continue

            sub_df['sample_point'] = point
            sub_df['tag'] = tag
            sub_df['unit'] = unit
            sub_df['source'] = 'LIMS'

            all_records.append(sub_df)

    if not all_records:
        return pd.DataFrame(columns=['timestamp', 'tag', 'value', 'source', 'sample_point', 'unit'])

    result_df = pd.concat(all_records, ignore_index=True)
    # Удаляем возможные полные дубликаты
    result_df = result_df.drop_duplicates(subset=['timestamp', 'tag', 'sample_point']).reset_index(drop=True)
    return result_df


def parse_pak_xlsx(path: str) -> pd.DataFrame:
    """
    Парсит файл поточных анализаторов качества (ПАК.xlsx / Выгрузка ПАК.xlsx).
    Извлекает поточные временные ряды серы (Mg.Sulfur) и плотности (D15).

    Параметры:
        path (str): Путь к файлу ПАК.xlsx.

    Возвращает:
        pd.DataFrame: Long-format датафрейм со столбцами:
                     ['timestamp', 'tag', 'value', 'source', 'sample_point', 'unit']
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл ПАК не найден: {path}")

    excel_file = pd.ExcelFile(path, engine='openpyxl')
    all_records = []

    for sheet_name in excel_file.sheet_names:
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
        if df_sheet.empty:
            continue

        # В структуре ПАК:
        # Колонки 0, 1: Сера (Mg.Sulfur)
        # Колонки 3, 4: Плотность (D15)
        cols = df_sheet.columns.tolist()

        # 1. Блок серы (колонки 0 и 1)
        if len(cols) >= 2:
            unit_s = str(df_sheet.iloc[0, 0]).strip() if pd.notna(df_sheet.iloc[0, 0]) else "ppm"
            df_s = df_sheet.iloc[1:, [0, 1]].dropna()
            df_s.columns = ['timestamp', 'value']
            df_s['value'] = pd.to_numeric(df_s['value'], errors='coerce')
            df_s = df_s.dropna(subset=['value'])
            df_s['timestamp'] = pd.to_datetime(df_s['timestamp'], errors='coerce', utc=True)
            df_s = df_s.dropna(subset=['timestamp'])

            df_s['tag'] = 'Mg.Sulfur'
            df_s['sample_point'] = '24-2000'
            df_s['unit'] = unit_s
            df_s['source'] = 'PAK'
            all_records.append(df_s)

        # 2. Блок плотности D15 (колонки 3 и 4 при наличии)
        if len(cols) >= 5:
            unit_d = str(df_sheet.iloc[0, 3]).strip() if pd.notna(df_sheet.iloc[0, 3]) else "кг/м3"
            df_d = df_sheet.iloc[1:, [3, 4]].dropna()
            df_d.columns = ['timestamp', 'value']
            df_d['value'] = pd.to_numeric(df_d['value'], errors='coerce')
            df_d = df_d.dropna(subset=['value'])
            df_d['timestamp'] = pd.to_datetime(df_d['timestamp'], errors='coerce', utc=True)
            df_d = df_d.dropna(subset=['timestamp'])

            df_d['tag'] = 'D15'
            df_d['sample_point'] = '24-2000'
            df_d['unit'] = unit_d
            df_d['source'] = 'PAK'
            all_records.append(df_d)

    if not all_records:
        return pd.DataFrame(columns=['timestamp', 'tag', 'value', 'source', 'sample_point', 'unit'])

    result_df = pd.concat(all_records, ignore_index=True)
    result_df = result_df.drop_duplicates(subset=['timestamp', 'tag', 'sample_point']).reset_index(drop=True)
    return result_df


def load_all_quality_data(lims_path: str, pak_path: str) -> pd.DataFrame:
    """
    Объединяет все показатели качества из ЛИМС и ПАК в единую long-format витрину.

    Параметры:
        lims_path (str): Путь к файлу ЛИМС.
        pak_path (str): Путь к файлу ПАК.

    Возвращает:
        pd.DataFrame: Полный long-format датасет качества, отсортированный по timestamp.
    """
    df_lims = parse_lims_xlsx(lims_path)
    df_pak = parse_pak_xlsx(pak_path)

    combined = pd.concat([df_lims, df_pak], ignore_index=True)
    combined = combined.drop_duplicates(subset=['timestamp', 'tag', 'sample_point', 'source'])
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    # Гарантируем требуемый порядок колонок
    expected_cols = ['timestamp', 'tag', 'value', 'source', 'sample_point', 'unit']
    return combined[expected_cols]
