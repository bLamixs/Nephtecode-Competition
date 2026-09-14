#!/usr/bin/env python3
"""
Скрипт: scripts/parse_tag_dict.py
Назначение: Парсинг справочника тегов из Теги_хакатон.xlsx, расчет статистических
норм по очищенной телеметрии (telemetry_clean.parquet), разметка управляемых
параметров и генерация tag_dict.csv, tag_dict.parquet, constraints.yaml.

Блок: Data (DATA-06)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import json
import openpyxl

# Теги с совпадающими именами между установками АВТ и 24-2000
COLLISION_TAGS = {'T6', 'F9', 'T11', 'F14', 'T18', 'F19', 'F25', 'F26'}

# Управляемые параметры и их диапазоны (согласовано с Optimizer-01 и ТЗ)
CONTROLLED_PARAMS_SPECS = {
    # Секция АВТ
    'F7': {'min': 80.0, 'max': 160.0, 'current': 120.0, 'unit': 'т/ч', 'desc': 'Расход обессоленной нефти (3-й ход)'},
    'F8': {'min': 80.0, 'max': 160.0, 'current': 120.0, 'unit': 'т/ч', 'desc': 'Расход обессоленной нефти (1-й ход)'},
    'F9_avt': {'min': 225.0, 'max': 275.0, 'current': 250.0, 'unit': 'т/ч', 'desc': 'Расход обессоленной нефти на гидроочистку'},
    'T55': {'min': 315.0, 'max': 330.0, 'current': 320.0, 'unit': '°C', 'desc': 'Температура на выходе из печи П-3 (П-1/1)'},

    # Секция 24-2000 (гидроочистка)
    'T6_hydro': {'min': 290.0, 'max': 305.0, 'current': 295.0, 'unit': '°C', 'desc': 'Температура реактора гидроочистки (Р-201)'},
    'T11_hydro': {'min': 320.0, 'max': 375.0, 'current': 345.0, 'unit': '°C', 'desc': 'Температура ГСС на входе в реактор Р-201'},
    'T23': {'min': 330.0, 'max': 380.0, 'current': 355.0, 'unit': '°C', 'desc': 'Температура в зоне гидрообессеривания реактора Р-202'},
    'F26_hydro': {'min': 150.0, 'max': 260.0, 'current': 200.0, 'unit': 'т/ч', 'desc': 'Расход сырья на установку (объемный/массовый)'},
    'P8': {'min': 25.0, 'max': 40.0, 'current': 32.0, 'unit': 'кгс/см2', 'desc': 'Давление в реакторе Р-202 на входе'},
    'P24': {'min': 20.0, 'max': 38.0, 'current': 28.0, 'unit': 'кгс/см2', 'desc': 'Расход/давление свежего ВСГ с КЦА'},

    # Секция блендинга (доли фракций, сумма = 1.0)
    'F30': {'min': 0.25, 'max': 0.35, 'current': 0.30, 'unit': 'доля', 'desc': 'Доля фракции 240-290°C (290-350°C)'},
    'F32': {'min': 0.20, 'max': 0.30, 'current': 0.25, 'unit': 'доля', 'desc': 'Доля фракции 290-350°C (240-290°C)'},
    'F34': {'min': 0.15, 'max': 0.25, 'current': 0.20, 'unit': 'доля', 'desc': 'Доля фракции 350-500°C (150-250°C)'},
    'F56': {'min': 0.05, 'max': 0.15, 'current': 0.10, 'unit': 'доля', 'desc': 'Доля лёгкого компонента в блендинге'},
    'F57': {'min': 0.05, 'max': 0.15, 'current': 0.10, 'unit': 'доля', 'desc': 'Доля тяжёлого компонента в блендинге'},
    'F59': {'min': 0.02, 'max': 0.08, 'current': 0.05, 'unit': 'доля', 'desc': 'Доля присадки в блендинге'},
}


def infer_unit_and_type(tag: str, description: str) -> Tuple[str, str]:
    """Определение физического типа параметра и единиц измерения по тегу и описанию."""
    desc_lower = description.lower() if description else ""
    first_char = tag[0].upper() if tag else ""

    # Единицы измерения
    if "температур" in desc_lower or " °с" in desc_lower or "°c" in desc_lower or first_char == 'T':
        param_type = "T"
        unit = "°C"
    elif "давлен" in desc_lower or "вакуум" in desc_lower or "перепад" in desc_lower or first_char == 'P':
        param_type = "P"
        if "мм рт.ст" in desc_lower:
            unit = "мм рт.ст."
        else:
            unit = "кгс/см2"
    elif "массов" in desc_lower or first_char == 'W':
        param_type = "W"
        unit = "т/ч"
    elif "расход" in desc_lower or "производительн" in desc_lower or first_char == 'F':
        param_type = "F"
        unit = "м3/ч" if "объемн" in desc_lower else "т/ч"
    elif "уровень" in desc_lower or first_char == 'L':
        param_type = "L"
        unit = "%"
    elif "плотност" in desc_lower or first_char == 'D':
        param_type = "D"
        unit = "кг/м3"
    elif "анализатор" in desc_lower or "сер" in desc_lower or first_char == 'Q':
        param_type = "Q"
        unit = "мг/кг" if "сер" in desc_lower else "-"
    else:
        param_type = first_char
        unit = "-"

    return param_type, unit


def parse_kip_sheet(ws) -> List[Dict[str, Any]]:
    """Парсинг листа КИП (колонки АВТ и 24-2000)."""
    rows = list(ws.iter_rows(values_only=True))
    tags = []

    # Первая строка — заголовок ('АВТ (описание)', 'АВТ', '24-2000 (описание)', '24-2000')
    for r in rows[1:]:
        # АВТ колонка
        avt_desc = r[0] if len(r) > 0 else None
        avt_raw = r[1] if len(r) > 1 else None

        if avt_raw and str(avt_raw).strip():
            raw_tag = str(avt_raw).strip()
            desc = str(avt_desc).strip() if avt_desc else ""
            canonical = f"{raw_tag}_avt" if raw_tag in COLLISION_TAGS else raw_tag
            p_type, unit = infer_unit_and_type(raw_tag, desc)

            tags.append({
                'tag': canonical,
                'raw_tag': raw_tag,
                'installation': 'AVT',
                'category': 'KIP',
                'description': desc,
                'param_type': p_type,
                'unit': unit,
            })

        # 24-2000 колонка
        hydro_desc = r[2] if len(r) > 2 else None
        hydro_raw = r[3] if len(r) > 3 else None

        if hydro_raw and str(hydro_raw).strip():
            raw_tag = str(hydro_raw).strip()
            desc = str(hydro_desc).strip() if hydro_desc else ""
            canonical = f"{raw_tag}_hydro" if raw_tag in COLLISION_TAGS else raw_tag
            p_type, unit = infer_unit_and_type(raw_tag, desc)

            tags.append({
                'tag': canonical,
                'raw_tag': raw_tag,
                'installation': '24-2000',
                'category': 'KIP',
                'description': desc,
                'param_type': p_type,
                'unit': unit,
            })

    return tags


def parse_pak_sheet(ws) -> List[Dict[str, Any]]:
    """Парсинг листа ПАК."""
    rows = list(ws.iter_rows(values_only=True))
    tags = []
    for r in rows[1:]:
        if len(r) >= 2 and r[1]:
            raw_tag = str(r[1]).strip()
            desc = str(r[0]).strip() if r[0] else ""
            p_type, unit = infer_unit_and_type(raw_tag, desc)
            tags.append({
                'tag': raw_tag,
                'raw_tag': raw_tag,
                'installation': '24-2000',
                'category': 'PAK',
                'description': desc,
                'param_type': p_type,
                'unit': unit,
            })
    return tags


def parse_vac_sheet(ws) -> List[Dict[str, Any]]:
    """Парсинг листа ВАК."""
    rows = list(ws.iter_rows(values_only=True))
    tags = []
    # Столбцы 0-1 (AVT6:240-350), 2-3 (AVT6:350), 4-5 (24-2000. ГО ДТ)
    sections = [
        (0, 1, 'AVT', 'AVT6:240-350'),
        (2, 3, 'AVT', 'AVT6:350'),
        (4, 5, '24-2000', '24-2000:GODT'),
    ]

    for r in rows[1:]:
        for tag_col, form_col, inst, sec_name in sections:
            if len(r) > form_col and r[tag_col] and r[form_col]:
                tag_name = str(r[tag_col]).strip()
                formula = str(r[form_col]).strip()
                p_type, unit = infer_unit_and_type(tag_name, tag_name)
                tags.append({
                    'tag': tag_name,
                    'raw_tag': tag_name,
                    'installation': inst,
                    'category': 'VAC',
                    'description': f"Виртуальный анализатор {sec_name}. Формула: {formula}",
                    'param_type': p_type,
                    'unit': unit,
                    'formula': formula,
                })
    return tags


def calculate_telemetry_stats(df_clean: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Вычисление технологических норм и статистик по очищенной телеметрии."""
    stats = {}
    num_cols = df_clean.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        s = df_clean[col].dropna()
        if len(s) == 0:
            continue
        mean_val = float(s.mean())
        std_val = float(s.std())
        if std_val <= 0 or np.isnan(std_val):
            std_val = 1e-6

        q005 = float(s.quantile(0.005))
        q01 = float(s.quantile(0.01))
        q05 = float(s.quantile(0.05))
        q50 = float(s.quantile(0.50))
        q95 = float(s.quantile(0.95))
        q99 = float(s.quantile(0.99))
        q995 = float(s.quantile(0.995))

        stats[col] = {
            'norm_mean': round(mean_val, 4),
            'norm_std': round(std_val, 4),
            'min_norm': round(q005, 4),
            'max_norm': round(q995, 4),
            'p01': round(q01, 4),
            'p05': round(q05, 4),
            'p50': round(q50, 4),
            'p95': round(q95, 4),
            'p99': round(q99, 4),
        }
    return stats


def build_tag_dictionary(
    excel_path: str,
    telemetry_path: Optional[str] = None
) -> pd.DataFrame:
    """Сборка полного справочника тегов."""
    wb = openpyxl.load_workbook(excel_path, data_only=True)

    all_records = []
    if 'КИП' in wb.sheetnames:
        all_records.extend(parse_kip_sheet(wb['КИП']))
    if 'ПАК' in wb.sheetnames:
        all_records.extend(parse_pak_sheet(wb['ПАК']))
    if 'ВАК' in wb.sheetnames:
        all_records.extend(parse_vac_sheet(wb['ВАК']))

    df_dict = pd.DataFrame(all_records)
    # Устраняем дубликаты тегов, если есть
    df_dict = df_dict.drop_duplicates(subset=['tag']).reset_index(drop=True)

    # Статистики из телеметрии
    telemetry_stats = {}
    if telemetry_path and os.path.exists(telemetry_path):
        df_clean = pd.read_parquet(telemetry_path)
        telemetry_stats = calculate_telemetry_stats(df_clean)

    # Обогащаем датафрейм статистиками и флагами управления
    for metric in ['norm_mean', 'norm_std', 'min_norm', 'max_norm', 'p01', 'p05', 'p50', 'p95', 'p99']:
        df_dict[metric] = df_dict['tag'].map(lambda t: telemetry_stats.get(t, {}).get(metric, np.nan))

    # Разметка управляемых параметров
    df_dict['is_controlled'] = df_dict['tag'].apply(lambda t: t in CONTROLLED_PARAMS_SPECS)
    df_dict['controlled_min'] = df_dict['tag'].map(lambda t: CONTROLLED_PARAMS_SPECS.get(t, {}).get('min', np.nan))
    df_dict['controlled_max'] = df_dict['tag'].map(lambda t: CONTROLLED_PARAMS_SPECS.get(t, {}).get('max', np.nan))
    df_dict['controlled_current'] = df_dict['tag'].map(lambda t: CONTROLLED_PARAMS_SPECS.get(t, {}).get('current', np.nan))

    # Если в телеметрии есть теги, которых не было в Excel (на всякий случай)
    if telemetry_stats:
        existing_tags = set(df_dict['tag'])
        missing_records = []
        for tag, s in telemetry_stats.items():
            if tag not in existing_tags:
                inst = 'AVT' if '_avt' in tag else ('24-2000' if '_hydro' in tag else 'COMMON')
                p_type, unit = infer_unit_and_type(tag, tag)
                missing_records.append({
                    'tag': tag,
                    'raw_tag': tag.replace('_avt', '').replace('_hydro', ''),
                    'installation': inst,
                    'category': 'KIP',
                    'description': f"Технологический параметр {tag}",
                    'param_type': p_type,
                    'unit': unit,
                    'norm_mean': s['norm_mean'],
                    'norm_std': s['norm_std'],
                    'min_norm': s['min_norm'],
                    'max_norm': s['max_norm'],
                    'p01': s['p01'],
                    'p05': s['p05'],
                    'p50': s['p50'],
                    'p95': s['p95'],
                    'p99': s['p99'],
                    'is_controlled': tag in CONTROLLED_PARAMS_SPECS,
                    'controlled_min': CONTROLLED_PARAMS_SPECS.get(tag, {}).get('min', np.nan),
                    'controlled_max': CONTROLLED_PARAMS_SPECS.get(tag, {}).get('max', np.nan),
                    'controlled_current': CONTROLLED_PARAMS_SPECS.get(tag, {}).get('current', np.nan),
                })
        if missing_records:
            df_dict = pd.concat([df_dict, pd.DataFrame(missing_records)], ignore_index=True)

    return df_dict


def dump_yaml_simple(data: Any, indent: int = 0) -> str:
    """Простой сериализатор структуры Python в стандартный YAML."""
    lines = []
    prefix = " " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                lines.append(dump_yaml_simple(v, indent + 2))
            elif v is None:
                lines.append(f"{prefix}{k}: null")
            elif isinstance(v, bool):
                lines.append(f"{prefix}{k}: {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{prefix}{k}: {v}")
            else:
                s_val = str(v).replace('"', '\\"')
                lines.append(f'{prefix}{k}: "{s_val}"')
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(dump_yaml_simple(item, indent + 2))
            elif isinstance(item, (int, float)):
                lines.append(f"{prefix}- {item}")
            elif isinstance(item, bool):
                lines.append(f"{prefix}- {'true' if item else 'false'}")
            else:
                lines.append(f'{prefix}- "{item}"')
    return "\n".join(lines)


def generate_constraints_yaml(output_path: str):
    """Генерация канонического файла constraints.yaml и constraints.json."""
    constraints = {
        'quality_hard_limits': {
            'sulfur_max_mg_kg': 10.0,
            'd15_min_kg_m3': 820.0,
            'd15_max_kg_m3': 845.0,
            'flash_point_min_c': 55.0,
            'cfpp_max_c': -5.0,
            't95_max_c': 360.0,
        },
        'blending_constraints': {
            'total_share': 1.0,
            'total_share_percent': 100.0,
            'tolerance_percent': 0.1,
            'components': ['F30', 'F32', 'F34', 'F56', 'F57', 'F59'],
        },
        'data_freshness_limits': {
            'pak_max_age_min': 60.0,
            'lims_max_age_hours': 48.0,
            'stale_threshold_min': 240.0,
            'response_lag_min': 30,
            'response_lag_max': 120,
        },
        'equipment_safety_limits': {
            'avt_total_feed_min': 50.0,
            'max_reactor_temperature_c': 380.0,
            'max_reactor_pressure_mpa': 4.5,
            'max_reactor_pressure_at': 45.0,
        },
        'controlled_parameters': CONTROLLED_PARAMS_SPECS,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    yaml_text = "# ==============================================================================\n" \
                "# ФАЙЛ КОНФИГУРАЦИИ: constraints.yaml\n" \
                "# НАЗНАЧЕНИЕ: Жёсткие технологические ограничения и контролируемые параметры\n" \
                "# ==============================================================================\n\n" + \
                dump_yaml_simple(constraints) + "\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_text)

    # Также сохраняем json-версию рядом для быстрого и безопасного парсинга
    json_path = output_path.replace('.yaml', '.json').replace('.yml', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(constraints, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Сборка справочника тегов и норм (DATA-06)")
    parser.add_argument("--excel", default="data/external/Теги_хакатон.xlsx", help="Путь к файлу тегов")
    parser.add_argument("--telemetry", default="data/processed/telemetry_clean.parquet", help="Путь к очищенной телеметрии")
    parser.add_argument("--output-dir", default="data/external", help="Папка для сохранения справочников")
    parser.add_argument("--constraints", default="data/external/constraints.yaml", help="Путь к constraints.yaml")

    args = parser.parse_args()

    print("=" * 70)
    print("DATA-06: Генерация справочника тегов и технологических норм")
    print("=" * 70)
    print(f"Входной Excel:     {args.excel}")
    print(f"Телеметрия:        {args.telemetry}")
    print(f"Выходная папка:    {args.output_dir}")

    df_dict = build_tag_dictionary(args.excel, args.telemetry)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "tag_dict.csv"
    parquet_path = out_dir / "tag_dict.parquet"

    df_dict.to_csv(csv_path, index=False, encoding='utf-8')
    df_dict.to_parquet(parquet_path, index=False)

    print(f"\n[OK] Справочник успешно сохранен:")
    print(f"  - CSV:     {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - Parquet: {parquet_path} ({parquet_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - Всего тегов: {len(df_dict)}")

    # Генерация constraints.yaml
    generate_constraints_yaml(args.constraints)
    # Также синхронизируем config/constraints.yaml
    generate_constraints_yaml("config/constraints.yaml")
    print(f"[OK] Конфигурация ограничений сохранена: {args.constraints} и config/constraints.yaml")

    # Сводка DoD
    print("\n--- DoD Метрики ---")
    print(f"Количество тегов КИП:        {len(df_dict[df_dict['category'] == 'KIP'])}")
    print(f"Количество тегов ПАК:        {len(df_dict[df_dict['category'] == 'PAK'])}")
    print(f"Количество тегов ВАК:        {len(df_dict[df_dict['category'] == 'VAC'])}")
    print(f"Управляемых параметров (MV): {df_dict['is_controlled'].sum()}")
    print(f"Тегов со статистикой норм:   {df_dict['norm_mean'].notna().sum()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
