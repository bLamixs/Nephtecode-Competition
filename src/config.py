"""
Модуль: src/config.py
Назначение: Централизованный доступ к справочнику тегов, технологическим нормам,
жёстким ограничениям качества и безопасным диапазонам управления.

Блок: Data (DATA-06)
Зависимости: Используется OptimizationAgent, QualityAgent, ReliabilityAgent,
Orchestrator и FeatureStore.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


# Глобальные пути по умолчанию
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAG_DICT_PARQUET = PROJECT_ROOT / "data" / "external" / "tag_dict.parquet"
DEFAULT_TAG_DICT_CSV = PROJECT_ROOT / "data" / "external" / "tag_dict.csv"
DEFAULT_CONSTRAINTS_JSON = PROJECT_ROOT / "config" / "constraints.json"
DEFAULT_CONSTRAINTS_YAML = PROJECT_ROOT / "config" / "constraints.yaml"

# Кэш в памяти
_TAG_DICT_CACHE: Optional[pd.DataFrame] = None
_CONSTRAINTS_CACHE: Optional[Dict[str, Any]] = None


def load_tag_dict(path: Optional[str] = None, reload: bool = False) -> pd.DataFrame:
    """
    Загрузка справочника тегов со статистическими нормами и метаданными.
    Кэширует результат в памяти для быстродействия агентов.
    """
    global _TAG_DICT_CACHE
    if _TAG_DICT_CACHE is not None and not reload:
        return _TAG_DICT_CACHE

    target_path = Path(path) if path else DEFAULT_TAG_DICT_PARQUET
    if not target_path.exists():
        # Fallback на CSV
        target_path = DEFAULT_TAG_DICT_CSV

    if not target_path.exists():
        raise FileNotFoundError(f"Справочник тегов не найден по пути: {target_path}")

    if str(target_path).endswith('.parquet'):
        df = pd.read_parquet(target_path)
    else:
        df = pd.read_csv(target_path)

    _TAG_DICT_CACHE = df
    return _TAG_DICT_CACHE


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Простой легковесный парсер YAML для словарей и списков без внешних зависимостей."""
    res = {}
    stack = [(res, -1)]

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.strip().startswith('#'):
            continue

        indent = len(line) - len(line.lstrip(' '))
        stripped = line.strip()

        while len(stack) > 1 and indent <= stack[-1][1]:
            stack.pop()

        current_dict, _ = stack[-1]

        if stripped.startswith('- '):
            # Элемент списка
            val_str = stripped[2:].strip()
            if isinstance(current_dict, list):
                val = _convert_scalar(val_str)
                current_dict.append(val)
            continue

        if ':' in stripped:
            parts = stripped.split(':', 1)
            key = parts[0].strip()
            val_str = parts[1].strip()

            if not val_str:
                # Начало вложенного блока
                # Смотрим следующую строку эвристически или создаем dict
                new_container = {}
                if isinstance(current_dict, dict):
                    current_dict[key] = new_container
                stack.append((new_container, indent))
            else:
                val = _convert_scalar(val_str)
                if isinstance(current_dict, dict):
                    current_dict[key] = val

    return res


def _convert_scalar(val_str: str) -> Any:
    """Преобразование строкового скаляра в число, булево или строку."""
    val_clean = val_str.strip('"').strip("'")
    if val_clean.lower() in ('true', 'yes'):
        return True
    if val_clean.lower() in ('false', 'no'):
        return False
    if val_clean.lower() in ('null', 'none', '~'):
        return None
    try:
        if '.' in val_clean or 'e' in val_clean.lower():
            return float(val_clean)
        return int(val_clean)
    except ValueError:
        return val_clean


def load_constraints(path: Optional[str] = None, reload: bool = False) -> Dict[str, Any]:
    """
    Загрузка технологических ограничений (hard constraints, fresh limits, safety limits).
    """
    global _CONSTRAINTS_CACHE
    if _CONSTRAINTS_CACHE is not None and not reload:
        return _CONSTRAINTS_CACHE

    target_path = Path(path) if path else DEFAULT_CONSTRAINTS_JSON
    if not target_path.exists():
        target_path = DEFAULT_CONSTRAINTS_YAML
        if not target_path.exists():
            target_path = PROJECT_ROOT / "data" / "external" / "constraints.json"

    if not target_path.exists():
        raise FileNotFoundError(f"Файл ограничений не найден: {target_path}")

    if str(target_path).endswith('.json'):
        with open(target_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(target_path, 'r', encoding='utf-8') as f:
            data = _parse_simple_yaml(f.read())

    _CONSTRAINTS_CACHE = data
    return _CONSTRAINTS_CACHE


def get_tag_meta(tag: str) -> Dict[str, Any]:
    """
    Получение подробной метаинформации о теге: описание, нормы, единицы измерения.
    """
    df = load_tag_dict()
    match = df[df['tag'] == tag]
    if match.empty:
        # Попробуем найти по raw_tag
        match = df[df['raw_tag'] == tag]

    if match.empty:
        return {
            'tag': tag,
            'description': f'Параметр {tag}',
            'param_type': tag[0].upper() if tag else 'X',
            'unit': '-',
            'norm_mean': np.nan,
            'norm_std': np.nan,
            'min_norm': np.nan,
            'max_norm': np.nan,
            'is_controlled': False,
        }

    return match.iloc[0].to_dict()


def get_controlled_params() -> Dict[str, Dict[str, Any]]:
    """
    Получение словаря параметров, доступных для оптимизации (Manipulated Variables),
    с их допустимыми диапазонами регулирования.
    """
    constraints = load_constraints()
    if 'controlled_parameters' in constraints:
        return constraints['controlled_parameters']

    # Fallback из tag_dict
    df = load_tag_dict()
    controlled_df = df[df['is_controlled'] == True]
    res = {}
    for _, row in controlled_df.iterrows():
        res[row['tag']] = {
            'min': row['controlled_min'],
            'max': row['controlled_max'],
            'current': row.get('controlled_current', row['norm_mean']),
            'unit': row['unit'],
            'desc': row['description'],
        }
    return res


def get_blending_components() -> List[str]:
    """Список тегов компонентов блендинга дизельного топлива."""
    constraints = load_constraints()
    blending_cfg = constraints.get('blending_constraints', {})
    if 'components' in blending_cfg:
        return blending_cfg['components']
    return ['F30', 'F32', 'F34', 'F56', 'F57', 'F59']


def get_hard_limits() -> Dict[str, Any]:
    """Жёсткие ограничения спецификации качества (Евро-5 / ГОСТ)."""
    constraints = load_constraints()
    return constraints.get('quality_hard_limits', {
        'sulfur_max_mg_kg': 10.0,
        'd15_min_kg_m3': 820.0,
        'd15_max_kg_m3': 845.0,
        'flash_point_min_c': 55.0,
        'cfpp_max_c': -5.0,
        't95_max_c': 360.0,
    })


def get_data_freshness_limits() -> Dict[str, Any]:
    """Ограничения на свежесть и актуальность замеров ЛИМС и ПАК."""
    constraints = load_constraints()
    return constraints.get('data_freshness_limits', {
        'pak_max_age_min': 60.0,
        'lims_max_age_hours': 48.0,
        'stale_threshold_min': 240.0,
        'response_lag_min': 30,
        'response_lag_max': 120,
    })


def get_vac_registry() -> Dict[str, Dict[str, Any]]:
    """Реестр формул виртуальных анализаторов (ВАК)."""
    df = load_tag_dict()
    vac_df = df[df['category'] == 'VAC']
    registry = {}
    for _, r in vac_df.iterrows():
        registry[r['tag']] = {
            'formula': r.get('formula', ''),
            'installation': r.get('installation', ''),
            'description': r.get('description', ''),
        }
    return registry


def is_value_safe(tag: str, value: float) -> Tuple[bool, Optional[str]]:
    """
    Проверка допустимости значения технологического параметра по жёстким и аварийным нормам.

    Возвращает:
        (is_safe, violation_reason)
    """
    if np.isnan(value):
        return False, f"Значение {tag} является NaN (пропуск данных)"

    hard_limits = get_hard_limits()

    # Проверка серы
    if 'sulfur' in tag.lower() or tag in ('W7', '24-2000:Mg.Sulfur.Q'):
        max_s = hard_limits.get('sulfur_max_mg_kg', 10.0)
        if value > max_s:
            return False, f"Содержание серы {value:.2f} превышает предел ГОСТ ({max_s} мг/кг)"

    # Проверка плотности
    if 'd15' in tag.lower() or tag == '24-2000:D15':
        min_d = hard_limits.get('d15_min_kg_m3', 820.0)
        max_d = hard_limits.get('d15_max_kg_m3', 845.0)
        if value < min_d or value > max_d:
            return False, f"Плотность D15 {value:.1f} выходит за ГОСТ [{min_d}..{max_d}] кг/м3"

    # Проверка температуры реактора гидроочистки
    if tag in ('T6_hydro', 'T23', 'T5'):
        safety = load_constraints().get('equipment_safety_limits', {})
        max_t = safety.get('max_reactor_temperature_c', 380.0)
        if value > max_t:
            return False, f"Температура реактора {tag}={value:.1f}°C выше безопасного предела {max_t}°C (риск закоксовывания)"

    # Проверка контролируемых параметров на допустимый диапазон
    controlled = get_controlled_params()
    if tag in controlled:
        p_min = controlled[tag]['min']
        p_max = controlled[tag]['max']
        if value < p_min or value > p_max:
            return False, f"Параметр {tag}={value:.2f} вне допустимого диапазона [{p_min}..{p_max}]"

    return True, None
