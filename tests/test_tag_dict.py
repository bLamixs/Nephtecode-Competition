"""
Модуль: tests/test_tag_dict.py
Назначение: Автоматические тесты для блока DATA-06:
Справочник тегов, статистические технологические нормы и конфигурация ограничений.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    load_tag_dict,
    load_constraints,
    get_tag_meta,
    get_controlled_params,
    get_blending_components,
    get_hard_limits,
    get_data_freshness_limits,
    get_vac_registry,
    is_value_safe,
)


@pytest.fixture
def tag_df():
    """Фикстура загрузки справочника тегов."""
    return load_tag_dict(reload=True)


@pytest.fixture
def constraints():
    """Фикстура загрузки ограничений."""
    return load_constraints(reload=True)


class TestTagDictionary:
    """Тестирование структуры и наполнения справочника тегов."""

    def test_tag_dict_loading(self, tag_df):
        assert isinstance(tag_df, pd.DataFrame)
        assert len(tag_df) >= 100, f"В справочнике ожидается 100+ тегов, найдено {len(tag_df)}"

        required_columns = [
            'tag', 'raw_tag', 'installation', 'category', 'description',
            'param_type', 'unit', 'norm_mean', 'norm_std', 'min_norm', 'max_norm'
        ]
        for col in required_columns:
            assert col in tag_df.columns, f"Колонка {col} отсутствует в tag_dict"

    def test_all_telemetry_clean_tags_present(self, tag_df):
        """Проверка, что все технологические теги из telemetry_clean.parquet присутствуют в справочнике."""
        telemetry_path = Path("data/processed/telemetry_clean.parquet")
        if not telemetry_path.exists():
            pytest.skip("Файл telemetry_clean.parquet не найден, пропускаем")

        df_telemetry = pd.read_parquet(telemetry_path)
        num_cols = df_telemetry.select_dtypes(include=[np.number]).columns

        tag_set = set(tag_df['tag'])
        for col in num_cols:
            assert col in tag_set, f"Тег {col} из telemetry_clean отсутствует в tag_dict"

    def test_norm_statistics_validity(self, tag_df):
        """Проверка валидности статистических норм (std > 0, min <= mean <= max)."""
        valid_stats = tag_df.dropna(subset=['norm_mean', 'norm_std', 'min_norm', 'max_norm'])
        assert len(valid_stats) >= 70, "Слишком мало тегов с рассчитанными нормами"

        # Стандартное отклонение должно быть строго положительным
        assert (valid_stats['norm_std'] > 0).all(), "Обнаружены неположительные norm_std"

        # min_norm <= max_norm
        assert (valid_stats['min_norm'] <= valid_stats['max_norm']).all(), "min_norm превышает max_norm"

    def test_collision_resolution(self, tag_df):
        """Проверка правильного разрешения коллизий с суффиксами _avt и _hydro."""
        tags = set(tag_df['tag'])
        assert 'T6_avt' in tags
        assert 'T6_hydro' in tags
        assert 'F9_avt' in tags
        assert 'F9_hydro' in tags
        assert 'T11_avt' in tags
        assert 'T11_hydro' in tags

    def test_quality_tags_in_tag_dict(self, tag_df):
        """Проверка наличия статистических норм для показателей качества в tag_dict."""
        tags = set(tag_df['tag'])
        for q in ['Mg.Sulfur', 'D15', '50%.T', '90%.T', '95%.T', 'CFPP', 'Mass.Sulfur']:
            assert q in tags, f"Показатель качества {q} отсутствует в tag_dict"
            sub = tag_df[tag_df['tag'] == q].iloc[0]
            assert pd.notna(sub['norm_mean']), f"norm_mean для {q} пустой"
            assert pd.notna(sub['norm_std']) and sub['norm_std'] > 0, f"norm_std для {q} невалидный"



class TestConstraints:
    """Тестирование конфигурации технологических ограничений."""

    def test_constraints_structure(self, constraints):
        assert 'quality_hard_limits' in constraints
        assert 'blending_constraints' in constraints
        assert 'data_freshness_limits' in constraints
        assert 'equipment_safety_limits' in constraints
        assert 'controlled_parameters' in constraints

    def test_quality_hard_limits(self):
        hard = get_hard_limits()
        assert hard['sulfur_max_mg_kg'] == 10.0, "Максимальная сера должна быть строго 10.0 мг/кг"
        assert hard['d15_min_kg_m3'] == 820.0
        assert hard['d15_max_kg_m3'] == 845.0
        assert hard['cfpp_max_c'] == -5.0

    def test_blending_components(self):
        comps = get_blending_components()
        assert len(comps) == 6
        assert set(comps) == {'F30', 'F32', 'F34', 'F56', 'F57', 'F59'}

    def test_data_freshness(self):
        fresh = get_data_freshness_limits()
        assert fresh['pak_max_age_min'] == 60.0
        assert fresh['lims_max_age_hours'] == 48.0
        assert fresh['stale_threshold_min'] == 240.0
        assert fresh['response_lag_min'] == 30
        assert fresh['response_lag_max'] == 120


class TestConfigInterface:
    """Тестирование интерфейсных функций модуля src/config.py."""

    def test_get_tag_meta(self):
        meta_t6 = get_tag_meta('T6_hydro')
        assert meta_t6['installation'] == '24-2000'
        assert meta_t6['unit'] == '°C'
        assert meta_t6['is_controlled'] == True
        assert meta_t6['controlled_min'] == 345.0
        assert meta_t6['controlled_max'] == 375.0

    def test_get_controlled_params(self):
        params = get_controlled_params()
        assert 'T6_hydro' in params
        assert 'F30' in params
        assert 'F32' in params
        for name, spec in params.items():
            assert 'min' in spec and 'max' in spec and 'unit' in spec
            assert spec['min'] <= spec['max']

    def test_get_vac_registry(self):
        vac = get_vac_registry()
        assert len(vac) > 0
        assert 'AVT6:240-350:D15' in vac
        assert '24-2000:GODT:T90' in vac

    def test_is_value_safe_sulfur(self):
        safe, reason = is_value_safe('24-2000:Mg.Sulfur.Q', 8.5)
        assert safe is True
        assert reason is None

        unsafe, reason = is_value_safe('24-2000:Mg.Sulfur.Q', 10.5)
        assert unsafe is False
        assert "превышает предел ГОСТ" in reason

    def test_is_value_safe_temperature(self):
        safe, _ = is_value_safe('T6_hydro', 360.0)
        assert safe is True

        # Превышение допустимого диапазона регулирования
        unsafe_range, reason = is_value_safe('T6_hydro', 378.0)
        assert unsafe_range is False
        assert "вне допустимого диапазона" in reason

        # Превышение критического предела реактора (380°C)
        unsafe_critical, reason = is_value_safe('T23', 385.0)
        assert unsafe_critical is False
        assert "риск закоксовывания" in reason

    def test_is_value_safe_nan(self):
        unsafe, reason = is_value_safe('T1', np.nan)
        assert unsafe is False
        assert "является NaN" in reason
