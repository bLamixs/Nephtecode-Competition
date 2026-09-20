"""
Модуль: tests/test_feature_store.py
Назначение: Автоматические тесты для блока DATA-05: Feature Store (Витрина признаков).
"""

import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from src.etl.feature_store import (
    generate_lag_features,
    generate_rolling_features,
    generate_zscore_features,
    build_feature_store,
)


@pytest.fixture
def synthetic_long_df():
    """Синтетический Long-DataFrame для тестирования математики признаков."""
    dates = pd.date_range('2023-01-01', periods=100, freq='10min', tz='UTC')
    data1 = pd.DataFrame({
        'date': dates,
        'tag': 'TAG_A',
        'value': np.arange(100, dtype=float),
    })
    data2 = pd.DataFrame({
        'date': dates,
        'tag': 'TAG_B',
        'value': np.full(100, 50.0, dtype=float),  # Константный сигнал
    })
    return pd.concat([data1, data2], ignore_index=True)


class TestFeatureStoreMath:
    """Тестирование корректности математических преобразований."""

    def test_lag_features_math(self, synthetic_long_df):
        df_lags = generate_lag_features(
            synthetic_long_df,
            tags=['TAG_A'],
            lags=[10, 30, 60],
            value_col='value',
            step_min=10,
        )

        assert 'lag10' in df_lags.columns
        assert 'lag30' in df_lags.columns
        assert 'lag60' in df_lags.columns

        # Для TAG_A: значения 0, 1, 2, ...
        # lag10 (1 шаг назад): на позиции 1 должно быть значение 0
        assert pd.isna(df_lags.loc[0, 'lag10'])
        assert df_lags.loc[1, 'lag10'] == 0.0
        assert df_lags.loc[2, 'lag10'] == 1.0

        # lag30 (3 шага назад): на позиции 3 должно быть значение 0
        assert pd.isna(df_lags.loc[2, 'lag30'])
        assert df_lags.loc[3, 'lag30'] == 0.0

        # lag60 (6 шагов назад): на позиции 6 должно быть значение 0
        assert pd.isna(df_lags.loc[5, 'lag60'])
        assert df_lags.loc[6, 'lag60'] == 0.0

    def test_rolling_features_math(self, synthetic_long_df):
        df_roll = generate_rolling_features(
            synthetic_long_df,
            tags=['TAG_A', 'TAG_B'],
            window=60,
            value_col='value',
        )

        assert 'rolling_mean_60' in df_roll.columns
        assert 'rolling_std_60' in df_roll.columns

        # Для константного сигнала TAG_B rolling_mean должен быть равен 50.0
        sub_b = df_roll[df_roll['tag'] == 'TAG_B']
        assert np.isclose(sub_b['rolling_mean_60'], 50.0).all()
        # Защита от деления на ноль: rolling_std должен быть неотрицательным
        assert (sub_b['rolling_std_60'] >= 0.0).all()

    def test_zscore_features_math(self, synthetic_long_df):
        # Передаем фиксированные нормы: mean = 50.0, std = 10.0
        df_z = generate_zscore_features(
            synthetic_long_df,
            tags=['TAG_A'],
            norm_mean=50.0,
            norm_std=10.0,
            value_col='value',
        )

        assert 'zscore' in df_z.columns
        sub_a = df_z[df_z['tag'] == 'TAG_A'].reset_index(drop=True)
        # При value = 50.0 -> zscore = 0.0
        assert np.isclose(sub_a.loc[50, 'zscore'], 0.0)
        # При value = 70.0 -> zscore = (70 - 50) / 10 = 2.0
        assert np.isclose(sub_a.loc[70, 'zscore'], 2.0)
        # При value = 30.0 -> zscore = (30 - 50) / 10 = -2.0
        assert np.isclose(sub_a.loc[30, 'zscore'], -2.0)


class TestFeatureStoreArtifactDoD:
    """Тестирование выходного артефакта data/processed/features.parquet по критериям DoD."""

    def test_features_parquet_exists_and_valid(self):
        feat_path = Path("data/processed/features.parquet")
        if not feat_path.exists():
            pytest.skip("Файл features.parquet не найден, пропускаем")

        df = pd.read_parquet(feat_path)
        assert len(df) > 100_000, f"Ожидалось > 100k строк, получено {len(df)}"

        # Проверка наличия колонок DoD
        required_cols = [
            'date', 'tag', 'value', 'lag10', 'lag30', 'lag60',
            'rolling_mean_60', 'rolling_std_60', 'zscore'
        ]
        for col in required_cols:
            assert col in df.columns, f"Колонка {col} отсутствует в features.parquet"

        # Проверка, что нет бесконечностей (+/- inf)
        for col in ['lag10', 'rolling_mean_60', 'zscore']:
            vals = df[col].dropna()
            assert not np.isinf(vals).any(), f"В колонке {col} обнаружены бесконечные значения"


class TestFeatureStorePipeline:
    """Тестирование конвейера сборки и интеграции с источниками данных."""

    def test_build_feature_store_end_to_end(self, synthetic_long_df):
        dummy_dict = pd.DataFrame([
            {'tag': 'TAG_A', 'norm_mean': 50.0, 'norm_std': 10.0},
            {'tag': 'TAG_B', 'norm_mean': 50.0, 'norm_std': 1.0},
        ])
        df_out = build_feature_store(
            df=synthetic_long_df,
            tag_dict_df=dummy_dict,
            tags=['TAG_A', 'TAG_B'],
            lags=[10, 30, 60],
            window=60,
        )

        assert 'lag10' in df_out.columns
        assert 'lag30' in df_out.columns
        assert 'lag60' in df_out.columns
        assert 'rolling_mean_60' in df_out.columns
        assert 'rolling_std_60' in df_out.columns
        assert 'zscore' in df_out.columns

        # Проверка префиксных колонок
        assert 'tag_lag10' in df_out.columns
        assert 'tag_zscore' in df_out.columns

    def test_zscore_with_real_tag_dict(self):
        """Проверка расчёта Z-score по реальному tag_dict.csv (DATA-06)."""
        td_path = Path("data/external/tag_dict.csv")
        if not td_path.exists():
            pytest.skip("Файл tag_dict.csv не найден")

        tag_dict = pd.read_csv(td_path)
        # Тестируем на теге гидроочистки T6_hydro и показателе качества Mg.Sulfur
        test_df = pd.DataFrame([
            {'date': pd.Timestamp('2023-01-01 00:00:00', tz='UTC'), 'tag': 'T6_hydro', 'value': 348.1418},
            {'date': pd.Timestamp('2023-01-01 00:00:00', tz='UTC'), 'tag': 'Mg.Sulfur', 'value': 8.44},
        ])

        df_z = generate_zscore_features(test_df, tag_dict_df=tag_dict)
        # При значении равном norm_mean, zscore должен быть равен 0.0
        z_t6 = df_z[df_z['tag'] == 'T6_hydro']['zscore'].values[0]
        z_s = df_z[df_z['tag'] == 'Mg.Sulfur']['zscore'].values[0]

        assert abs(z_t6) < 0.01, f"Ожидался zscore ~ 0 для T6_hydro, получено {z_t6}"
        assert abs(z_s) < 0.01, f"Ожидался zscore ~ 0 для Mg.Sulfur, получено {z_s}"

    def test_wide_format_processing(self):
        """Проверка работы с широкоформатным представлением (Wide-format)."""
        dates = pd.date_range('2023-01-01', periods=50, freq='10min', tz='UTC')
        df_wide = pd.DataFrame({
            'date': dates,
            'T1': np.arange(50, dtype=float),
            'P2': np.full(50, 10.0, dtype=float),
        })

        df_lags = generate_lag_features(df_wide, tags=['T1', 'P2'], lags=[10, 30])
        assert 'T1_lag10' in df_lags.columns
        assert 'P2_lag30' in df_lags.columns

        df_roll = generate_rolling_features(df_wide, tags=['T1'], window=10)
        assert 'T1_rolling_mean_10' in df_roll.columns
        assert 'T1_rolling_std_10' in df_roll.columns

    def test_nan_handling(self):
        """Проверка устойчивости к пропускам (NaN) в исходных данных."""
        dates = pd.date_range('2023-01-01', periods=20, freq='10min', tz='UTC')
        vals = [np.nan if i % 3 == 0 else float(i) for i in range(20)]
        df_nan = pd.DataFrame({'date': dates, 'tag': 'TAG_NAN', 'value': vals})

        df_out = build_feature_store(df_nan, tags=['TAG_NAN'], lags=[10])
        # Присутствуют NaN, но нет критических исключений и inf
        assert df_out['zscore'].isna().sum() > 0
        assert not np.isinf(df_out['zscore'].dropna()).any()

    def test_save_and_load_parquet(self, tmp_path):
        """Проверка сохранения и загрузки витрины признаков."""
        from src.etl.feature_store import save_features_parquet

        sample_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='10min', tz='UTC'),
            'tag': ['T1'] * 5,
            'value': [1.0, 2.0, 3.0, 4.0, 5.0],
            'lag10': [np.nan, 1.0, 2.0, 3.0, 4.0],
            'zscore': [0.1, 0.2, 0.3, 0.4, 0.5],
        })

        out_file = tmp_path / "test_features.parquet"
        save_features_parquet(sample_df, out_file)
        assert out_file.exists()

        loaded_df = pd.read_parquet(out_file)
        assert len(loaded_df) == 5
        assert 'lag10' in loaded_df.columns

