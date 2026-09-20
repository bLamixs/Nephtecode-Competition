"""
Модуль: tests/test_vac_formulas.py
Назначение: Автоматические тесты формул Виртуальных Анализаторов Качества (ВАК)
для блока Agent (AGENT-01).
"""

import pytest
import numpy as np
import pandas as pd
from src.utils.vac_formulas import VirtualAnalyzers


@pytest.fixture
def sample_telemetry_row():
    """Тестовая строка телеметрии с реалистичными значениями датчиков."""
    return {
        'F30': 30.0,
        'F32': 25.0,
        'F34': 20.0,
        'F56': 10.0,
        'F57': 10.0,
        'F59': 5.0,
        'T66': 220.0,
        'T33': 180.0,
        'F7': 120.0,
        'F45': 50.0,
        'F63': 30.0,
        'F36': 25.0,
        'T37': 160.0,
        'T40': 150.0,
        'T58': 240.0,
        'P67': 3.5,
        'P4': 3.8,
        'F65': 20.0,
        'T42': 320.0,
        'F31': 80.0,
        'T48': 290.0,
        'L43': 45.0,
        'T6': 295.0,
        'T18': 140.0,
        'F64': 90.0,
        'T15': 160.0,
        'T11': 345.0,
        'T12': 180.0,
        'F15': 4.5,
        'W7': 0.2,
        'T23': 355.0,
        'F1': 3.5,
        'F26': 200.0,
        'P13': 3.2,
        'F9': 250.0,
        'T5': 350.0,
        'F25': 2.0,
        'F14': 3.0,
        'T16': 170.0,
        'F22': 5.0,
        'P8': 32.0,
        'P24': 28.0,
        'W4': 2.5,
        'T13': 210.0,
        'T20': 230.0,
        'P50': 2.1,
        'F53': 15.0,
        'P51': 1.8,
        'T61': 260.0,
        'F2': 170.0,
    }


class TestVacAvtFormulas:
    """Тестирование формул ВАК для установки АВТ-6."""

    def test_calc_avt6_240_350_d15(self, sample_telemetry_row):
        d15 = VirtualAnalyzers.calc_avt6_240_350_d15(sample_telemetry_row)
        assert d15 is not None
        assert 800.0 <= d15 <= 920.0, f"Плотность D15 {d15} вне физического диапазона"

    def test_calc_avt6_240_350_d15_zero_division(self, sample_telemetry_row):
        row_zero = sample_telemetry_row.copy()
        row_zero['F30'] = 0.0
        row_zero['F32'] = 0.0
        assert VirtualAnalyzers.calc_avt6_240_350_d15(row_zero) is None

    def test_calc_avt6_240_350_t50(self, sample_telemetry_row):
        t50 = VirtualAnalyzers.calc_avt6_240_350_t50(sample_telemetry_row)
        assert 200.0 <= t50 <= 350.0, f"Температура T50 {t50} вне диапазона"

    def test_calc_avt6_240_350_ebp(self, sample_telemetry_row):
        ebp = VirtualAnalyzers.calc_avt6_240_350_ebp(sample_telemetry_row)
        assert isinstance(ebp, float)
        assert not np.isnan(ebp)

    def test_calc_avt6_240_350_cfpp(self, sample_telemetry_row):
        cfpp = VirtualAnalyzers.calc_avt6_240_350_cfpp(sample_telemetry_row)
        assert cfpp is not None
        assert -50.0 <= cfpp <= 50.0

    def test_calc_avt6_240_350_cfpp_zero_division(self, sample_telemetry_row):
        row_zero = sample_telemetry_row.copy()
        row_zero['F30'] = 0.0
        row_zero['F32'] = 0.0
        assert VirtualAnalyzers.calc_avt6_240_350_cfpp(row_zero) is None

    def test_calc_avt6_350_formulas(self, sample_telemetry_row):
        t50 = VirtualAnalyzers.calc_avt6_350_t50(sample_telemetry_row)
        assert t50 is not None
        assert 350.0 <= t50 <= 1200.0

        i350 = VirtualAnalyzers.calc_avt6_350_i350(sample_telemetry_row)
        assert isinstance(i350, float)

        d15 = VirtualAnalyzers.calc_avt6_350_d15(sample_telemetry_row)
        assert d15 is not None

        visc = VirtualAnalyzers.calc_avt6_350_500_viscosity(sample_telemetry_row)
        assert isinstance(visc, float)

        cfpp = VirtualAnalyzers.calc_avt6_350_cfpp(sample_telemetry_row)
        assert cfpp is not None

    def test_calc_avt6_350_zero_division(self, sample_telemetry_row):
        row_zero = sample_telemetry_row.copy()
        row_zero['F57'] = 0.0
        assert VirtualAnalyzers.calc_avt6_350_t50(row_zero) is None
        assert VirtualAnalyzers.calc_avt6_350_d15(row_zero) is None
        assert VirtualAnalyzers.calc_avt6_350_cfpp(row_zero) is None


class TestVacHydroFormulas:
    """Тестирование формул ВАК для установки гидроочистки 24-2000."""

    def test_calc_godt_t90(self, sample_telemetry_row):
        t90 = VirtualAnalyzers.calc_godt_t90(sample_telemetry_row)
        assert t90 is not None
        assert isinstance(t90, float)
        assert not np.isnan(t90)

    def test_calc_godt_t90_zero_division(self, sample_telemetry_row):
        row_zero = sample_telemetry_row.copy()
        row_zero['F26'] = 0.0
        assert VirtualAnalyzers.calc_godt_t90(row_zero) is None

    def test_calc_godt_t50(self, sample_telemetry_row):
        t50 = VirtualAnalyzers.calc_godt_t50(sample_telemetry_row)
        assert 250.0 <= t50 <= 350.0

    def test_calc_godt_i250(self, sample_telemetry_row):
        i250 = VirtualAnalyzers.calc_godt_i250(sample_telemetry_row)
        assert isinstance(i250, float)

    def test_calc_godt_d15(self, sample_telemetry_row):
        d15 = VirtualAnalyzers.calc_godt_d15(sample_telemetry_row, d15_lims=835.0)
        assert 800.0 <= d15 <= 880.0

    def test_calc_godt_cloud_point(self, sample_telemetry_row):
        cp = VirtualAnalyzers.calc_godt_cloud_point(sample_telemetry_row)
        assert isinstance(cp, float)

    def test_calc_godt_t95(self, sample_telemetry_row):
        t95 = VirtualAnalyzers.calc_godt_t95(sample_telemetry_row, t95_lims=350.0)
        assert 300.0 <= t95 <= 400.0

    def test_calc_godt_cfpp(self, sample_telemetry_row):
        cfpp = VirtualAnalyzers.calc_godt_cfpp(sample_telemetry_row)
        assert isinstance(cfpp, float)

    def test_calc_godt_ibp(self, sample_telemetry_row):
        ibp = VirtualAnalyzers.calc_godt_ibp(sample_telemetry_row)
        assert isinstance(ibp, float)


class TestVacVectorizedFormulas:
    """Тестирование векторных функций ВАК и функции calculate_all_vac (AGENT-01)."""

    @pytest.fixture
    def sample_telemetry_df(self, sample_telemetry_row):
        """Тестовый DataFrame телеметрии из нескольких строк с временным индексом."""
        import pandas as pd
        dates = pd.date_range('2026-01-01', periods=5, freq='h')
        rows = [sample_telemetry_row.copy() for _ in range(5)]
        # Внесем вариацию во вторую строку
        rows[1]['F30'] = 35.0
        rows[1]['T6'] = 300.0
        # Нулевые значения в третьей строке для проверки деления на ноль
        rows[2]['F30'] = 0.0
        rows[2]['F32'] = 0.0
        rows[2]['F26'] = 0.0

        df = pd.DataFrame(rows)
        df['date'] = dates
        return df

    def test_vac_d15_240_350_vectorized(self, sample_telemetry_df):
        from src.utils.vac_formulas import vac_d15_240_350
        res = vac_d15_240_350(sample_telemetry_df)
        assert len(res) == len(sample_telemetry_df)
        assert not res.isna().any(), "Векторный расчёт не должен содержать NaN при делении на 0"
        assert res.iloc[0] > 700.0

    def test_vac_t50_240_350_vectorized(self, sample_telemetry_df):
        from src.utils.vac_formulas import vac_t50_240_350
        res = vac_t50_240_350(sample_telemetry_df)
        assert len(res) == len(sample_telemetry_df)
        assert 200.0 <= res.iloc[0] <= 350.0

    def test_vac_ebp_240_350_vectorized(self, sample_telemetry_df):
        from src.utils.vac_formulas import vac_ebp_240_350
        res = vac_ebp_240_350(sample_telemetry_df)
        assert len(res) == len(sample_telemetry_df)
        assert not res.isna().any()

    def test_vac_cfpp_240_350_vectorized(self, sample_telemetry_df):
        from src.utils.vac_formulas import vac_cfpp_240_350
        res = vac_cfpp_240_350(sample_telemetry_df)
        assert len(res) == len(sample_telemetry_df)
        assert not res.isna().any()

    def test_vac_sulfur_24_2000_vectorized(self, sample_telemetry_df):
        from src.utils.vac_formulas import vac_sulfur_24_2000
        res = vac_sulfur_24_2000(sample_telemetry_df)
        assert len(res) == len(sample_telemetry_df)
        assert (res >= 0.0).all(), "Сера не может быть отрицательной"

    def test_calculate_all_vac_contract_and_dod(self, sample_telemetry_df):
        from src.utils.vac_formulas import calculate_all_vac
        res_df = calculate_all_vac(sample_telemetry_df)
        
        # DoD проверки
        assert isinstance(res_df, pd.DataFrame)
        assert list(res_df.columns) == ['date', 'tag', 'vac_value'], "Колонки должны быть строго date, tag, vac_value"
        assert len(res_df) == len(sample_telemetry_df) * 10, "Должно быть 10 ВАК на каждую строку"
        
        # Проверка обязательных тегов (минимум 5: D15, T50, EBP, CFPP, Sulfur)
        tags = set(res_df['tag'].unique())
        required_tags = {
            'AVT6:240-350:D15',
            'AVT6:240-350:T50',
            'AVT6:240-350:EBP',
            'AVT6:240-350:CFPP',
            '24-2000:GODT:Sulfur'
        }
        assert required_tags.issubset(tags), f"Отсутствуют обязательные теги: {required_tags - tags}"
        assert not res_df['vac_value'].isna().any(), "Значения ВАК не должны быть NaN"

