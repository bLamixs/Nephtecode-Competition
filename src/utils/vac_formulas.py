"""
Виртуальные анализаторы качества (ВАК).
Расчётные аналитические зависимости, связывающие технологические параметры КИП
и показатели качества продукта (ТЗ НефтеКод).
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class VirtualAnalyzers:
    """Библиотека формул ВАК для АВТ и установки гидроочистки 24-2000."""

    @staticmethod
    def calc_avt6_240_350_d15(row: Dict[str, float]) -> Optional[float]:
        denom = row.get('F32', 0.0) + row.get('F30', 0.0)
        if denom == 0:
            return None
        return 791.22872 - 5.30294 * (row.get('F30', 0.0) / denom) + 0.52755 * row.get('T66', 0.0) - 0.15629 * row.get('T33', 0.0)

    @staticmethod
    def calc_avt6_240_350_t50(row: Dict[str, float]) -> float:
        return (283.177 - 0.01685 * row.get('F7', 0.0) + 0.06248 * row.get('F30', 0.0) +
                0.22048 * row.get('F34', 0.0) - 0.25816 * row.get('F45', 0.0) -
                0.12159 * row.get('F59', 0.0) + 0.01221 * row.get('F63', 0.0))

    @staticmethod
    def calc_avt6_240_350_ebp(row: Dict[str, float]) -> float:
        return (813.883 + 2.66463 * row.get('F30', 0.0) - 0.20239 * row.get('T33', 0.0) -
                3.65888 * row.get('F36', 0.0) - 14.08235 * row.get('T37', 0.0) -
                1.32603 * row.get('T40', 0.0) + 14.60206 * row.get('T58', 0.0))

    @staticmethod
    def calc_avt6_240_350_cfpp(row: Dict[str, float]) -> Optional[float]:
        denom = row.get('F32', 0.0) + row.get('F30', 0.0)
        if denom == 0:
            return None
        return (31.40363 - 0.06784 * row.get('T33', 0.0) + 17.411 * row.get('P67', 0.0) -
                8.11544 * row.get('P4', 0.0) - 0.47309 * (row.get('F65', 0.0) / denom))

    @staticmethod
    def calc_avt6_350_t50(row: Dict[str, float]) -> Optional[float]:
        f57 = row.get('F57', 0.0)
        if f57 == 0:
            return None
        return 981.06539 + 0.27467 * row.get('T42', 0.0) - 0.32983 * (row.get('F31', 0.0) / f57) - 0.49014 * row.get('T48', 0.0)

    @staticmethod
    def calc_avt6_350_d15(row: Dict[str, float]) -> Optional[float]:
        f57 = row.get('F57', 0.0)
        if f57 == 0:
            return None
        return 983.092 + 0.27467 * row.get('T42', 0.0) - 0.49014 * row.get('T48', 0.0) - 0.32983 * (row.get('F31', 0.0) / f57)

    @staticmethod
    def calc_godt_t90(row: Dict[str, float]) -> Optional[float]:
        f26 = row.get('F26', 0.0)
        if f26 == 0:
            return None
        return (162.998 + 0.12945 * row.get('T12', 0.0) + 59.57 * row.get('F15', 0.0) +
                0.00036 * row.get('W7', 0.0) + 0.26366 * row.get('T23', 0.0) -
                424.72638 * (row.get('F1', 0.0) / f26))

    @staticmethod
    def calc_godt_t50(row: Dict[str, float]) -> float:
        return 44.625 + 10.0224 * row.get('P13', 0.0) + 0.06981 * row.get('F9', 0.0) + 0.8052 * row.get('T6', 0.0)

    @staticmethod
    def calc_godt_d15(row: Dict[str, float], d15_lims: float = 835.0) -> float:
        return 667.881 + 0.15417 * d15_lims + 0.00005 * row.get('F22', 0.0) + 0.10774 * row.get('T11', 0.0)

    @staticmethod
    def calc_godt_t95(row: Dict[str, float], t95_lims: float = 350.0) -> float:
        return (0.03814 * row.get('F9', 0.0) - 9.201 - 0.00002 * row.get('F2', 0.0) +
                0.62259 * row.get('T6', 0.0) + 0.48321 * t95_lims)

    @staticmethod
    def calc_godt_cfpp(row: Dict[str, float]) -> float:
        return (0.22088 * row.get('T6', 0.0) - 102.375 - 47.75834 * row.get('P8', 0.0) +
                0.03862 * row.get('F9', 0.0) + 43.60207 * row.get('W7', 0.0) + 43.81849 * row.get('P24', 0.0))

    @staticmethod
    def calc_godt_ibp(row: Dict[str, float]) -> float:
        return (137.762 - 0.0653 * row.get('F26', 0.0) + 0.00011 * row.get('F22', 0.0) +
                5.78137 * row.get('P13', 0.0) - 34.58028 * row.get('P24', 0.0) -
                0.00993 * row.get('F14', 0.0) - 0.99962 * row.get('W4', 0.0) +
                0.32232 * row.get('T23', 0.0) - 0.09406 * row.get('T16', 0.0))
