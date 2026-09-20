"""
Модуль: src/agents/reliability_agent.py
Назначение: Reliability Agent (Агент технологической надёжности, AGENT-05).

Контекст:
Оценивает тяжесть режима работы оборудования (реактор Р-201, печи П-101/102, колонны)
и риск дезактивации катализатора / закоксования.
Поскольку разметка прямых аварий отсутствует, используются статистические прокси-метрики:
1. Z-score по каждому технологическому тегу относительно эталонного режима.
2. Расстояние Махаланобиса (Mahalanobis distance) с регуляризованной/псевдообратной ковариацией.
3. Интегральный risk_index в диапазоне [0..1] и категоризация risk_class (low, medium, high).
4. Топ-3 фактора риска (теги с максимальным отклонением).
5. Коридоры технологических ограничений (constraints_for_optimizer) для Optimization Agent.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from src.utils.logging_config import setup_logger
from src.agents.interfaces import ReliabilityAssessment

logger = setup_logger('reliability_agent')


class ReliabilityAgent:
    """
    Агент оценки технологической надёжности (AGENT-05).
    """

    KEY_TAGS = [
        'T6', 'T11', 'T55', 'T1', 'T5', 'T23',
        'P8', 'P24', 'P2', 'P3', 'P4',
        'F26', 'F7', 'F8', 'F9', 'F1', 'F2', 'F3', 'F5',
        'W7', 'W10', 'W4'
    ]

    def __init__(
        self,
        norm_mean: Optional[Dict[str, float]] = None,
        norm_std: Optional[Dict[str, float]] = None,
        tag_dict_path: str = 'data/external/tag_dict.csv'
    ):
        self.norm_mean = norm_mean or {}
        self.norm_std = norm_std or {}
        self.controlled_bounds = {}
        self.tag_dict_path = tag_dict_path
        self._cov_pinv: Optional[np.ndarray] = None
        self._cov_mean_vec: Optional[np.ndarray] = None
        self._cov_tags: List[str] = []

        self._load_tag_dict()

    def _load_tag_dict(self):
        """Загрузка средних, стандартных отклонений и границ из справочника тегов."""
        path = Path(self.tag_dict_path)
        if not path.exists():
            # Попытка найти в родительских директориях
            alt_path = Path(__file__).resolve().parent.parent.parent / 'data/external/tag_dict.csv'
            if alt_path.exists():
                path = alt_path

        if path.exists():
            try:
                td_df = pd.read_csv(path)
                for _, row in td_df.iterrows():
                    tag = str(row['tag']).strip()
                    raw_tag = str(row.get('raw_tag', tag)).strip()

                    # Сохраняем среднее и дисперсию
                    if pd.notna(row.get('norm_mean')) and tag not in self.norm_mean:
                        self.norm_mean[tag] = float(row['norm_mean'])
                    if pd.notna(row.get('norm_std')) and tag not in self.norm_std:
                        self.norm_std[tag] = max(1e-4, float(row['norm_std']))

                    # Также привязываем к raw_tag если отличается (приоритет отдается контролируемым параметрам 24-2000)
                    if raw_tag and raw_tag != tag:
                        is_ctrl = row.get('is_controlled') is True or str(row.get('is_controlled')).lower() == 'true'
                        if (raw_tag not in self.norm_mean) or is_ctrl:
                            if tag in self.norm_mean:
                                self.norm_mean[raw_tag] = self.norm_mean[tag]
                            if tag in self.norm_std:
                                self.norm_std[raw_tag] = self.norm_std[tag]

                    # Границы для оптимизатора
                    c_min = row.get('controlled_min')
                    c_max = row.get('controlled_max')
                    if pd.notna(c_min) and pd.notna(c_max):
                        self.controlled_bounds[tag] = {
                            'min': float(c_min),
                            'max': float(c_max),
                            'desc': str(row.get('description', tag))
                        }
                logger.info(f"Справочник тегов загружен: {len(self.norm_mean)} тегов.")
            except Exception as e:
                logger.warning(f"Ошибка загрузки справочника тегов: {e}")

        # Дефолтные границы для ключевых регуляторов гидроочистки и АВТ
        defaults = {
            'T6': {'min': 345.0, 'max': 375.0, 'desc': 'Температура в реакторе Р-201'},
            'T6_hydro': {'min': 345.0, 'max': 375.0, 'desc': 'Температура в реакторе Р-201'},
            'T55': {'min': 375.0, 'max': 386.0, 'desc': 'Температура на выходе из печи П-3'},
            'F7': {'min': 200.0, 'max': 310.0, 'desc': 'Расход сырья 3-й ход'},
            'F8': {'min': 190.0, 'max': 325.0, 'desc': 'Расход сырья 1-й ход'},
            'F9': {'min': 160.0, 'max': 280.0, 'desc': 'Общий расход сырья'},
            'F9_avt': {'min': 160.0, 'max': 280.0, 'desc': 'Общий расход сырья на гидроочистку'},
            'P8': {'min': 0.10, 'max': 0.26, 'desc': 'Давление реактора Р-202'},
            'P24': {'min': 0.55, 'max': 0.63, 'desc': 'Давление циркулирующего ВСГ'}
        }
        for k, v in defaults.items():
            if k not in self.controlled_bounds:
                self.controlled_bounds[k] = v

    def _find_column(self, telemetry: pd.DataFrame, tag: str) -> Optional[str]:
        """Поиск столбца в телеметрии по тегу (с учетом суффиксов _hydro, _avt)."""
        if tag in telemetry.columns:
            return tag
        for col in telemetry.columns:
            if col.startswith(f"{tag}_") or col.endswith(f"_{tag}"):
                return col
        return None

    def _calculate_zscore(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """
        Расчёт Z-score для всех параметров:
        z_score = (value - norm_mean) / norm_std
        """
        zscore_df = pd.DataFrame(index=telemetry.index)

        # Выбираем числовые колонки
        numeric_cols = telemetry.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['date', 'timestamp', 'horizon_min', 'is_lims']:
                continue

            # Ищем базовый тег без суффиксов установки
            if 'ratio' in col:
                mean_val = None
                std_val = None
            else:
                base_tag = col.split('_')[0]
                mean_val = self.norm_mean.get(col, self.norm_mean.get(base_tag))
                std_val = self.norm_std.get(col, self.norm_std.get(base_tag))

            if mean_val is None or std_val is None:
                # Если в справочнике нет, используем статистику самой серии
                s = telemetry[col].dropna()
                if len(s) > 1:
                    mean_val = float(s.mean())
                    std_val = float(s.std()) if s.std() > 1e-4 else 1.0
                else:
                    mean_val = float(s.iloc[0]) if len(s) > 0 else 0.0
                    std_val = 1.0

            std_val = max(1e-4, std_val)
            zscore_df[col] = (telemetry[col] - mean_val) / std_val

        return zscore_df.fillna(0.0)

    def _fit_mahalanobis_covariance(self, telemetry: pd.DataFrame):
        """
        Построение ковариационной матрицы и псевдообратной матрицы (pinv)
        для расчёта расстояния Махаланобиса.
        """
        # Фильтрация строк тренировочного периода (2023-2025), если есть колонка date
        df_train = telemetry
        if 'date' in telemetry.columns:
            dates = pd.to_datetime(telemetry['date'], errors='coerce')
            train_mask = (dates >= '2023-01-01') & (dates <= '2025-12-31')
            if train_mask.sum() > 50:
                df_train = telemetry[train_mask]

        # Подбираем ключевые доступные теги
        available_tags = []
        for tag in self.KEY_TAGS:
            col = self._find_column(df_train, tag)
            if col and col not in available_tags:
                available_tags.append(col)

        if len(available_tags) < 2:
            # Fallback: берем любые первые числовые колонки
            num_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns if c not in ['date']]
            available_tags = num_cols[:15]

        self._cov_tags = available_tags
        sub_df = df_train[self._cov_tags].dropna()

        if len(sub_df) > len(available_tags):
            mean_vec = sub_df.mean().values
            cov_mat = np.cov(sub_df.values, rowvar=False)
            # Добавляем регуляризацию по главной диагонали (Ridge regularization) для устойчивости
            reg = 1e-3 * np.eye(cov_mat.shape[0])
            pinv = np.linalg.pinv(cov_mat + reg)
            self._cov_mean_vec = mean_vec
            self._cov_pinv = pinv
        else:
            dim = len(self._cov_tags)
            self._cov_mean_vec = np.zeros(dim)
            self._cov_pinv = np.eye(dim)

    def _calculate_mahalanobis(self, telemetry: pd.DataFrame) -> pd.Series:
        """
        Расчёт расстояния Махаланобиса от "нормального" режима.
        Использует псевдообратную матрицу np.linalg.pinv для устранения сингулярности.
        """
        if self._cov_pinv is None or self._cov_mean_vec is None:
            self._fit_mahalanobis_covariance(telemetry)

        if not self._cov_tags or self._cov_pinv is None:
            return pd.Series(0.0, index=telemetry.index)

        # Извлекаем подмножество признаков
        X = telemetry.reindex(columns=self._cov_tags).ffill().bfill().fillna(0.0).values
        diff = X - self._cov_mean_vec

        # Векторизованный расчёт: D_M = sqrt(sum(diff * (diff @ pinv), axis=1))
        left = np.dot(diff, self._cov_pinv)
        dist_sq = np.sum(left * diff, axis=1)
        dist_sq = np.maximum(0.0, dist_sq)
        dist = np.sqrt(dist_sq)

        # Нормируем на корень из размерности пространства признаков
        k = max(1, len(self._cov_tags))
        norm_dist = dist / np.sqrt(k)

        return pd.Series(norm_dist, index=telemetry.index, name='mahalanobis')

    def _calculate_risk_index(
        self,
        zscore_df: pd.DataFrame,
        mahalanobis_dist: pd.Series
    ) -> pd.Series:
        """
        Интегральный риск = среднее по топ-10 |z-score| + нормированный mahalanobis.
        Нормализация в 0..1 (через сигмоиду / clipping).
        """
        abs_z = zscore_df.abs()
        n_cols = abs_z.shape[1]
        k = min(10, n_cols)

        if k > 0:
            # Находим top-k z-score для каждой строки
            top_k_vals = np.sort(abs_z.values, axis=1)[:, -k:]
            mean_top_z = np.mean(top_k_vals, axis=1)
        else:
            mean_top_z = np.zeros(len(zscore_df))

        m_vals = mahalanobis_dist.values

        # Взвешенная сумма отклонений: при z ~ 3 и Mahalanobis ~ 3 риск стремится к 1.0
        # Преобразование через сигмоиду с точкой перегиба в области предкритического режима
        raw_score = 0.6 * (mean_top_z / 3.0) + 0.4 * (m_vals / 3.0)
        
        # Сигмоидальная нормализация в 0..1:
        # При raw_score = 0 (норма): risk ~ 0.05
        # При raw_score = 1 (граница 3-sigma): risk ~ 0.50
        # При raw_score = 2: risk ~ 0.90
        risk_index = 1.0 / (1.0 + np.exp(-4.0 * (raw_score - 1.0)))
        risk_index = np.clip(risk_index, 0.0, 1.0)

        return pd.Series(risk_index, index=zscore_df.index, name='risk_index')

    def _get_risk_factors(
        self,
        zscore_df: pd.DataFrame,
        row_idx: int = -1
    ) -> List[Dict[str, Any]]:
        """
        Определение факторов риска для конкретной строки (по умолчанию последней):
        Для каждого тега: deviation = abs(z_score), weight = 1/num_tags.
        Возвращает топ-3 фактора с наибольшим deviation.
        """
        if zscore_df.empty:
            return []

        row = zscore_df.iloc[row_idx].abs()
        # Сортируем по убыванию отклонения
        sorted_tags = row.sort_values(ascending=False)
        top3_tags = sorted_tags.head(3)

        num_tags = max(1, len(row))
        total_top_dev = top3_tags.sum()

        risk_factors = []
        for tag, dev in top3_tags.items():
            # Вес фактора: доля в общем отклонении топ-факторов
            weight = float(dev / total_top_dev) if total_top_dev > 0 else float(1.0 / num_tags)
            risk_factors.append({
                'tag': str(tag),
                'deviation': round(float(dev), 2),
                'weight': round(weight, 3)
            })

        return risk_factors

    def _get_constraints_for_optimizer(self) -> List[Dict[str, Any]]:
        """
        Формирование списка технологических коридоров для Optimizer Agent.
        Формат: [{"tag": "T6", "min": 345.0, "max": 375.0}, ...]
        """
        constraints = []
        for tag, bounds in self.controlled_bounds.items():
            constraints.append({
                'tag': tag,
                'min': bounds['min'],
                'max': bounds['max']
            })
        return constraints

    def assess_telemetry(self, telemetry: pd.DataFrame) -> ReliabilityAssessment:
        """
        Синхронный метод расчёта оценки надёжности.
        """
        if telemetry.empty:
            return ReliabilityAssessment(
                timestamp=pd.Timestamp.now(),
                risk_index=0.1,
                risk_class="low",
                is_safe=True,
                risk_factors=[],
                constraints_for_optimizer=self._get_constraints_for_optimizer(),
                confidence=0.5,
                assumptions=["Телеметрия пуста, режим принят номинальным"]
            )

        # 1. Z-score
        zscore_df = self._calculate_zscore(telemetry)

        # 2. Mahalanobis distance
        mahalanobis_dist = self._calculate_mahalanobis(telemetry)

        # 3. Risk index
        risk_series = self._calculate_risk_index(zscore_df, mahalanobis_dist)
        latest_risk = float(risk_series.iloc[-1])

        # 4. Категория риска (low, medium, high)
        if latest_risk < 0.35:
            risk_class = "low"
            is_safe = True
        elif latest_risk < 0.70:
            risk_class = "medium"
            is_safe = True
        else:
            risk_class = "high"
            is_safe = False

        # 5. Топ-3 фактора риска
        risk_factors = self._get_risk_factors(zscore_df, row_idx=-1)

        # 6. Коридоры для оптимизатора
        constraints = self._get_constraints_for_optimizer()

        # 7. Временная метка
        ts = pd.Timestamp.now()
        if 'date' in telemetry.columns:
            try:
                ts = pd.to_datetime(telemetry['date'].iloc[-1])
            except Exception:
                pass
        elif isinstance(telemetry.index, pd.DatetimeIndex):
            ts = telemetry.index[-1]

        # Допущения
        assumptions = [
            f"Использован эталонный режим по справочнику ({len(self.norm_mean)} тегов)",
            "Ковариационная матрица регуляризована через pinv",
            f"Число оценённых параметров: {zscore_df.shape[1]}"
        ]

        return ReliabilityAssessment(
            timestamp=ts,
            risk_index=latest_risk,
            risk_class=risk_class,
            is_safe=is_safe,
            risk_factors=risk_factors,
            constraints_for_optimizer=constraints,
            confidence=0.95,
            assumptions=assumptions
        )

    def assess(self, telemetry: Any) -> ReliabilityAssessment:
        """
        Интерфейс для вызова Оркестратором (await self.reliability_agent.assess(request))
        или напрямую в тестах (self.reliability_agent.assess(df)).
        Поддерживает как AgentRequest, так и pd.DataFrame.
        Возвращаемый ReliabilityAssessment поддерживает __await__.
        """
        if hasattr(telemetry, 'telemetry'):
            telemetry = telemetry.telemetry
        return self.assess_telemetry(telemetry)
