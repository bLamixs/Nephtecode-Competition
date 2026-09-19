"""
Модуль: src/agents/quality_agent.py
Назначение: Quality Agent (Агент Качества).
Коррекция прогнозов качества нефтепродуктов с помощью ML-модели на остатках (LIMS - ВАК).

Контекст:
Виртуальный анализатор качества (ВАК) даёт аналитический baseline, однако имеет
систематические смещения из-за дезактивации катализатора, изменения состава сырья
и дрейфа датчиков. Quality Agent использует градиентный бустинг (LightGBM/HistGradientBoosting)
или регуляризованную линейную модель (Ridge) для предсказания остатков (Residuals = LIMS - VAC)
на основе лагов режимных параметров (T6, F26, P8...), возраста лабораторных анализов (age_min)
и нормированных отклонений (z-score).
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from src.utils.logging_config import setup_logger
from src.agents.interfaces import QualityAssessment
from src.utils.vac_formulas import (
    vac_sulfur_24_2000,
    vac_d15_godt,
    vac_t50_godt,
    vac_t95_godt,
    vac_cfpp_godt,
    _get_series
)

logger = setup_logger('quality_agent')


class QualityAgent:
    """
    Агент качества технологического процесса (AGENT-02).
    Отвечает за оценку физико-химических показателей качества (сера, фракционка, плотность, ПТФ)
    и оценку риска нарушения технологических норм.
    """

    FEATURE_COLS = [
        'T6', 'T6_lag10', 'T6_lag30', 'T6_lag60', 'T6_zscore',
        'F26', 'F26_lag10', 'F26_lag30', 'F26_lag60', 'F26_zscore',
        'P8', 'P8_lag10', 'P8_lag30', 'P8_lag60', 'P8_zscore',
        'T11', 'T11_lag10', 'T11_lag30', 'T11_lag60',
        'F9', 'P13', 'W7', 'age_min', 'is_lims'
    ]

    def __init__(self, model_path: str = 'output/models/quality_lgbm.pkl'):
        self.model_path = model_path
        self.model = self._load_model(model_path) if os.path.exists(model_path) else None

    def _load_model(self, model_path: str):
        """Загрузка обученной модели из файла."""
        try:
            model = joblib.load(model_path)
            logger.info(f"Модель QualityAgent успешно загружена из: {model_path}")
            return model
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель из {model_path}: {e}")
            return None

    def _get_tag_series(self, df: pd.DataFrame, tag: str) -> pd.Series:
        """Извлечение серии тега с учетом возможных префиксов/суффиксов установки."""
        return _get_series(df, tag, installation='hydro')

    def _prepare_features(
        self,
        telemetry: pd.DataFrame,
        quality: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Формирование матрицы признаков для модели прогноза остатков.

        Признаки:
        - Лаги режимных параметров (T6, F26, P8, T11): текущие, lag10, lag30, lag60
        - Отклонения от технологической нормы (Z-score): T6_zscore, F26_zscore, P8_zscore
        - Возраст анализа: age_min
        - Источник данных: is_lims (1 для лабораторного анализа, 0 для ПАК)
        """
        features = pd.DataFrame(index=telemetry.index)

        # 1. Извлечение базовых сигналов
        t6 = self._get_tag_series(telemetry, 'T6')
        f26 = self._get_tag_series(telemetry, 'F26')
        p8 = self._get_tag_series(telemetry, 'P8')
        t11 = self._get_tag_series(telemetry, 'T11')
        f9 = self._get_tag_series(telemetry, 'F9')
        p13 = self._get_tag_series(telemetry, 'P13')
        w7 = self._get_tag_series(telemetry, 'W7')

        features['T6'] = t6
        features['F26'] = f26
        features['P8'] = p8
        features['T11'] = t11
        features['F9'] = f9
        features['P13'] = p13
        features['W7'] = w7

        # 2. Формирование временных лагов (10, 30, 60 минут)
        # Если шаг телеметрии 1 минута: сдвиг 10, 30, 60.
        # Если шаг 10 минут: сдвиг 1, 3, 6.
        step_min = 1
        if 'date' in telemetry.columns and len(telemetry) > 1:
            diff_sec = (pd.to_datetime(telemetry['date'].iloc[1]) - pd.to_datetime(telemetry['date'].iloc[0])).total_seconds()
            step_min = max(1, int(round(diff_sec / 60.0)))

        s10 = max(1, 10 // step_min)
        s30 = max(1, 30 // step_min)
        s60 = max(1, 60 // step_min)

        for tag_name, s in [('T6', t6), ('F26', f26), ('P8', p8), ('T11', t11)]:
            features[f'{tag_name}_lag10'] = s.shift(s10).bfill().ffill()
            features[f'{tag_name}_lag30'] = s.shift(s30).bfill().ffill()
            features[f'{tag_name}_lag60'] = s.shift(s60).bfill().ffill()

        # 3. Z-score отклонения от средних значений
        for tag_name in ['T6', 'F26', 'P8']:
            mean_val = features[tag_name].mean()
            std_val = features[tag_name].std()
            if std_val is None or std_val == 0 or np.isnan(std_val):
                std_val = 1.0
            features[f'{tag_name}_zscore'] = (features[tag_name] - mean_val) / std_val

        # 4. Признаки качества: age_min и is_lims
        if quality is not None and 'age_min' in quality.columns:
            features['age_min'] = pd.to_numeric(quality['age_min'], errors='coerce').fillna(0.0)
        elif 'age_min' in telemetry.columns:
            features['age_min'] = pd.to_numeric(telemetry['age_min'], errors='coerce').fillna(0.0)
        else:
            features['age_min'] = 0.0

        if quality is not None and 'source' in quality.columns:
            features['is_lims'] = (quality['source'].str.upper() == 'LIMS').astype(float)
        elif 'source' in telemetry.columns:
            features['is_lims'] = (telemetry['source'].str.upper() == 'LIMS').astype(float)
        else:
            features['is_lims'] = 1.0

        # Упорядочиваем колонки
        cols_to_use = [col for col in self.FEATURE_COLS if col in features.columns]
        return features[cols_to_use].fillna(0.0)

    def _calculate_residuals(
        self,
        actual: Union[pd.DataFrame, pd.Series],
        vac: Union[pd.DataFrame, pd.Series]
    ) -> pd.Series:
        """
        Расчёт остатков модели: y = Фактическое качество - ВАК.
        """
        y_act = actual['value'] if isinstance(actual, pd.DataFrame) and 'value' in actual.columns else (
            actual['value_quality'] if isinstance(actual, pd.DataFrame) and 'value_quality' in actual.columns else actual
        )
        y_vac = vac['vac_value'] if isinstance(vac, pd.DataFrame) and 'vac_value' in vac.columns else vac
        return pd.Series(y_act.values - y_vac.values, index=y_act.index, name='residual')

    def train(
        self,
        telemetry: pd.DataFrame,
        quality: pd.DataFrame,
        vac: pd.DataFrame,
        model_type: str = 'auto'
    ):
        """
        Обучение ML-модели на остатках (LIMS - ВАК).

        Использует:
        - LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
        - Автоматический fallback на HistGradientBoostingRegressor / Ridge(alpha=1.0)
        """
        logger.info(f"Подготовка данных для обучения QualityAgent ({len(telemetry)} строк)...")
        X = self._prepare_features(telemetry, quality)
        y = self._calculate_residuals(quality, vac)

        # Отсекаем пропуски в целевой переменной
        valid_mask = ~y.isna() & ~np.isinf(y)
        X_train = X[valid_mask]
        y_train = y[valid_mask]

        logger.info(f"Обучающая выборка: {len(X_train)} строк, {X_train.shape[1]} признаков.")

        regressor = None
        if model_type in ('auto', 'lgbm'):
            try:
                import lightgbm as lgb
                regressor = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
                regressor.fit(X_train, y_train)
                logger.info("Модель LightGBM успешно обучена.")
            except Exception as e:
                logger.warning(f"LightGBM недоступен ({e}). Переключение на HistGradientBoostingRegressor.")

        if regressor is None:
            if len(X_train) < 500:
                from sklearn.linear_model import Ridge
                regressor = Ridge(alpha=1.0)
                logger.info("Размер выборки < 500: использование Ridge(alpha=1.0).")
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                regressor = HistGradientBoostingRegressor(
                    max_iter=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                logger.info("Использование HistGradientBoostingRegressor.")
            regressor.fit(X_train, y_train)

        self.model = regressor

        # Сохранение модели
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Модель сохранена в: {self.model_path}")
        return self.model

    def predict(
        self,
        telemetry: pd.DataFrame,
        quality: Optional[pd.DataFrame] = None,
        vac: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Прогноз содержания серы с гибридной коррекцией:
        Y = ВАК(T6, F26) + ML_Residual_Correction
        """
        if vac is None:
            vac_base = vac_sulfur_24_2000(telemetry)
        else:
            vac_base = vac

        if self.model is None:
            logger.warning("Модель не обучена/не загружена, возврат baseline ВАК.")
            return vac_base

        X = self._prepare_features(telemetry, quality)
        correction = self.model.predict(X)
        corrected_pred = np.maximum(0.0, vac_base.values + correction)
        return pd.Series(corrected_pred, index=telemetry.index, name='predicted_sulfur')

    async def assess(
        self,
        telemetry: pd.DataFrame,
        quality_data: Optional[pd.DataFrame] = None
    ) -> QualityAssessment:
        """
        Контрактный метод для вызова Оркестратором (src/orchestrator/orchestrator.py).
        Возвращает структурированный dataclass QualityAssessment.
        """
        # 1. Расчёт прогноза серы
        pred_sulfur_series = self.predict(telemetry, quality_data)
        latest_sulfur = float(pred_sulfur_series.iloc[-1]) if len(pred_sulfur_series) > 0 else 8.5

        # 2. Расчёт сопутствующих показателей качества по формулам ВАК
        d15_series = vac_d15_godt(telemetry)
        t50_series = vac_t50_godt(telemetry)
        t95_series = vac_t95_godt(telemetry)
        cfpp_series = vac_cfpp_godt(telemetry)

        latest_d15 = float(d15_series.iloc[-1]) if len(d15_series) > 0 else 835.0
        latest_t50 = float(t50_series.iloc[-1]) if len(t50_series) > 0 else 280.0
        latest_t95 = float(t95_series.iloc[-1]) if len(t95_series) > 0 else 350.0
        latest_cfpp = float(cfpp_series.iloc[-1]) if len(cfpp_series) > 0 else -10.0

        # 3. Оценка риска нарушения спецификации P(S > 10 ppm)
        # Логистическая аппроксимация вероятности превышения жесткого порога 10.0 мг/кг
        # При сере 9.0 мг/кг риск мал (~0.05), при сере 10.0 мг/кг риск = 0.50, при сере 11.0 риск = 0.95
        risk_sulfur = float(1.0 / (1.0 + np.exp(-3.0 * (latest_sulfur - 10.0))))

        # 4. Анализ возраста замеров и источников
        age_min = 0.0
        data_source = "HYBRID_VAC_ML"
        warnings = []

        if quality_data is not None and not quality_data.empty:
            if 'age_min' in quality_data.columns:
                age_min = float(quality_data['age_min'].max())
            if 'source' in quality_data.columns:
                src_set = set(quality_data['source'].dropna().str.upper())
                if 'LIMS' in src_set:
                    data_source = "LIMS+VAC_ML"
                elif 'PAK' in src_set:
                    data_source = "PAK+VAC_ML"

        if age_min > 120:
            warnings.append(f"Анализы устарели: age_min={age_min:.0f} мин > 120 мин")

        # Доверительный интервал
        confidence = max(0.2, min(1.0, 1.0 - (age_min / 300.0)))

        from datetime import timezone
        ts = datetime.now(timezone.utc)
        if 'date' in telemetry.columns and len(telemetry) > 0:
            try:
                ts = pd.to_datetime(telemetry['date'].iloc[-1]).to_pydatetime()
            except Exception:
                pass

        return QualityAssessment(
            timestamp=ts,
            sulfur_forecast_mg_kg=latest_sulfur,
            d15_forecast_kg_m3=latest_d15,
            t50_forecast_c=latest_t50,
            t90_forecast_c=None,
            t95_forecast_c=latest_t95,
            cfpp_forecast_c=latest_cfpp,
            risk_sulfur_violation=risk_sulfur,
            risk_overall_quality=risk_sulfur,
            confidence=confidence,
            data_source=data_source,
            lims_age_hours=age_min / 60.0,
            pak_age_minutes=age_min,
            warnings=warnings
        )
