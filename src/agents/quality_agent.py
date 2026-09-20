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
        self._train_data: Optional[pd.DataFrame] = None
        self._test_data: Optional[pd.DataFrame] = None

    @property
    def train_data(self) -> pd.DataFrame:
        """Обучающая выборка (до 2026 года) без утечек из будущего (Holdout)."""
        if self._train_data is None:
            train_path = Path('data/processed/train_data.csv')
            if train_path.exists():
                self._train_data = pd.read_csv(train_path, parse_dates=['date'])
            else:
                self._train_data = pd.DataFrame({
                    'date': pd.date_range('2023-01-01', '2025-12-31', freq='1D')
                })
        return self._train_data

    @property
    def test_data(self) -> pd.DataFrame:
        """Тестовая валидационная выборка (2026 год)."""
        if self._test_data is None:
            test_path = Path('data/processed/test_data.csv')
            if test_path.exists():
                self._test_data = pd.read_csv(test_path, parse_dates=['date'])
            else:
                self._test_data = pd.DataFrame({
                    'date': pd.date_range('2026-01-01', '2026-12-31', freq='1D')
                })
        return self._test_data

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

    def _predict_sulfur(
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

    def predict_forecast(
        self,
        telemetry: pd.DataFrame,
        timestamps: Optional[List[Any]] = None,
        horizons_min: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Мультигоризонтный прогноз показателей качества (AGENT-03):
        Горизонты: 30, 60, 120 минут.
        
        Учёт запаздывания отклика (lag_report):
        - Sulfur: lag = 60 мин
        - D15: lag = 30 мин
        - T50/T90/T95: lag = 90 мин
        - CFPP: lag = 30 мин
        - flash: lag = 30 мин
        
        Returns:
            DataFrame с колонками: date, horizon_min, Sulfur, D15, T50, T90, T95, CFPP, flash
        """
        if horizons_min is None:
            horizons_min = [30, 60, 120]

        lags = {
            'Sulfur': 60,
            'D15': 30,
            'T50': 90,
            'T90': 90,
            'T95': 90,
            'CFPP': 30,
            'flash': 30
        }

        # Определение дат
        if 'date' in telemetry.columns:
            dates = pd.to_datetime(telemetry['date'])
        elif isinstance(telemetry.index, pd.DatetimeIndex):
            dates = telemetry.index.to_series()
        else:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=len(telemetry), freq='10min')

        forecast_records = []

        # Базовый шаг дискретизации (10 мин)
        step_min = 10

        for horizon in horizons_min:
            # Для каждого горизонта и целевого показателя рассчитываем эффективное запаздывание
            # tau: время отклика процесса.
            # Если tau > horizon: изменение еще не дошло до выхода, сказывается прошлое состояние на (tau - horizon) мин назад
            # Если tau <= horizon: новое установившееся состояние (текущий режим сохраняется)
            
            # 1. Sulfur
            shift_sulfur = max(0, int((lags['Sulfur'] - horizon) / step_min))
            telem_sulfur = telemetry.shift(shift_sulfur).bfill() if shift_sulfur > 0 else telemetry
            pred_sulfur = self._predict_sulfur(telem_sulfur).values

            # 2. D15
            shift_d15 = max(0, int((lags['D15'] - horizon) / step_min))
            telem_d15 = telemetry.shift(shift_d15).bfill() if shift_d15 > 0 else telemetry
            pred_d15 = vac_d15_godt(telem_d15).values

            # 3. T50, T90, T95
            shift_dist = max(0, int((lags['T95'] - horizon) / step_min))
            telem_dist = telemetry.shift(shift_dist).bfill() if shift_dist > 0 else telemetry
            pred_t50 = vac_t50_godt(telem_dist).values
            pred_t95 = vac_t95_godt(telem_dist).values
            pred_t90 = np.maximum(pred_t50, pred_t95 - 12.0)

            # 4. CFPP
            shift_cfpp = max(0, int((lags['CFPP'] - horizon) / step_min))
            telem_cfpp = telemetry.shift(shift_cfpp).bfill() if shift_cfpp > 0 else telemetry
            pred_cfpp = vac_cfpp_godt(telem_cfpp).values

            # 5. Flash point
            shift_flash = max(0, int((lags['flash'] - horizon) / step_min))
            telem_flash = telemetry.shift(shift_flash).bfill() if shift_flash > 0 else telemetry
            t6_series = self._get_tag_series(telem_flash, 'T6')
            pred_flash = np.clip(68.0 - 0.15 * (t6_series.values - 360.0), 45.0, 90.0)

            df_h = pd.DataFrame({
                'date': dates.values,
                'horizon_min': horizon,
                'Sulfur': pred_sulfur,
                'D15': pred_d15,
                'T50': pred_t50,
                'T90': pred_t90,
                'T95': pred_t95,
                'CFPP': pred_cfpp,
                'flash': pred_flash
            })
            forecast_records.append(df_h)

        result_df = pd.concat(forecast_records, ignore_index=True)

        # Если переданы конкретные timestamps — фильтруем по ним
        if timestamps is not None and len(timestamps) > 0:
            ts_series = pd.to_datetime(timestamps)
            result_df = result_df[result_df['date'].isin(ts_series)].copy()

        return result_df

    def predict(
        self,
        telemetry: pd.DataFrame,
        quality: Optional[Any] = None,
        vac: Optional[pd.Series] = None,
        timestamps: Optional[List[Any]] = None,
        horizons_min: Optional[List[int]] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Универсальный метод прогнозирования (AGENT-02 и AGENT-03):
        - Если передан список timestamps или указан horizons_min -> возвращает DataFrame прогнозов на 30/60/120 мин.
        - Иначе возвращает pd.Series прогноза серы (для совместимости с AGENT-02).
        """
        # Если во 2-й позиционный аргумент передан список timestamps (например, predict(df, [ts1, ts2]))
        if isinstance(quality, (list, pd.DatetimeIndex, np.ndarray)):
            return self.predict_forecast(telemetry, timestamps=list(quality), horizons_min=horizons_min)
        
        if timestamps is not None or horizons_min is not None:
            return self.predict_forecast(telemetry, timestamps=timestamps, horizons_min=horizons_min)

        return self._predict_sulfur(telemetry, quality=quality, vac=vac)

    def assess_risk(
        self,
        forecast: pd.DataFrame,
        specs: Optional[Dict[str, float]] = None,
        window: int = 6
    ) -> pd.DataFrame:
        """
        Оценка риска нарушения спецификаций качества (AGENT-04).
        
        Параметры:
            forecast: DataFrame с прогнозами (Sulfur, T95, D15, CFPP, flash)
            specs: словарь ограничений спецификаций
            window: размер скользящего окна (по умолчанию 6 точек = 60 мин при шаге 10 мин)
            
        Returns:
            DataFrame с рассчитанными вероятностями нарушений:
            P_S_gt_10, P_T95_gt_spec, P_D15_violation, P_CFPP_violation, P_flash_violation, risk_overall
        """
        default_specs = {
            'Sulfur': 10.0,
            'T95': 360.0,
            'D15_min': 820.0,
            'D15_max': 845.0,
            'CFPP': -5.0,
            'flash': 55.0
        }
        if specs:
            default_specs.update(specs)

        risk_df = pd.DataFrame(index=forecast.index)

        # 1. P(S > 10) = доля прогнозов > 10 в скользящем окне 60 мин
        if 'Sulfur' in forecast.columns:
            s_violation = (forecast['Sulfur'] > default_specs['Sulfur']).astype(float)
            risk_df['P_S_gt_10'] = s_violation.rolling(window=window, min_periods=1).mean()
        else:
            risk_df['P_S_gt_10'] = 0.0

        # 2. P(T95 > spec) = доля прогнозов T95 > spec в скользящем окне
        if 'T95' in forecast.columns:
            t95_violation = (forecast['T95'] > default_specs['T95']).astype(float)
            risk_df['P_T95_gt_spec'] = t95_violation.rolling(window=window, min_periods=1).mean()
        else:
            risk_df['P_T95_gt_spec'] = 0.0

        # 3. P(D15 violation)
        if 'D15' in forecast.columns:
            d15_violation = (
                (forecast['D15'] < default_specs['D15_min']) |
                (forecast['D15'] > default_specs['D15_max'])
            ).astype(float)
            risk_df['P_D15_violation'] = d15_violation.rolling(window=window, min_periods=1).mean()
        else:
            risk_df['P_D15_violation'] = 0.0

        # 4. P(CFPP violation)
        if 'CFPP' in forecast.columns:
            cfpp_violation = (forecast['CFPP'] > default_specs['CFPP']).astype(float)
            risk_df['P_CFPP_violation'] = cfpp_violation.rolling(window=window, min_periods=1).mean()
        else:
            risk_df['P_CFPP_violation'] = 0.0

        # 5. P(flash violation)
        if 'flash' in forecast.columns:
            flash_violation = (forecast['flash'] < default_specs['flash']).astype(float)
            risk_df['P_flash_violation'] = flash_violation.rolling(window=window, min_periods=1).mean()
        else:
            risk_df['P_flash_violation'] = 0.0

        # 6. Общий интегральный риск нарушения качества
        risk_df['risk_overall'] = risk_df[
            ['P_S_gt_10', 'P_T95_gt_spec', 'P_D15_violation', 'P_CFPP_violation', 'P_flash_violation']
        ].max(axis=1)

        return pd.concat([forecast, risk_df], axis=1)

    def assess(
        self,
        telemetry: Any,
        quality_data: Optional[pd.DataFrame] = None
    ) -> QualityAssessment:
        """
        Контрактный метод для вызова Оркестратором (src/orchestrator/orchestrator.py).
        Поддерживает как AgentRequest, так и pd.DataFrame.
        Возвращает структурированный dataclass QualityAssessment.
        """
        if hasattr(telemetry, 'telemetry'):
            req = telemetry
            telemetry = req.telemetry
            quality_data = getattr(req, 'quality', quality_data)

        # 1. Расчёт прогноза серы
        pred_sulfur_series = self._predict_sulfur(telemetry, quality_data)
        latest_sulfur = float(pred_sulfur_series.iloc[-1]) if len(pred_sulfur_series) > 0 else 8.5

        # 2. Расчёт сопутствующих показателей качества по формулам ВАК
        d15_series = vac_d15_godt(telemetry)
        t50_series = vac_t50_godt(telemetry)
        t95_series = vac_t95_godt(telemetry)
        cfpp_series = vac_cfpp_godt(telemetry)
        t6_series = self._get_tag_series(telemetry, 'T6')
        latest_t6 = float(t6_series.iloc[-1]) if len(t6_series) > 0 else 360.0
        latest_flash = float(np.clip(68.0 - 0.15 * (latest_t6 - 360.0), 45.0, 90.0))

        latest_d15 = float(d15_series.iloc[-1]) if len(d15_series) > 0 else 835.0
        latest_t50 = float(t50_series.iloc[-1]) if len(t50_series) > 0 else 280.0
        latest_t95 = float(t95_series.iloc[-1]) if len(t95_series) > 0 else 350.0
        latest_t90 = max(latest_t50, latest_t95 - 12.0)
        latest_cfpp = float(cfpp_series.iloc[-1]) if len(cfpp_series) > 0 else -10.0

        # 3. Оценка риска нарушения спецификаций
        # Логистическая аппроксимация вероятности превышения жесткого порога 10.0 мг/кг
        exp_s = np.clip(-3.0 * (latest_sulfur - 10.0), -50.0, 50.0)
        risk_sulfur = float(1.0 / (1.0 + np.exp(exp_s)))
        
        exp_t95 = np.clip(-1.0 * (latest_t95 - 360.0), -50.0, 50.0)
        risk_t95 = float(1.0 / (1.0 + np.exp(exp_t95)))
        
        risk_d15 = 0.05 if (820.0 <= latest_d15 <= 845.0) else 0.85
        
        exp_cfpp = np.clip(-1.0 * (latest_cfpp - (-5.0)), -50.0, 50.0)
        risk_cfpp = float(1.0 / (1.0 + np.exp(exp_cfpp)))
        
        exp_flash = np.clip(1.0 * (latest_flash - 55.0), -50.0, 50.0)
        risk_flash = float(1.0 / (1.0 + np.exp(exp_flash)))

        risk_spec_violation = {
            'P_S_gt_10': risk_sulfur,
            'P_T95_gt_spec': risk_t95,
            'P_D15_violation': risk_d15,
            'P_CFPP_violation': risk_cfpp,
            'P_flash_violation': risk_flash
        }
        risk_overall = max(risk_sulfur, risk_t95, risk_d15, risk_cfpp, risk_flash)

        # 4. Анализ возраста замеров и источников (AGENT-04)
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

        # Оценка уверенности по ТЗ (AGENT-04):
        # confidence = 1.0 - age_min / 240 (снижение доверия с возрастом).
        # Если ЛИМС/ПАК нет >240 мин -> confidence = 0.3.
        if age_min > 240:
            confidence = 0.3
        else:
            confidence = max(0.3, min(1.0, 1.0 - (age_min / 240.0)))

        from datetime import timezone
        ts = datetime.now(timezone.utc)
        if 'date' in telemetry.columns and len(telemetry) > 0:
            try:
                ts = pd.to_datetime(telemetry['date'].iloc[-1]).to_pydatetime()
            except Exception:
                pass

        predictions_dict = {
            'Sulfur': latest_sulfur,
            'D15': latest_d15,
            'T50': latest_t50,
            'T90': latest_t90,
            'T95': latest_t95,
            'CFPP': latest_cfpp,
            'flash': latest_flash
        }

        return QualityAssessment(
            timestamp=ts,
            sulfur_forecast_mg_kg=latest_sulfur,
            predictions=predictions_dict,
            d15_forecast_kg_m3=latest_d15,
            t50_forecast_c=latest_t50,
            t90_forecast_c=latest_t90,
            t95_forecast_c=latest_t95,
            cfpp_forecast_c=latest_cfpp,
            flash_forecast_c=latest_flash,
            risk_spec_violation=risk_spec_violation,
            risk_sulfur_violation=risk_sulfur,
            risk_overall_quality=risk_overall,
            confidence=confidence,
            data_source=data_source,
            age_min=int(age_min),
            lims_age_hours=age_min / 60.0,
            pak_age_minutes=age_min,
            warnings=warnings
        )
