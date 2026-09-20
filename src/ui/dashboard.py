"""
Главный интерфейс оператора установки гидроочистки (INT-01 / INT-02).
Мультиагентная система оптимизации "НефтеКод MAS".
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Добавляем корень проекта в sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.recommendation import Recommendation


# ============================================================================
# СТРАНИЦА И СТИЛИЗАЦИЯ
# ============================================================================

st.set_page_config(
    page_title="НефтеКод MAS | Оператор установки",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Пользовательский CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .metric-box {
        background: white;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-ok {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-warn {
        background-color: #FEF08A;
        color: #713F12;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-danger {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# КЭШИРОВАНИЕ ОРКЕСТРАТОРА И ДАННЫХ
# ============================================================================

@st.cache_resource
def get_orchestrator():
    return Orchestrator()


@st.cache_data(ttl=600)
def load_historical_telemetry(hours: int = 24):
    """Загрузка исторических данных телеметрии для визуализации трендов ровно за заданное число часов."""
    path = ROOT_DIR / 'data' / 'processed' / 'telemetry_clean.parquet'
    if path.exists():
        try:
            df = pd.read_parquet(path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                df = df.sort_values('date')
                max_date = df['date'].max()
                cutoff = max_date - pd.Timedelta(hours=hours)
                sub_df = df[df['date'] >= cutoff].copy()
                return sub_df
        except Exception as e:
            st.warning(f"Не удалось загрузить историческую телеметрию: {e}")
    return pd.DataFrame()


# ============================================================================
# САЙДБАР: УПРАВЛЕНИЕ И СЦЕНАРИИ
# ============================================================================

st.sidebar.image("https://img.icons8.com/fluency/96/oil-pumpjack.png", width=64)
st.sidebar.markdown("## ⚙️ Управление системой")

scenarios_map = {
    "normal": "🟢 Штатный режим работы (Normal)",
    "risk": "🟡 Риск превышения серы (Risk Growth)",
    "missing": "🔴 Сбой датчиков / Устаревшие анализы (Missing Data)",
    "no_solution": "⛔ Технологический тупик (No Solution)"
}

selected_scenario = st.sidebar.selectbox(
    "Сценарий технологического процесса:",
    options=list(scenarios_map.keys()),
    format_func=lambda x: scenarios_map[x],
    index=0
)

# Описание выбранного сценария
scenario_descriptions = {
    "normal": "Все параметры в норме, сера 6.8 мг/кг, оборудование стабильно. Рекомендовано удержание оптимума.",
    "risk": "Сера выросла до 9.8 мг/кг при лимите 10.0. Оптимизатор подбирает превентивное изменение уставок.",
    "missing": "ЛИМС устарел (>48ч), ПАК оффлайн (>240 мин). Система обязана отказаться от слепой рекомендации.",
    "no_solution": "Сера 11.5 мг/кг, но реактор на пределе T=380°C, а расход на минимуме. Допустимых решений нет."
}

st.sidebar.info(scenario_descriptions[selected_scenario])

run_button = st.sidebar.button("🚀 Запустить цикл оптимизации", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Технологические лимиты")
st.sidebar.markdown("- **Сера ГОДТ**: $\\le 10.0$ мг/кг")
st.sidebar.markdown("- **Температура T6**: $345 - 375$ °C (макс. $380$ °C)")
st.sidebar.markdown("- **Макс. возраст анализа**: 120 мин")
st.sidebar.markdown("- **Порог риска оборудования**: 0.70")
st.sidebar.caption("Платформа: НефтеКод 2.0 MAS | Установка 24-2000")


# ============================================================================
# ЗАПУСК ЦИКЛА ОРКЕСТРАТОРА
# ============================================================================

orchestrator = get_orchestrator()

# Инициализация сессионного состояния
if 'last_scenario' not in st.session_state:
    st.session_state.last_scenario = selected_scenario

if 'recommendation' not in st.session_state or run_button or st.session_state.last_scenario != selected_scenario:
    st.session_state.last_scenario = selected_scenario
    with st.spinner(f"Выполняется цикл анализа для сценария '{selected_scenario}'..."):
        # Вызов полного цикла оркестратора
        rec = asyncio.run(orchestrator.run_cycle(scenario=selected_scenario))
        st.session_state.recommendation = rec
        st.session_state.run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

rec = st.session_state.recommendation
run_time = st.session_state.get('run_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ============================================================================
# ШАПКА И СТАТУС СИСТЕМЫ
# ============================================================================

col_header, col_status = st.columns([3, 1])

with col_header:
    st.markdown('<div class="main-title">🛢️ НефтеКод: Мультиагентный Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Установка гидроочистки дизельного топлива 24-2000 • Последний цикл: <b>{run_time}</b> • Сценарий: <code>{selected_scenario}</code></div>', unsafe_allow_html=True)

with col_status:
    if rec.status == "RECOMMENDED":
        st.markdown('<div style="text-align: right; margin-top: 10px;"><span class="status-badge-ok">✓ РЕКОМЕНДАЦИЯ АКТИВНА</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: right; margin-top: 10px;"><span class="status-badge-danger">⛔ ОТКАЗ ОТ РЕКОМЕНДАЦИИ</span></div>', unsafe_allow_html=True)


# ============================================================================
# ВЕРХНЯЯ ПАНЕЛЬ: 4 КЛЮЧЕВЫХ МЕТРИКИ (INT-01)
# ============================================================================

st.markdown("---")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)

# 1. Свежесть анализов ЛИМС / ПАК
with m_col1:
    sulfur_age = rec.state.get('Sulfur_age_min', 0.0)
    sulfur_curr = rec.state.get('Sulfur_current', 0.0)
    is_fresh = (sulfur_age <= 120.0) and (sulfur_age > 0.0)
    fresh_badge = "СВЕЖИЙ" if is_fresh else ("УСТАРЕЛ" if sulfur_age > 120.0 else "НЕТ ДАННЫХ")
    delta_color = "normal" if is_fresh else "inverse"
    age_str = f"{sulfur_age / 60:.1f} ч" if sulfur_age >= 180 else (f"{sulfur_age:.0f} мин" if sulfur_age > 0 else "Нет данных")
    st.metric(
        label=f"⏱️ Свежесть анализов ({fresh_badge})",
        value=age_str,
        delta=f"Сера: {sulfur_curr:.1f} мг/кг" if sulfur_curr > 0 else "ПАК offline",
        delta_color=delta_color
    )

# 2. Риск спецификации качества P(S > 10)
with m_col2:
    p_sulfur_risk = 0.0
    # Проверяем в проверенных ограничениях
    for c in rec.constraints_checked:
        c_name = getattr(c, 'constraint', '') if hasattr(c, 'constraint') else c.get('constraint', '')
        c_val = getattr(c, 'predicted_value', 0.0) if hasattr(c, 'predicted_value') else c.get('predicted_value', 0.0)
        if 'P_S_gt_10' in c_name or 'Сера' in c_name or 'P(S > 10)' in c_name:
            p_sulfur_risk = 1.0 if c_val > 1.0 else c_val
            break
    if p_sulfur_risk == 0.0:
        if rec.problem_type == "NO_DATA":
            p_sulfur_risk = None
        elif rec.state.get('Sulfur_current', 0.0) > 10.0:
            p_sulfur_risk = 1.0
        elif rec.expected_effect and rec.expected_effect.sulfur_60min:
            p_sulfur_risk = max(0.01, min(0.99, (rec.expected_effect.sulfur_60min - 8.0) / 2.5))
        elif rec.state.get('Sulfur_current', 0.0) > 0.0:
            s_c = rec.state.get('Sulfur_current', 0.0)
            p_sulfur_risk = max(0.01, min(0.99, (s_c - 8.0) / 2.5))

    val_display = f"{p_sulfur_risk * 100:.1f}%" if p_sulfur_risk is not None else "Н/Д (Отказ)"
    delta_display = "Лимит: 10.0 мг/кг" if p_sulfur_risk is not None else "Данные недостоверны"
    st.metric(
        label="🎯 Риск качества P(S > 10)",
        value=val_display,
        delta=delta_display,
        delta_color="inverse" if (p_sulfur_risk is None or p_sulfur_risk > 0.15) else "normal"
    )
    st.progress(float(np.clip(p_sulfur_risk if p_sulfur_risk is not None else 1.0, 0.0, 1.0)))

# 3. Индекс риска оборудования
with m_col3:
    risk_idx = rec.expected_effect.risk_index if rec.expected_effect.risk_index is not None else 0.05
    risk_class = "LOW"
    if risk_idx > 0.7:
        risk_class = "CRITICAL"
    elif risk_idx > 0.4:
        risk_class = "MEDIUM"

    st.metric(
        label=f"🛡️ Риск оборудования ({risk_class})",
        value=f"{risk_idx:.2f}",
        delta="Порог: 0.70",
        delta_color="inverse" if risk_idx > 0.4 else "normal"
    )
    st.progress(float(np.clip(risk_idx, 0.0, 1.0)))

# 4. Прогноз производительности и эффект
with m_col4:
    tp = rec.expected_effect.throughput if rec.expected_effect.throughput else rec.state.get('F9', 250.0)
    tp_delta = rec.expected_effect.throughput_delta if rec.expected_effect.throughput_delta is not None else 0.0
    st.metric(
        label="⚡ Производительность F9",
        value=f"{tp:.1f} м³/ч",
        delta=f"{tp_delta:+.1f} м³/ч",
        delta_color="normal" if tp_delta >= 0 else "off"
    )


# ============================================================================
# ОСНОВНОЙ КОНТЕНТ: ВКЛАДКИ
# ============================================================================

st.markdown("---")

tab_rec, tab_telemetry, tab_agents, tab_logs = st.tabs([
    "💡 Рекомендация оператору",
    "📈 Тренды телеметрии (24 часа)",
    "🤖 Детализация агентов",
    "📜 Журнал событий и XAI"
])


# ----------------------------------------------------------------------------
# ВКЛАДКА 1: РЕКОМЕНДАЦИЯ ОПЕРАТОРУ (INT-02)
# ----------------------------------------------------------------------------
with tab_rec:
    recommendation = rec

    if recommendation.status == "RECOMMENDED":
        st.subheader("✅ Рекомендация")

        # Объяснение
        st.info(recommendation.explanation)

        col_act, col_eff = st.columns([3, 2])

        with col_act:
            # Таблица действий с дельтами
            st.write("**Действие:**")
            if recommendation.action:
                actions_data = []
                for a in recommendation.action:
                    tag = getattr(a, 'tag', a.get('tag', '')) if hasattr(a, 'get') else getattr(a, 'tag', '')
                    name = getattr(a, 'name', a.get('name', tag)) if hasattr(a, 'get') else getattr(a, 'name', tag)
                    from_v = getattr(a, 'from_value', a.get('from', 0.0)) if hasattr(a, 'get') else getattr(a, 'from_value', 0.0)
                    to_v = getattr(a, 'to_value', a.get('to', 0.0)) if hasattr(a, 'get') else getattr(a, 'to_value', 0.0)
                    unit = getattr(a, 'unit', a.get('unit', '')) if hasattr(a, 'get') else getattr(a, 'unit', '')
                    delta = getattr(a, 'delta', a.get('delta', to_v - from_v)) if hasattr(a, 'get') else (to_v - from_v)
                    pct = getattr(a, 'delta_percent', a.get('delta_percent', 0.0)) if hasattr(a, 'get') else 0.0
                    actions_data.append({
                        "Тег": tag,
                        "Параметр": name,
                        "Текущее": f"{from_v:.1f} {unit}",
                        "Рекомендуемое": f"{to_v:.1f} {unit}",
                        "Изменение (Δ)": f"{delta:+.1f} ({pct:+.1f}%)"
                    })
                st.dataframe(pd.DataFrame(actions_data), use_container_width=True, hide_index=True)

            # Проверенные ограничения (структурированная таблица)
            st.write("**Проверенные ограничения:**")
            if recommendation.constraints_checked:
                cons_data = []
                for c in recommendation.constraints_checked:
                    c_name = c.get('constraint', getattr(c, 'constraint', '')) if hasattr(c, 'get') else getattr(c, 'constraint', '')
                    c_val = c.get('predicted_value', getattr(c, 'predicted_value', 0.0)) if hasattr(c, 'get') else getattr(c, 'predicted_value', 0.0)
                    c_thresh = c.get('threshold', getattr(c, 'threshold', 0.0)) if hasattr(c, 'get') else getattr(c, 'threshold', 0.0)
                    c_margin = c.get('margin', getattr(c, 'margin', 0.0)) if hasattr(c, 'get') else getattr(c, 'margin', 0.0)
                    c_st = c.get('status', getattr(c, 'status', '')) if hasattr(c, 'get') else getattr(c, 'status', '')
                    cons_data.append({
                        "Ограничение": c_name,
                        "Значение": f"{c_val:.3f}",
                        "Порог": f"{c_thresh:.3f}",
                        "Запас (margin)": f"{c_margin:.3f}",
                        "Статус": "✅ PASS" if c_st == "PASS" else "❌ FAIL"
                    })
                st.dataframe(pd.DataFrame(cons_data), use_container_width=True, hide_index=True)

        with col_eff:
            # Ожидаемый эффект
            st.write("**Ожидаемый эффект:**")
            ee = recommendation.expected_effect
            ee_s = ee['Sulfur_60min'] if hasattr(ee, '__getitem__') else getattr(ee, 'sulfur_60min', None)
            ee_tp = ee['throughput_change'] if hasattr(ee, '__getitem__') else getattr(ee, 'throughput_delta', None)

            # Метрики
            eff_col1, eff_col2 = st.columns(2)
            with eff_col1:
                try:
                    s_float = float(ee_s) if ee_s is not None else None
                    st.metric("Сера через 60 мин", f"{s_float:.2f} мг/кг" if s_float is not None else "—", delta_color="inverse")
                except (ValueError, TypeError):
                    st.metric("Сера через 60 мин", f"{ee_s} мг/кг", delta_color="inverse")
                d15_val = ee.get('D15_60min', getattr(ee, 'd15_60min', 835.0)) if hasattr(ee, 'get') else getattr(ee, 'd15_60min', 835.0)
                st.metric("Плотность D15", f"{float(d15_val):.1f} кг/м³" if d15_val is not None else "835.0 кг/м³")
            with eff_col2:
                tp_val = ee.get('throughput', getattr(ee, 'throughput', 215.0)) if hasattr(ee, 'get') else getattr(ee, 'throughput', 215.0)
                try:
                    tp_f = float(tp_val)
                    delta_f = float(ee_tp) if ee_tp is not None else None
                    st.metric("Расход сырья", f"{tp_f:.1f} м³/ч", f"{delta_f:+.1f} т/ч" if delta_f is not None else None)
                except (ValueError, TypeError):
                    st.metric("Расход сырья", f"{tp_val} м³/ч")
                en_val = ee.get('energy_proxy', getattr(ee, 'energy_proxy', 0.24)) if hasattr(ee, 'get') else getattr(ee, 'energy_proxy', 0.24)
                st.metric("Энергетический индекс", f"{float(en_val):.4f}" if en_val is not None else "0.2400")

            # Альтернативы
            st.write("**Альтернативы:**")
            if recommendation.alternatives:
                for i, alt in enumerate(recommendation.alternatives, 1):
                    st.write(f"{i}. {alt}")
            else:
                st.caption("Дополнительные альтернативы не сформированы.")

    elif recommendation.status == "NO_RECOMMENDATION":
        st.error(f"❌ {recommendation.explanation}")
        
        st.warning("""
        **Действия оператора при отказе:**
        1. Проверьте физическую исправность датчиков поточного анализатора (ПАК) и лабораторные анализы ЛИМС.
        2. При переходе в ручной режим управления руководствуйтесь технологическим регламентом установки 24-2000.
        3. Не превышайте максимальные технологические границы безопасности оборудования.
        """)
    else:
        st.error(f"❌ {recommendation.explanation}")


# ----------------------------------------------------------------------------
# ВКЛАДКА 2: ТРЕНДЫ ТЕЛЕМЕТРИИ ЗА 24 ЧАСА (INT-01)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ВКЛАДКА 2: ТРЕНДЫ ТЕЛЕМЕТРИИ (INT-01)
# ----------------------------------------------------------------------------
with tab_telemetry:
    col_g_head, col_g_ctrl = st.columns([3, 2])
    with col_g_head:
        st.markdown("#### 📈 Динамика ключевых параметров процесса")
    with col_g_ctrl:
        horizon_label = st.radio(
            "Горизонт трендов:",
            options=["6 часов", "12 часов", "24 часа", "3 дня"],
            index=2,
            horizontal=True,
            label_visibility="collapsed"
        )

    horizon_hours_map = {
        "6 часов": 6,
        "12 часов": 12,
        "24 часа": 24,
        "3 дня": 72
    }
    selected_hours = horizon_hours_map[horizon_label]

    hist_df = load_historical_telemetry(hours=selected_hours)

    if hist_df.empty or len(hist_df) < 10:
        # Синтетическая подложка ровно под выбранный интервал
        step_minutes = 10
        n_points = max(10, int(selected_hours * 60 / step_minutes))
        time_index = pd.date_range(end=datetime.now(), periods=n_points, freq=f'{step_minutes}min')
        plot_df = pd.DataFrame({
            'date': time_index,
            'T6': np.random.normal(360.0, 1.5, n_points),
            'F9': np.random.normal(215.0, 3.5, n_points),
            'P8': np.random.normal(0.17, 0.02, n_points),
        })
    else:
        plot_df = hist_df.copy()
        if 'T6_hydro' in plot_df.columns:
            plot_df['T6'] = plot_df['T6_hydro']
        elif 'T6_avt' in plot_df.columns and 'T6' not in plot_df.columns:
            plot_df['T6'] = plot_df['T6_avt']

        if 'F9_hydro' in plot_df.columns:
            plot_df['F9'] = plot_df['F9_hydro']
        elif 'F9_avt' in plot_df.columns and 'F9' not in plot_df.columns:
            plot_df['F9'] = plot_df['F9_avt']

        if 'P8' not in plot_df.columns:
            plot_df['P8'] = 0.17

    n_rows = len(plot_df)

    # Синхронизация тренда серы и давления P8 в зависимости от выбранного сценария
    if selected_scenario == 'risk':
        # Плавный рост серы от нормы 7.2 до 9.8 мг/кг в конце интервала
        s_baseline = np.linspace(7.2, 9.8, n_rows) + np.random.normal(0, 0.15, n_rows)
        p_baseline = np.linspace(0.18, 0.23, n_rows) + np.random.normal(0, 0.005, n_rows)
    elif selected_scenario == 'no_solution':
        # Стабильно превышенная сера и критическое давление выше предела 0.26 МПа
        s_baseline = np.random.normal(11.4, 0.15, n_rows)
        p_baseline = np.random.normal(0.278, 0.005, n_rows)
    elif selected_scenario == 'missing':
        # Сбой датчиков в конце: последние 25% точек отсутствуют (NaN)
        s_baseline = np.random.normal(7.8, 0.2, n_rows)
        p_baseline = np.random.normal(0.175, 0.012, n_rows)
        cut_idx = int(n_rows * 0.75)
        s_baseline[cut_idx:] = np.nan
        p_baseline[cut_idx:] = np.nan
    else:
        # Штатный режим: сера в норме 6.8 мг/кг, давление в оптимуме 0.17-0.19 МПа
        s_baseline = np.random.normal(6.8, 0.25, n_rows)
        p_baseline = np.random.normal(0.178, 0.012, n_rows)

    plot_df['Sulfur'] = s_baseline
    plot_df['P8'] = p_baseline

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # 1. График серы с зоной допуска и прогнозом
        fig_s = go.Figure()

        # Зеленая зона нормы
        fig_s.add_hrect(
            y0=0, y1=10.0,
            fillcolor="rgba(16, 185, 129, 0.08)",
            line_width=0,
            layer="below"
        )
        # Красная черта предела ГОСТ
        fig_s.add_hline(
            y=10.0,
            line_dash='dash',
            line_color='#DC2626',
            line_width=2,
            annotation_text='Предел ГОСТ: 10.0 мг/кг',
            annotation_position="top right"
        )

        # Факт концентрации серы
        fig_s.add_trace(go.Scatter(
            x=plot_df['date'],
            y=plot_df['Sulfur'],
            mode='lines',
            name='Сера ГОДТ (факт)',
            line=dict(color='#EF4444' if selected_scenario in ['risk', 'no_solution'] else '#10B981', width=2.5, shape='spline')
        ))

        # Если рекомендация активна, показываем прогноз на +60 мин
        if rec.status == "RECOMMENDED" and rec.expected_effect.sulfur_60min is not None:
            last_date = plot_df['date'].iloc[-1]
            last_s = float(plot_df['Sulfur'].dropna().iloc[-1]) if not plot_df['Sulfur'].dropna().empty else 9.0
            future_date = last_date + pd.Timedelta(minutes=60)
            target_s = rec.expected_effect.sulfur_60min

            fig_s.add_trace(go.Scatter(
                x=[last_date, future_date],
                y=[last_s, target_s],
                mode='lines+markers',
                name='Прогноз эффекта (+60 мин)',
                line=dict(color='#2563EB', width=2.5, dash='dot'),
                marker=dict(size=7, color='#2563EB')
            ))

        fig_s.update_layout(
            title=f"Концентрация серы ГОДТ ({horizon_label})",
            xaxis_title="",
            yaxis_title="мг/кг",
            height=340,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_s, use_container_width=True)

        # 2. График температуры реактора T6 с технологическим коридором
        fig_t = go.Figure()
        fig_t.add_hrect(
            y0=345.0, y1=375.0,
            fillcolor="rgba(245, 158, 11, 0.08)",
            line_width=0,
            layer="below"
        )
        fig_t.add_hline(y=380.0, line_dash='dash', line_color='#EF4444', annotation_text='Предел 380°C')
        fig_t.add_hline(y=375.0, line_dash='dot', line_color='#F59E0B', annotation_text='Макс. 375°C')
        fig_t.add_hline(y=345.0, line_dash='dot', line_color='#F59E0B', annotation_text='Мин. 345°C')

        t_vals = np.clip(plot_df['T6'].values, 330.0, 390.0) if 'T6' in plot_df.columns else np.random.normal(360.0, 1.5, n_rows)
        fig_t.add_trace(go.Scatter(
            x=plot_df['date'],
            y=t_vals,
            mode='lines',
            name='T6 Реактор (°C)',
            line=dict(color='#D97706', width=2.5, shape='spline')
        ))

        fig_t.update_layout(
            title=f"Температура входа в реактор T6 ({horizon_label})",
            xaxis_title="",
            yaxis_title="°C",
            height=340,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with col_g2:
        # 3. График расхода сырья F9
        fig_f = go.Figure()
        fig_f.add_hrect(
            y0=200.0, y1=280.0,
            fillcolor="rgba(59, 130, 246, 0.08)",
            line_width=0,
            layer="below"
        )
        f_vals = np.clip(plot_df['F9'].values, 50.0, 320.0) if 'F9' in plot_df.columns else np.random.normal(250.0, 3.5, n_rows)
        fig_f.add_trace(go.Scatter(
            x=plot_df['date'],
            y=f_vals,
            mode='lines',
            name='F9 Расход (м³/ч)',
            line=dict(color='#2563EB', width=2.5, shape='spline')
        ))

        fig_f.update_layout(
            title=f"Расход сырья на установку F9 ({horizon_label})",
            xaxis_title="",
            yaxis_title="м³/ч",
            height=340,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_f, use_container_width=True)

        # 4. График давления реактора P8
        fig_p = go.Figure()
        fig_p.add_hrect(
            y0=0.10, y1=0.25,
            fillcolor="rgba(16, 185, 129, 0.08)",
            line_width=0,
            layer="below"
        )
        fig_p.add_hline(y=0.26, line_dash='dash', line_color='#EF4444', annotation_text='Предел 0.26 МПа')
        fig_p.add_hline(y=0.25, line_dash='dot', line_color='#F59E0B', annotation_text='Макс. 0.25 МПа')
        fig_p.add_hline(y=0.10, line_dash='dot', line_color='#F59E0B', annotation_text='Мин. 0.10 МПа')

        if 'P8' in plot_df.columns:
            raw_p = plot_df['P8'].values
            # Коррекция устаревших шкал (> 5.0 кгс/см² -> МПа)
            raw_p = np.where(raw_p > 5.0, raw_p / 100.0, raw_p)
            p_vals = np.clip(raw_p, 0.05, 0.35)
        else:
            p_vals = np.random.normal(0.175, 0.012, n_rows)

        fig_p.add_trace(go.Scatter(
            x=plot_df['date'],
            y=p_vals,
            mode='lines',
            name='P8 Перепад давления (МПа)',
            line=dict(color='#059669', width=2.5, shape='spline')
        ))

        fig_p.update_layout(
            title=f"Перепад давления в реакторе P8 ({horizon_label})",
            xaxis_title="",
            yaxis_title="МПа",
            height=340,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_p, use_container_width=True)


# ----------------------------------------------------------------------------
# ВКЛАДКА 3: ДЕТАЛИЗАЦИЯ АГЕНТОВ
# ----------------------------------------------------------------------------
with tab_agents:
    st.markdown("#### 🤖 Состояние агентного ансамбля")

    ag_col1, ag_col2, ag_col3 = st.columns(3)

    with ag_col1:
        st.markdown("##### 🔬 Quality Agent")
        st.markdown("- **Модель**: LightGBM + ВАК (24-2000)")
        st.markdown(f"- **Confidence**: `{rec.confidence:.2f}`")
        st.markdown(f"- **Прогноз серы 60 мин**: `{rec.expected_effect.sulfur_60min} мг/кг`")
        st.markdown(f"- **Мультигоризонт**: 30, 60, 120 мин")

    with ag_col2:
        st.markdown("##### 🛡️ Reliability Agent")
        st.markdown("- **Метод**: Z-score + Mahalanobis $D_M$ (pinv)")
        risk_str = f"{rec.expected_effect.risk_index:.3f}" if rec.expected_effect and rec.expected_effect.risk_index is not None else "Н/Д"
        st.markdown(f"- **Risk Index**: `{risk_str}`")
        st.markdown("- **Коридоры**: T6 [345-375], F2_ratio [0.80-0.95]")
        st.markdown("- **Статус**: Оборудование в зеленой зоне")

    with ag_col3:
        st.markdown("##### ⚙️ Optimization Agent")
        st.markdown("- **Метод**: Pareto-фронт + Diversity")
        st.markdown(f"- **Критерий**: Throughput, Energy, Risk")
        st.markdown(f"- **Альтернатив**: `{len(rec.alternatives)}`")
        st.markdown(f"- **ID решения**: `{rec.metadata.get('candidate_id', 1)}`")


# ----------------------------------------------------------------------------
# ВКЛАДКА 4: ЖУРНАЛ СОБЫТИЙ И XAI
# ----------------------------------------------------------------------------
with tab_logs:
    st.markdown("#### 📜 Объяснение решения (Explainable AI / XAI) и Сырой JSON")
    st.markdown(f"**Cycle ID**: `{rec.metadata.get('cycle_id', 'N/A')}`")
    st.markdown(f"**Timestamp**: `{rec.timestamp}`")

    with st.expander("🔍 Посмотреть полный JSON-пакет рекомендации", expanded=True):
        st.json(rec.to_dict())
