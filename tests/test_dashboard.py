"""
Тесты для UI-дашборда оператора (INT-01).
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.orchestrator.orchestrator import Orchestrator


def test_dashboard_orchestrator_integration():
    """Проверка доступности данных и циклов для UI."""
    orchestrator = Orchestrator()

    for sc in ['normal', 'risk', 'missing', 'no_solution']:
        telem, qual = orchestrator._load_scenario_data(sc)
        assert isinstance(telem, pd.DataFrame)
        assert isinstance(qual, pd.DataFrame)
        assert not qual.empty
        assert 'tag' in qual.columns
        assert 'value' in qual.columns


def test_dashboard_imports():
    """Проверка доступности всех библиотек UI."""
    import streamlit
    import plotly.graph_objects as go
    import plotly.express as px
    from src.ui import dashboard

    assert streamlit is not None
    assert go is not None
    assert px is not None


def test_dashboard_int02_recommendation_contract():
    """Проверка контракта INT-02 для отображения рекомендации оператору."""
    from src.orchestrator.recommendation import (
        Recommendation, ActionItem, ExpectedEffect, ConstraintCheck, Alternative, ActionList
    )

    # 1. Тест статуса RECOMMENDED
    rec = Recommendation(
        recommendation_id="rec_test_01",
        timestamp="2026-09-20T21:00:00",
        state={"T6": 360.0, "Sulfur": 9.2},
        problem_type="RISK_SPEC_VIOLATION",
        action=[
            ActionItem(
                tag="T6", name="Температура реактора", from_value=360.0, to_value=363.0,
                unit="°C", delta=3.0, delta_percent=0.83
            )
        ],
        expected_effect=ExpectedEffect(
            sulfur_60min=7.5, sulfur_delta=-1.7, throughput=262.5, throughput_delta=2.5
        ),
        constraints_checked=[
            ConstraintCheck(
                constraint_id="C001", constraint="Сера ≤ 10 мг/кг", predicted_value=7.5,
                threshold=10.0, status="PASS", margin=2.5
            )
        ],
        confidence=0.85,
        status="RECOMMENDED",
        alternatives=[
            Alternative(
                id=1, action={"T6": 362.0}, score=0.45, throughput=261.0,
                energy_proxy=0.46, risk_index=0.18, delta_score=-0.01, delta_throughput=-1.5
            )
        ],
        explanation="Повышение T6 на 3°C снизит серу до 7.5 мг/кг."
    )

    # Проверка формата INT-02:
    assert rec.status == "RECOMMENDED"
    assert rec.explanation == "Повышение T6 на 3°C снизит серу до 7.5 мг/кг."

    # Действие: items()
    action_dict = dict(rec.action.items())
    assert "T6" in action_dict
    assert action_dict["T6"]["from"] == 360.0
    assert action_dict["T6"]["to"] == 363.0
    assert action_dict["T6"]["unit"] == "°C"

    # Ожидаемый эффект
    assert rec.expected_effect["Sulfur_60min"] == 7.5
    assert rec.expected_effect["throughput_change"] == 2.5

    # Альтернативы
    assert len(rec.alternatives) == 1
    alt_str = str(rec.alternatives[0])
    assert "T6: 362.0" in alt_str

    # Проверенные ограничения
    assert len(rec.constraints_checked) == 1
    assert rec.constraints_checked[0]["constraint"] == "Сера ≤ 10 мг/кг"
    assert rec.constraints_checked[0]["status"] == "PASS"

    # 2. Тест статуса NO_RECOMMENDATION
    no_rec = Recommendation(
        recommendation_id="rec_test_no",
        timestamp="2026-09-20T21:00:00",
        state={},
        problem_type="NO_DATA",
        action={},
        expected_effect=ExpectedEffect(),
        constraints_checked=[],
        confidence=0.0,
        status="NO_RECOMMENDATION",
        alternatives=[],
        explanation="Данные поточного анализатора отсутствуют."
    )
    assert no_rec.status == "NO_RECOMMENDATION"
    assert "отсутствуют" in no_rec.explanation


def test_dashboard_int02_orchestrator_live_scenarios():
    """Проверка генерации рекомендаций оркестратора для UI-сценариев risk и missing."""
    import asyncio
    orchestrator = Orchestrator()

    # Сценарий risk -> статус RECOMMENDED
    rec_risk = asyncio.run(orchestrator.run_cycle(scenario="risk"))
    assert rec_risk.status == "RECOMMENDED"
    assert hasattr(rec_risk.action, "items")
    assert rec_risk.expected_effect["Sulfur_60min"] is not None
    assert rec_risk.expected_effect["throughput_change"] is not None
    assert len(rec_risk.constraints_checked) > 0
    assert rec_risk.explanation

    # Сценарий missing -> статус NO_RECOMMENDATION
    rec_missing = asyncio.run(orchestrator.run_cycle(scenario="missing"))
    assert rec_missing.status == "NO_RECOMMENDATION"
    assert rec_missing.explanation


def test_dashboard_parse_explanation_blocks():
    """Проверка структурирования объяснений рекомендации для оператора."""
    from src.ui.dashboard import parse_explanation_blocks

    # 1. Legacy текст с "Рекомендовано:"
    old_text = (
        "Снижение T6 на 3.3°C (до 356.7°C) изменит серу с 6.8 до 7.5 мг/кг. "
        "Рекомендовано: Температура реактора: 360.0 → 356.7 °C; "
        "Производительность: 278.4 м³/ч (Δ=+63.4 м³/ч). "
        "Индекс риска оборудования: 0.17 (low). "
        "Выбран вариант как компромисс между качеством, производительностью и ресурсом оборудования."
    )
    blocks_old = parse_explanation_blocks(old_text)
    assert len(blocks_old) >= 3
    assert any("режим и качество" in b[0].lower() for b in blocks_old)
    assert any("356.7°C" in b[1] for b in blocks_old)

    # 2. Современный текст оркестратора
    new_text = (
        "Снижение температуры T6 на 3.3°C (до 356.7°C) обеспечивает прогноз серы 7.5 мг/кг. "
        "Решение обеспечивает рост выработки на +62.7 м³/ч при безопасном риске оборудования (0.17, low). "
        "Выбран оптимальный компромисс между качеством, производительностью и ресурсом катализатора."
    )
    blocks_new = parse_explanation_blocks(new_text)
    assert len(blocks_new) == 3
    assert "режим и качество" in blocks_new[0][0].lower()
    assert "эффект и надежность" in blocks_new[1][0].lower()


def test_dashboard_is_dark_theme():
    """Проверка работы вспомогательной функции темы."""
    from src.ui.dashboard import is_dark_theme
    # Без контекста браузера должна возвращать False без исключений
    assert is_dark_theme() is False



