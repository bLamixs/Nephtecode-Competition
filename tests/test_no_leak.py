"""
Модуль проверки отсутствия утечки данных из будущего и отсутствия перемешивания (Блок: Tests-04).

Проверяет:
- Временной сплит (Holdout): данные обучающей выборки строго предшествуют валидационной (2023–2025 vs 2026)
- Запрет случайного перемешивания временных рядов (no shuffle, no sample(frac=1)).
"""

import inspect
import pytest
from src.agents.quality_agent import QualityAgent


def test_no_future_leak():
    """
    Проверка отсутствия утечки из будущего (Holdout):
    Все даты в train_data должны быть строго меньше минимальной даты в test_data.
    """
    agent = QualityAgent()
    train_data = agent.train_data
    test_data = agent.test_data

    assert train_data['date'].max() < test_data['date'].min(), (
        f"Утечка данных! Train max ({train_data['date'].max()}) >= Test min ({test_data['date'].min()})"
    )
    assert train_data['date'].min() < test_data['date'].max()


def test_no_shuffle():
    """
    Проверка кода обучения моделей:
    Запрещено использование shuffle и sample(frac=1), нарушающих причинно-следственные связи.
    """
    source = inspect.getsource(QualityAgent.train)
    assert 'shuffle' not in source.lower(), "Обнаружено использование shuffle в QualityAgent.train!"
    assert 'sample(frac=1)' not in source, "Обнаружено использование df.sample(frac=1) в QualityAgent.train!"
