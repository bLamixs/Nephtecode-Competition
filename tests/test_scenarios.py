"""
Модуль тестирования сквозных сценариев оптимизации (Блок: Tests-03).

Проверяет 4 технологических сценария:
1. normal: штатный режим, нет риска нарушения
2. risk: повышенный риск серы, формируется рекомендация (T6/Sulfur), confidence > 0.5
3. missing: отказ от рекомендации из-за нехватки/устаревания данных (NO_DATA)
4. no_solution: отказ из-за отсутствия допустимого решения (технологический тупик).
"""

import pytest
from src.main import main


@pytest.mark.asyncio
async def test_scenario_normal():
    """
    Сценарий 'normal':
    - Статус NO_RECOMMENDATION (или RECOMMENDED, если найдена экономическая оптимизация)
    - Тип проблемы не содержит 'risk'
    """
    rec = await main('normal')
    assert rec.status in ('NO_RECOMMENDATION', 'RECOMMENDED')
    assert 'risk' not in rec.problem_type.lower()


@pytest.mark.asyncio
async def test_scenario_risk():
    """
    Сценарий 'risk':
    - Рекомендация выдана: status == 'RECOMMENDED'
    - Управляющее воздействие направлено на серу / температуру T6
    - Уверенность системы confidence > 0.5
    """
    rec = await main('risk')
    assert rec.status == 'RECOMMENDED'
    assert 'Sulfur' in rec.action or 'T6' in rec.action
    assert rec.confidence > 0.5
    assert len(rec.action) > 0


@pytest.mark.asyncio
async def test_scenario_missing():
    """
    Сценарий 'missing':
    - Отказ от выдачи управляющих воздействий: status == 'NO_RECOMMENDATION'
    - В объяснении зафиксирована нехватка данных
    """
    rec = await main('missing')
    assert rec.status == 'NO_RECOMMENDATION'
    assert 'недостаточно данных' in rec.explanation.lower()


@pytest.mark.asyncio
async def test_scenario_no_solution():
    """
    Сценарий 'no_solution':
    - Отказ от выдачи управляющих воздействий: status == 'NO_RECOMMENDATION'
    - В объяснении зафиксировано отсутствие допустимого решения
    """
    rec = await main('no_solution')
    assert rec.status == 'NO_RECOMMENDATION'
    assert 'нет допустимого' in rec.explanation.lower()
