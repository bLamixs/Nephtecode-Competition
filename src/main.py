#!/usr/bin/env python3
"""
Главный пайплайн: запуск сценария через CLI (INT-05).

Usage:
    python src/main.py --scenario normal|risk|missing|no_solution
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.orchestrator import Orchestrator
from src.utils.logging_config import setup_logger

logger = setup_logger("main")


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="НефтеКод 2.0: Запуск мультиагентного цикла оптимизации (INT-05)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=["normal", "risk", "missing", "no_solution"],
        help="Технологический сценарий для запуска (normal, risk, missing, no_solution)"
    )
    return parser.parse_args()


async def main(scenario: str):
    """
    Запуск полного цикла оптимизации для выбранного сценария.

    Args:
        scenario: имя сценария ('normal', 'risk', 'missing', 'no_solution')

    Returns:
        Recommendation: объект рекомендации оркестратора
    """
    logger.info(f"Запуск сценария: {scenario}")
    print(f"\n================================================================================")
    print(f"🚀 Запуск пайплайна НефтеКод 2.0 | Сценарий: {scenario}")
    print(f"================================================================================\n")

    orchestrator = Orchestrator()
    recommendation = await orchestrator.run_cycle(scenario=scenario)

    logger.info(f"Рекомендация: {recommendation.status}")
    logger.info(f"Объяснение: {recommendation.explanation}")

    # Вывод результатов в консоль оператора
    print(f"📋 Результат цикла:")
    print(f"  • ID рекомендации: {recommendation.recommendation_id}")
    print(f"  • Статус: {recommendation.status}")
    print(f"  • Тип проблемы: {recommendation.problem_type}")
    print(f"  • Уверенность: {recommendation.confidence:.2f}")
    print(f"  • Объяснение: {recommendation.explanation}\n")

    if recommendation.status == "RECOMMENDED":
        print("✅ Управляющие воздействия:")
        for act in recommendation.action:
            tag = getattr(act, 'tag', '')
            name = getattr(act, 'name', tag)
            from_v = getattr(act, 'from_value', 0.0)
            to_v = getattr(act, 'to_value', 0.0)
            unit = getattr(act, 'unit', '')
            delta = getattr(act, 'delta', to_v - from_v)
            print(f"   - {tag} ({name}): {from_v:.1f} → {to_v:.1f} {unit} (Δ = {delta:+.1f})")

        ee = recommendation.expected_effect
        if ee and ee.sulfur_60min is not None:
            print(f"\n📈 Ожидаемый эффект (+60 мин):")
            print(f"   - Сера: {ee.sulfur_60min:.2f} мг/кг (Δ = {ee.sulfur_delta:+.2f} мг/кг)")
            if ee.throughput is not None:
                print(f"   - Выпуск F9: {ee.throughput:.1f} м³/ч (Δ = {ee.throughput_delta:+.1f} м³/ч)")
            if ee.risk_index is not None:
                print(f"   - Индекс риска: {ee.risk_index:.3f}")
    else:
        print(f"⛔ Рекомендация не выдана: система переведена в безопасный режим.")

    print(f"\n================================================================================\n")
    return recommendation


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.scenario))
