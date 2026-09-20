"""
Тесты для главного пайплайна и CLI (INT-05).
"""

import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch

from src.main import parse_args, main
from src.orchestrator.recommendation import Recommendation


def test_main_cli_argparse():
    """Проверка аргументов CLI."""
    with patch("sys.argv", ["main.py", "--scenario", "risk"]):
        args = parse_args()
        assert args.scenario == "risk"

    with patch("sys.argv", ["main.py"]):
        args = parse_args()
        assert args.scenario == "normal"


@pytest.mark.anyio
async def test_main_pipeline_normal_scenario():
    """Проверка выполнения главного пайплайна для сценария normal."""
    rec = await main(scenario="normal")
    assert isinstance(rec, Recommendation)
    assert rec.status == "RECOMMENDED"
    assert rec.problem_type in ["RISK_SPEC_VIOLATION", "SUBOPTIMAL"]


@pytest.mark.anyio
async def test_main_pipeline_all_scenarios():
    """Проверка выполнения главного пайплайна для всех сценариев (missing, no_solution)."""
    rec_missing = await main(scenario="missing")
    assert isinstance(rec_missing, Recommendation)
    assert rec_missing.status == "NO_RECOMMENDATION"
    assert rec_missing.problem_type == "NO_DATA"

    rec_no_sol = await main(scenario="no_solution")
    assert isinstance(rec_no_sol, Recommendation)
    assert rec_no_sol.status == "NO_RECOMMENDATION"
    assert rec_no_sol.problem_type == "NO_SOLUTION"


def test_main_database_and_json_persistence():
    """Проверка сохранения результатов в SQLite БД и JSON файлы."""
    db_path = Path("output/recommendations.db")
    assert db_path.exists(), "Файл БД recommendations.db не найден"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM recommendations")
        count = cursor.fetchone()[0]
        assert count > 0, "В таблице recommendations нет записей"

    rec_dir = Path("output/recommendations")
    assert rec_dir.exists() and any(rec_dir.glob("*.json")), "JSON-файлы рекомендаций не найдены"
