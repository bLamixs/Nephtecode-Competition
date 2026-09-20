"""
Модуль: tests/test_logging_config.py
Назначение: Тестирование настройки структурированного логирования (src/utils/logging_config.py).
"""

import pytest
import logging
from pathlib import Path
from src.utils.logging_config import setup_logger


def test_setup_logger(tmp_path):
    log_dir = str(tmp_path / "logs")
    logger = setup_logger("test_agent", log_dir=log_dir)

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_agent"
    assert len(logger.handlers) >= 2

    # Запись тестового сообщения
    test_msg = "Тестовое сообщение для проверки логгера"
    logger.info(test_msg)

    # Проверка создания файла логов
    log_file = Path(log_dir) / "system.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert test_msg in content

    # Проверка создания JSON логов по INT-03
    json_files = list(Path(log_dir).glob("*.json"))
    assert len(json_files) >= 1
    json_content = json_files[0].read_text(encoding="utf-8")
    assert test_msg in json_content


def test_int03_orchestrator_step_logging(tmp_path):
    """Проверка логирования шагов МАС в формате JSON по DoD INT-03."""
    import asyncio
    import json
    from src.orchestrator.orchestrator import Orchestrator

    log_dir = tmp_path / "logs"
    orchestrator = Orchestrator(log_dir=str(log_dir))

    # Запускаем цикл
    rec = asyncio.run(orchestrator.run_cycle(scenario="normal"))
    assert rec.status == "RECOMMENDED"

    # Проверяем JSON-файлы
    json_files = list(log_dir.glob("*.json"))
    assert len(json_files) >= 1, "Должен быть создан хотя бы один JSON-лог файл"

    all_lines = []
    for jf in json_files:
        lines = jf.read_text(encoding="utf-8").strip().split("\n")
        all_lines.extend(lines)

    # Находим все JSON-шаги
    logged_steps = set()
    for line in all_lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if isinstance(record, dict) and "step" in record:
                logged_steps.add(record["step"])
                assert "timestamp" in record
                assert "data" in record
        except json.JSONDecodeError:
            pass

    # Проверяем обязательные шаги из DoD INT-03:
    assert "input_validation" in logged_steps
    assert "quality_assessment" in logged_steps
    assert "reliability_assessment" in logged_steps
    assert "optimization" in logged_steps
    assert "recommendation" in logged_steps
