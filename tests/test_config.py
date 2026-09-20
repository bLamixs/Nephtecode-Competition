"""
Модуль: tests/test_config.py
Назначение: Тестирование загрузки конфигураций и переменных окружения (INT-04).
"""

import os
from pathlib import Path
import pytest
from src.config import CONFIG, PROJECT_ROOT, yaml, load_dotenv


def test_config_yaml_loading():
    """Проверка структуры и значений config.yaml по ТЗ INT-04."""
    assert isinstance(CONFIG, dict)
    assert "optimizer" in CONFIG
    assert "quality" in CONFIG
    assert "reliability" in CONFIG

    # 1. Секция optimizer
    opt = CONFIG["optimizer"]
    assert "controlled_params" in opt
    assert "T6" in opt["controlled_params"]
    t6 = opt["controlled_params"]["T6"]
    assert t6["current"] in (295.0, 360.0)
    assert t6["min"] in (290.0, 345.0)
    assert t6["unit"] == "°C"

    assert "F2_F26_ratio" in opt["controlled_params"]
    f2_ratio = opt["controlled_params"]["F2_F26_ratio"]
    assert f2_ratio["current"] == 0.85
    assert f2_ratio["min"] == 0.80
    assert f2_ratio["max"] == 0.95
    assert f2_ratio["unit"] == "-"

    assert "weights" in opt
    weights = opt["weights"]
    assert weights["throughput"] == 0.5
    assert weights["energy"] == 0.3
    assert weights["risk"] == 0.2

    # 2. Секция quality
    qual = CONFIG["quality"]
    assert qual["risk_threshold"] == 0.1
    assert qual["max_age_min"] == 120

    # 3. Секция reliability
    rel = CONFIG["reliability"]
    assert "risk_class_thresholds" in rel
    r_thresh = rel["risk_class_thresholds"]
    assert r_thresh["low"] == 0.3
    assert r_thresh["medium"] == 0.6
    assert r_thresh["high"] == 1.0


def test_env_example_and_gitignore():
    """Проверка наличия .env.example и добавления .env в .gitignore."""
    env_example = PROJECT_ROOT / ".env.example"
    assert env_example.exists(), ".env.example должен существовать в корне проекта"

    content = env_example.read_text(encoding="utf-8")
    assert "DATABASE_URL" in content
    assert "LOG_LEVEL" in content
    assert "DEFAULT_SCENARIO" in content
    assert "USE_VAC_BASELINE" in content
    assert "USE_LGBM_CORRECTION" in content

    # Проверка .gitignore
    gitignore = PROJECT_ROOT / ".gitignore"
    assert gitignore.exists()
    gi_content = gitignore.read_text(encoding="utf-8")
    assert ".env" in gi_content
