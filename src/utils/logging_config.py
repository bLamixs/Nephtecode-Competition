"""
Модуль: src/utils/logging_config.py
Назначение: Настройка структурированного JSON и консольного логирования для аудита решений МАС (INT-03).
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "neftecode_mas", log_dir: str = "logs") -> logging.Logger:
    """
    Создает и настраивает логгер в соответствии с требованиями INT-03:
    1. File handler (JSON) -> logs/{datetime.now():%Y-%m-%d_%H-%M-%S}.json
    2. File handler (текстовый) -> logs/system.log
    3. Console handler -> stdout с форматированием '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    existing_files = [
        str(Path(getattr(h, 'baseFilename', '')).resolve())
        for h in logger.handlers
        if isinstance(h, logging.FileHandler)
    ]

    # 1. File handler (JSON) по спецификации INT-03
    json_path = Path(log_dir) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    if str(json_path.resolve()) not in existing_files:
        fh = logging.FileHandler(json_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        existing_files.append(str(json_path.resolve()))

    # 2. File handler (system.log) для постоянного аудита и совместимости с тестами
    sys_log = Path(log_dir) / "system.log"
    if str(sys_log.resolve()) not in existing_files:
        fh_sys = logging.FileHandler(sys_log, encoding="utf-8")
        fh_sys.setFormatter(logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(fh_sys)
        existing_files.append(str(sys_log.resolve()))

    # 3. Console handler
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)

    return logger
