"""
Настройка структурированного логирования для воспроизводимости и аудита решений МАС.
"""
import logging
import sys
from pathlib import Path

def setup_logger(name: str = "neftecode_mas", log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Консольный вывод
        c_handler = logging.StreamHandler(sys.stdout)
        c_format = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        # Файловый вывод
        f_handler = logging.FileHandler(f"{log_dir}/system.log", encoding="utf-8")
        f_handler.setFormatter(c_format)
        logger.addHandler(f_handler)

    return logger
