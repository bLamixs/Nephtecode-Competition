"""
Модуль: src/utils/logging_config.py
Назначение: Настройка структурированного логирования для воспроизводимости и аудита решений МАС.

Контекст задачи:
В соответствии с требованиями ТЗ «НефтеКод», мультиагентная система обязана сохранять
и протоколировать:
1. Входное состояние процесса (телеметрия, возраст анализов ЛИМС/ПАК).
2. Оценки всех агентов (качество, надежность, альтернативы оптимизатора).
3. Принятые решения или причины отказа от рекомендации («Надёжной рекомендации нет»).

Данный модуль обеспечивает единую точку конфигурации логгеров, выводящих
сообщения одновременно в стандартный поток вывода (консоль оператора) и в файл журнала logs/system.log.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "neftecode_mas", log_dir: str = "logs") -> logging.Logger:
    """
    Создает и настраивает потокобезопасный логгер с форматированием даты и уровней логирования.

    Параметры:
        name (str): Имя логгера для трассировки компонента (например, 'QualityAgent', 'Orchestrator').
        log_dir (str): Директория для сохранения файлов журнала (по умолчанию 'logs').

    Возвращает:
        logging.Logger: Сконфигурированный экземпляр стандартного логгера Python.
    """
    # Гарантируем существование директории для логов
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Предотвращаем дублирование обработчиков при повторных вызовах функции
    if not logger.handlers:
        # 1. Потоковый обработчик (вывод в консоль/терминал)
        console_handler = logging.StreamHandler(sys.stdout)
        log_format = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        # 2. Файловый обработчик (персистентный журнал для аудита и воспроизводимости)
        file_handler = logging.FileHandler(f"{log_dir}/system.log", encoding="utf-8")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
