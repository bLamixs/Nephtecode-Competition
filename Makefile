.PHONY: help run run-risk run-missing run-no-solution test test-agents test-scenarios test-no-leak dashboard db-init load-data README zip pack

# Определение интерпретатора Python (предпочтительно venv, если есть)
PYTHON ?= $(shell if [ -f "./venv/bin/python" ]; then echo "./venv/bin/python"; else echo "python3"; fi)
PYTEST ?= $(shell if [ -f "./venv/bin/pytest" ]; then echo "./venv/bin/pytest"; else echo "pytest"; fi)
STREAMLIT ?= $(shell if [ -f "./venv/bin/streamlit" ]; then echo "./venv/bin/streamlit"; else echo "streamlit"; fi)

help:
	@echo "Команды проекта НефтеКод 2.0:"
	@echo "  make run              - Запуск сценария normal (Штатный режим)"
	@echo "  make run-risk         - Запуск сценария risk (Риск превышения серы)"
	@echo "  make run-missing      - Запуск сценария missing (Сбой датчиков / устаревшие данные)"
	@echo "  make run-no-solution  - Запуск сценария no-solution (Технологический тупик)"
	@echo "  make test             - Запуск тестов с отчётом о покрытии кода (HTML & term)"
	@echo "  make test-agents      - Запуск тестов интерфейсов агентов (Tests-02)"
	@echo "  make test-scenarios   - Запуск тестов 4 технологических сценариев (Tests-03)"
	@echo "  make test-no-leak     - Запуск тестов отсутствия утечки данных и shuffle (Tests-04)"
	@echo "  make dashboard        - Запуск веб-дашборда оператора"
	@echo "  make db-init          - Инициализация схемы базы данных"
	@echo "  make load-data        - Первичная загрузка и конвертация данных в Parquet"
	@echo "  make zip              - Сборка полного zip-архива с сырыми данными data/raw (~135 МБ)"
	@echo "  make zip-light        - Сборка легковесного zip-архива без больших CSV (~8.5 МБ)"
	@echo "  make README           - Вывод ссылки на README"

README:
	@echo "См. README.md"

db-init:
	$(PYTHON) -c "import sqlite3; con = sqlite3.connect('output/recommendations.db'); con.execute('CREATE TABLE IF NOT EXISTS recommendations (recommendation_id TEXT PRIMARY KEY, cycle_id TEXT, timestamp TEXT, status TEXT, problem_type TEXT, confidence REAL, explanation TEXT, payload_json TEXT)'); con.close(); print('База данных успешно инициализирована.')"

load-data:
	$(PYTHON) scripts/load_raw_data.py --avt data/raw/avt_tags.csv --hydro data/raw/242000_tags.csv --output data/processed/

run:
	$(PYTHON) src/main.py --scenario normal

run-risk:
	$(PYTHON) src/main.py --scenario risk

run-missing:
	$(PYTHON) src/main.py --scenario missing

run-no-solution:
	$(PYTHON) src/main.py --scenario no_solution

test:
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term

test-agents:
	$(PYTEST) tests/test_agents.py -v

test-scenarios:
	$(PYTEST) tests/test_scenarios.py -v

test-no-leak:
	$(PYTEST) tests/test_no_leak.py -v

dashboard:
	$(STREAMLIT) run src/ui/dashboard.py

zip: zip-full

zip-full:
	@echo "📦 Сборка полного zip-архива с сырыми данными data/raw (~135 МБ)..."
	@rm -f neftecode_solution.zip
	@zip -q -r neftecode_solution.zip . \
		-x "venv/*" \
		-x "logs/*.log" -x "logs/*.json" -x "logs/*.jsonl" \
		-x "htmlcov/*" \
		-x ".git/*" \
		-x ".pytest_cache/*" \
		-x "*/__pycache__/*" \
		-x "*.pyc" \
		-x ".coverage" \
		-x ".DS_Store" -x "*/.DS_Store" \
		-x "data/raw/*.rar" \
		-x "data/processed/*.parquet" -x "data/processed/*.csv" \
		-x "output/recommendations/*" -x "output/*.db" -x "*.zip"
	@echo "✅ Полный архив с сырыми данными создан: neftecode_solution.zip"
	@du -sh neftecode_solution.zip

zip-light:
	@echo "📦 Сборка легковесного zip-архива без больших CSV (~8.5 МБ)..."
	@rm -f neftecode_solution_light.zip
	@zip -q -r neftecode_solution_light.zip . \
		-x "venv/*" \
		-x "logs/*.log" -x "logs/*.json" -x "logs/*.jsonl" \
		-x "htmlcov/*" \
		-x ".git/*" \
		-x ".pytest_cache/*" \
		-x "*/__pycache__/*" \
		-x "*.pyc" \
		-x ".coverage" \
		-x ".DS_Store" -x "*/.DS_Store" \
		-x "data/raw/avt_tags.csv" -x "data/raw/242000_tags.csv" -x "data/raw/*.rar" \
		-x "data/processed/*.parquet" -x "data/processed/*.csv" \
		-x "output/recommendations/*" -x "output/*.db" -x "*.zip"
	@echo "✅ Легковесный архив создан: neftecode_solution_light.zip"
	@du -sh neftecode_solution_light.zip
