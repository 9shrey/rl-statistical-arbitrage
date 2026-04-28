PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
CONFIG ?= configs/smoke.yaml

.PHONY: help setup install test test-cov lint format smoke data train backtest report clean

help:
	@echo "Targets:"
	@echo "  setup       - install package + dev extras (editable)"
	@echo "  test        - run unit + property tests"
	@echo "  test-cov    - run tests with coverage"
	@echo "  lint        - ruff check + black --check"
	@echo "  format      - ruff fix + black"
	@echo "  smoke       - end-to-end smoke run on fixtures"
	@echo "  train       - arb train (CONFIG=...)"
	@echo "  backtest    - arb backtest (CONFIG=...)"
	@echo "  report      - arb report (CONFIG=...)"
	@echo "  clean       - remove caches and artifacts"

setup:
	$(PIP) install -e ".[dev]"

install:
	$(PIP) install -e .

test:
	$(PYTEST)

test-cov:
	$(PYTEST) --cov=arb --cov-report=term-missing --cov-report=html

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m black --check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m black src tests

smoke:
	$(PYTHON) -m arb.cli smoke --config $(CONFIG)

train:
	$(PYTHON) -m arb.cli train --config $(CONFIG)

backtest:
	$(PYTHON) -m arb.cli backtest --config $(CONFIG)

report:
	$(PYTHON) -m arb.cli report --config $(CONFIG)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
