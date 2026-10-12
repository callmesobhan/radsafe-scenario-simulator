.PHONY: install test lint format check examples clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

check: lint test

examples:
	radsafe run scenarios/cs137_bench_source.yaml
	radsafe run scenarios/co60_irradiator_corridor.yaml
	python examples/thickness_sweep.py

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
