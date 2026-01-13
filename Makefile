.PHONY: help install dev test lint format check clean train-trm train-transformer run-quick

# Default target
help:
	@echo "Bench-TRM Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  install      Install dependencies"
	@echo "  dev          Install dev dependencies and pre-commit hooks"
	@echo ""
	@echo "Quality:"
	@echo "  lint         Run ruff linter"
	@echo "  format       Format code with ruff"
	@echo "  check        Run all checks (lint + type check)"
	@echo "  test         Run tests with pytest"
	@echo ""
	@echo "Training:"
	@echo "  run-quick    Quick test run (2 epochs, small data)"
	@echo "  train-trm    Train TRM model"
	@echo "  train-transformer  Train Transformer baseline"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        Remove build artifacts and caches"

# Setup
install:
	uv sync

dev: install
	uv sync --group dev
	uv run pre-commit install

# Quality
lint:
	uv run ruff check .

format:
	uv run ruff check --fix .
	uv run ruff format .

check: lint
	uv run ty check src/ main.py

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src --cov-report=html

# Training
run-quick:
	./scripts/run_experiments.sh quick

train-trm:
	./scripts/run_experiments.sh trm

train-transformer:
	./scripts/run_experiments.sh transformer

train-all:
	./scripts/run_experiments.sh all

# Utilities
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	rm -rf build/ dist/ *.egg-info/
	rm -rf htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
