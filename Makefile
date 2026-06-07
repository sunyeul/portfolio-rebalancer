.PHONY: help run frontend-dev dev build-frontend format lint clean

help:
	@echo "Portfolio Rebalancer - Available Commands"
	@echo "=========================================="
	@echo "  make run            - Run FastAPI API server"
	@echo "  make frontend-dev   - Run Vite frontend dev server"
	@echo "  make dev            - Run API and frontend dev servers"
	@echo "  make build-frontend - Build React frontend"
	@echo "  make format         - Format Python code with ruff"
	@echo "  make lint           - Lint Python code with ruff"
	@echo "  make clean          - Remove cache and build files"

run:
	uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

frontend-dev:
	cd frontend && bun run dev

dev:
	$(MAKE) run & $(MAKE) frontend-dev

build-frontend:
	cd frontend && bun run build

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned up cache files"
