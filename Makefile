.PHONY: help run format lint clean

help:
	@echo "Portfolio Rebalancer - Available Commands"
	@echo "=========================================="
	@echo "  make run      - Run Streamlit app (uv run streamlit run main.py)"
	@echo "  make format   - Format code with ruff"
	@echo "  make lint     - Lint code with ruff"
	@echo "  make clean    - Remove cache and build files"

run:
	uv run streamlit run main.py

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned up cache files"
