.DEFAULT_GOAL := help

SRC_DIR := src
TEST_DIR := tests

.PHONY: help
help:
	@echo "Please use 'make <target>' where target is one of:"
	@echo ""
	@echo "  static      - Check formatting, lint, and types (no changes; CI)"
	@echo "  static-fix  - Format, lint-fix, and type-check"
	@echo "  test        - Run unit tests with pytest and generate coverage report"
	@echo "  clean       - Remove cached files and temporary files"
	@echo ""

.PHONY: static
static:
	uv run ruff format --check $(SRC_DIR) $(TEST_DIR)
	uv run ruff check $(SRC_DIR) $(TEST_DIR)
	uv run mypy $(SRC_DIR) $(TEST_DIR)

.PHONY: static-fix
static-fix:
	uv run ruff format $(SRC_DIR) $(TEST_DIR)
	uv run ruff check --fix $(SRC_DIR) $(TEST_DIR)
	uv run mypy $(SRC_DIR) $(TEST_DIR)

.PHONY: test
test:
	uv run pytest -n auto --cov=$(SRC_DIR) --cov-report=term-missing $(TEST_DIR)

.PHONY: clean
clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache .tox .coverage coverage.xml
	find $(SRC_DIR) $(TEST_DIR) -type d -name "__pycache__" -exec rm -rf {} +
