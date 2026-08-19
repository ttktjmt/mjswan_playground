.PHONY: sync
sync:
	uv sync --extra wbc

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: test
test:
	uv run pytest tests/ -v

.PHONY: check
check: format test
