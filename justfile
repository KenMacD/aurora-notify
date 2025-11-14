
list:
	@just --list

run *ARGS:
	uv run aurora {{ARGS}}

check:
	uv run ruff check

format:
	uv run ruff format

test:
	uv run pytest
