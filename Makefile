# Callback task runner. Run every target from the repo root. uv resolves the
# pinned 3.11 interpreter itself, so nothing here needs an activated venv.
#
#   make            show this help
#   make install    create .venv with dev and deploy dependencies
#   make test       run the suite
#   make lint       ruff check, plus a formatting check that rewrites nothing
#   make fmt        ruff format, plus safe lint fixes
#   make run        serve the detector locally on :8000
#   make snapshot   rewrite the frozen contract schema snapshot
#   make preflight  check a Modal deploy would work, without deploying
#
# Requires GNU make and uv on PATH.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

UV   ?= uv
HOST ?= 127.0.0.1
PORT ?= 8000

# ruff is not declared in pyproject.toml yet, so it comes from a pinned uv
# overlay. The pin keeps local results identical to CI. Once
# `ruff>=0.17.3,<0.18` is added to the dev extra, simplify this to
# `RUFF ?= $(UV) run ruff`, which is what the CI workflow already uses.
RUFF ?= $(UV) run --with "ruff==0.17.3" ruff

.DEFAULT_GOAL := help
.PHONY: help install test lint fmt run snapshot preflight check

help: ## Show this help
	@echo "Callback detector. Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  %-10s %s\n", $$1, $$2}'

install: ## Create .venv with the dev and deploy dependencies
	$(UV) sync --all-extras

test: ## Run the full test suite, then the callback_shared contract tests
	$(UV) run pytest
	$(UV) run pytest shared/tests -q

lint: ## Lint with ruff and verify formatting, rewriting nothing
	$(RUFF) check .
	$(RUFF) format --check .

fmt: ## Format with ruff and apply its safe lint fixes
	$(RUFF) format .
	$(RUFF) check . --fix

# --no-access-log: an access log line carries the query string, and
# GET /v1/users?name= puts a caller's name there. The observability layer
# logs route templates instead.
run: ## Serve the detector locally with reload (HOST/PORT override the defaults)
	$(UV) run uvicorn detector.app:app --reload --no-access-log --host $(HOST) --port $(PORT)

snapshot: ## Rewrite the frozen contract schema snapshot
	CALLBACK_UPDATE_SNAPSHOT=1 $(UV) run pytest tests/contract/test_schema_snapshot.py
	@echo
	@echo "Snapshot rewritten. If the contract shape changed, bump"
	@echo "CONTRACT_VERSION in shared/contract.py and commit it together with"
	@echo "schemas/contract.schema.json."

preflight: ## Check a Modal deploy would succeed. Deploys nothing, spends nothing
	$(UV) run python scripts/modal_preflight.py

check: lint test ## Everything CI runs: lint, formatting, suite, Modal preflight
	$(UV) run python scripts/modal_preflight.py --allow-unauthenticated
