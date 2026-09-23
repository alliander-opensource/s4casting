.PHONY: install lint check test lock-local lock-reset

help:	   ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

install:   ## Install packages
	uv sync

check:     ## Check formatting and tests scripts
	uv lock --check
	uv run ruff format --check src tests scripts notebooks
	uv run ruff check src tests scripts notebooks
	uv run deptry src tests scripts
	uv run ty check src tests scripts
	uv run pytest tests -s

lint:      ## Check formatting
	uv run ruff format src tests scripts notebooks
	uv run ruff check --fix src tests scripts notebooks

lock-local: ## Build a local-only lockfile via your default package index (hidden from git)
	git update-index --skip-worktree uv.lock
	uv lock --upgrade

lock-reset: ## Undo lock-local; run before pull/checkout when uv.lock changed upstream
	git update-index --no-skip-worktree uv.lock
	git checkout -- uv.lock

license:
	find . -type f -name '*.py' \
	    -not -path './.venv/*' \
	    -not -path './src/s4casting/model/mamba.py' \
	    -not -path './src/s4casting/model/mambacpu.py' \
	    -not -path './src/s4casting/model/_s4_kernel.py' \
	    -not -path './src/s4casting/model/_selective_scan_interface.py' \
	    -print0 \
	  | xargs -0 reuse annotate --style python \
	      -c "Contributors to the s4casting project" -l MPL-2.0 --exclude-year
	find ./docs -type f -name '*.md' -print0 \
	  | xargs -0 reuse annotate --style html \
	      -c "Contributors to the s4casting project" -l MPL-2.0 --exclude-year
	find . -maxdepth 1 -type f -name '*.md' -print0 \
	  | xargs -0 reuse annotate --style html \
	      -c "Contributors to the s4casting project" -l MPL-2.0 --exclude-year

test:      ## Run test scripts
	uv run pytest tests -x --pdb
