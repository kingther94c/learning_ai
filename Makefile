CONDA ?= conda
CONDA_ENV ?= py313
PY ?= $(CONDA) run --no-capture-output -n $(CONDA_ENV) python
PORT ?= 8000

.DEFAULT_GOAL := help

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- setup

setup:  ## install into the native arm64 Conda py313 env and register its kernel
	@command -v $(CONDA) >/dev/null 2>&1 || { \
		echo "conda is required; see README.md#quick-start"; \
		exit 1; \
	}
	@$(PY) -c "import platform, sys; \
		assert sys.version_info[:2] == (3, 13), f'expected Python 3.13, got {platform.python_version()}'; \
		assert sys.platform != 'darwin' or platform.machine() == 'arm64', \
		f'expected native arm64 Python on macOS, got {platform.machine()}'"
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[notebooks,dev]"
	$(PY) -m ipykernel install --sys-prefix --name learnai-py313 --display-name "Python 3.13 (learnai)"
	@echo
	@$(PY) -m learnai.cli doctor
	@echo "\nready.  'make lab' to open the notebooks, 'make serve' for the interactive pages."

# --------------------------------------------------------------------------- run

lab:  ## open JupyterLab on the notebooks
	$(PY) -m jupyterlab notebooks/

serve:  ## serve interactive/ over http (only needed for cross-page links; file:// works too)
	@echo "http://localhost:$(PORT)/"
	@cd interactive && $(PY) -m http.server $(PORT)

data:  ## retrain the reference GPT and regenerate the interactive pages' data
	$(PY) tools/build_data.py

lesson:  ## scaffold a new lesson:  make lesson NAME=03_nlp/02_embeddings TITLE="Word vectors"
	$(PY) tools/new_lesson.py "$(NAME)" --title "$(TITLE)"

doctor:  ## diagnose Python, dependencies, device, and Jupyter kernel
	$(PY) -m learnai.cli doctor

# --------------------------------------------------------------------------- check

test:  ## run the python + javascript test suites
	$(PY) -m pytest tests/ -q
	node tests/test_interactive_js.mjs

lessons:  ## check notebook hygiene, teaching structure, offline pages, and navigation
	$(PY) tools/check_lessons.py

notebooks:  ## execute every notebook top to bottom and report failures (slow)
	$(PY) tools/nbuild.py --run

notebooks-fast:  ## execute every notebook with tiny training runs (CI/smoke test)
	LEARNAI_FAST=1 LEARNAI_OFFLINE=1 $(PY) tools/nbuild.py --run --timeout 300

lint:  ## ruff check + format check
	$(PY) -m ruff check learnai tools tests
	$(PY) -m ruff format --check learnai tools tests

fix:  ## apply ruff autofixes and formatting
	$(PY) -m ruff check --fix learnai tools tests
	$(PY) -m ruff format learnai tools tests

check: test lint lessons  ## the pre-commit gate

clean:  ## remove caches and downloaded data (all regenerable)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ .ipynb_checkpoints
	rm -rf data/cache runs
	find . -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} +

.PHONY: help setup lab serve data lesson doctor test lessons notebooks notebooks-fast lint fix check clean
