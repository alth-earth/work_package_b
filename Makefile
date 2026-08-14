MAMBA_PREFIX := $(CURDIR)/.mamba-env
MAMBA_ROOT_PREFIX := $(CURDIR)/.mamba-root
UV := $(if $(wildcard $(MAMBA_PREFIX)/bin/uv),$(MAMBA_PREFIX)/bin/uv,uv)
export UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_PYTHON_INSTALL_DIR ?= $(CURDIR)/.uv-python
export UV_PYTHON_DOWNLOADS ?= never
MODEL_EXTRA_FLAG := $(if $(wildcard $(CURDIR)/.venv/lib/python3.13/site-packages/torch),--extra model-cpu,)

.PHONY: env-create env-update lock sync model-sync model-check test integration lint check clean

env-create:
	mamba env create --root-prefix $(MAMBA_ROOT_PREFIX) --prefix $(MAMBA_PREFIX) -f environment.yml --yes

env-update:
	mamba env update --root-prefix $(MAMBA_ROOT_PREFIX) --prefix $(MAMBA_PREFIX) -f environment.yml --prune --yes

lock:
	$(UV) lock --python "$(if $(wildcard $(MAMBA_PREFIX)/bin/python),$(MAMBA_PREFIX)/bin/python,python3)"

sync:
	$(UV) sync --locked --python "$(if $(wildcard $(MAMBA_PREFIX)/bin/python),$(MAMBA_PREFIX)/bin/python,python3)"

test:
	$(UV) run --locked pytest -m "not integration and not model"

model-sync:
	$(UV) sync --locked --python "$(if $(wildcard $(MAMBA_PREFIX)/bin/python),$(MAMBA_PREFIX)/bin/python,python3)" --extra model-cpu

model-check:
	$(UV) run --locked --extra model-cpu pytest -m model

integration:
	$(UV) run --locked pytest -m integration

lint:
	$(UV) run --locked ruff check src tests

check: lint test integration
	$(UV) lock --check --python "$(if $(wildcard $(MAMBA_PREFIX)/bin/python),$(MAMBA_PREFIX)/bin/python,python3)"
	$(UV) sync --check --python "$(if $(wildcard $(MAMBA_PREFIX)/bin/python),$(MAMBA_PREFIX)/bin/python,python3)" $(MODEL_EXTRA_FLAG)
	$(UV) run --locked arctic-route-risk --help

clean:
	rm -rf .venv .pytest_cache .ruff_cache .coverage htmlcov
