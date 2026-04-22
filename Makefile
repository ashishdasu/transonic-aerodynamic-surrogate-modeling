SHELL  := /bin/bash
PYTHON := python3
VENV   := .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

.PHONY: help setup test eda train eval report all clean distclean

help:
	@echo "Targets:"
	@echo "  setup    Create .venv and install pinned deps (Python >=3.11 required)."
	@echo "  test     Run unit tests."
	@echo "  eda      Generate EDA figures (results/figures/eda/)."
	@echo "  train    Train all models, serialize to results/models/."
	@echo "  eval     Generate evaluation tables and figures."
	@echo "  report   Compile report/final.pdf (needs pdflatex + bibtex)."
	@echo "  all      test -> eda -> train -> eval -> report."
	@echo "  clean    Remove build artifacts (keeps .venv and results/)."
	@echo "  distclean Also remove .venv and every regenerable artifact."

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

test: setup
	$(PY) -m pytest tests/ -v

eda: setup
	$(PY) scripts/run_all.py --stage eda

train: setup
	$(PY) scripts/run_all.py --stage train

eval: setup
	$(PY) scripts/run_all.py --stage eval

report:
	cd report && \
	pdflatex -interaction=nonstopmode final.tex && \
	bibtex final && \
	pdflatex -interaction=nonstopmode final.tex && \
	pdflatex -interaction=nonstopmode final.tex

all: test eda train eval report

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -f report/*.aux report/*.log report/*.bbl report/*.blg \
	      report/*.out report/*.toc report/*.fdb_latexmk report/*.fls \
	      report/*.synctex.gz

distclean: clean
	rm -rf $(VENV)
	rm -rf results/figures/* results/tables/* results/models/*
	touch results/figures/.gitkeep results/tables/.gitkeep results/models/.gitkeep
