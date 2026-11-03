# PathBench entry points. Run `make help` for the list.
#
# Model-free targets need only environment/requirements-analysis.lock.
# Inference targets need a GPU, the pinned checkpoints, and the inference environments
# (see README, section 2). Override interpreters with, e.g.:
#   make inference QWEN_LLAVA_PYTHON=/envs/qwen/bin/python INTERNVL_PYTHON=/envs/internvl/bin/python

PYTHON ?= python
QWEN_LLAVA_PYTHON ?= $(PYTHON)
INTERNVL_PYTHON ?= $(PYTHON)
DEVICE ?= cuda:0
RAGIG_ARCHIVE ?=
COCO_PARQUET ?=
COCO_DIR ?=

.PHONY: help install analyze reproduce figures test all models data-ragig data-lvis data-nuisance \
        smoke inference clean

help:
	@echo "Getting started (CPU, ~1 min):"
	@echo "  make install      install the pinned analysis dependencies"
	@echo "  make analyze      analyze the bundled model outputs"
	@echo "  make figures      regenerate Figures 1-2 into figures/"
	@echo "  make test         run the unit tests"
	@echo "  make all          install + analyze + figures + test"
	@echo "Model inference (GPU, ~8 GPU-hours):"
	@echo "  make models                                   download and verify pinned checkpoints"
	@echo "  make data-ragig RAGIG_ARCHIVE=images.tar.gz   rebuild RAG-IGBench views"
	@echo "  make data-lvis COCO_PARQUET='a.parquet b.parquet'  (or COCO_DIR=val2017) rebuild LVIS crops"
	@echo "  make smoke        run 2 cases of every protocol/model cell"
	@echo "  make inference    regenerate all raw records and score them"
	@echo "  make clean        remove generated outputs"

install:
	$(PYTHON) -m pip install -r environment/requirements-analysis.lock

analyze:
	$(PYTHON) scripts/reproduce_main.py

# Backward-compatible alias.
reproduce: analyze

figures:
	$(PYTHON) scripts/make_figures.py

test:
	$(PYTHON) -m pytest -q tests

all: install analyze figures test

models:
	$(QWEN_LLAVA_PYTHON) scripts/download_models.py

data-ragig:
	@test -n "$(RAGIG_ARCHIVE)" || (echo "set RAGIG_ARCHIVE=/path/to/images.tar.gz"; exit 1)
	$(PYTHON) scripts/prepare_ragig_views.py --images-archive $(RAGIG_ARCHIVE)

data-lvis:
ifneq ($(COCO_PARQUET),)
	$(PYTHON) scripts/prepare_lvis_crops.py --coco-parquet $(COCO_PARQUET)
else ifneq ($(COCO_DIR),)
	$(PYTHON) scripts/prepare_lvis_crops.py --coco-dir $(COCO_DIR)
else
	@echo "set COCO_PARQUET='val-00000-of-00002.parquet val-00001-of-00002.parquet' or COCO_DIR=/path/to/val2017"; exit 1
endif

data-nuisance:
	$(PYTHON) scripts/prepare_clevr_nuisance.py

smoke:
	for p in clevr_confirmation clevr_nuisance ragig_confirmation lvis_confirmation; do \
	  $(QWEN_LLAVA_PYTHON) scripts/run_inference.py --protocol $$p --model qwen --device $(DEVICE) \
	    --limit 2 --output results/smoke/$$p/qwen.jsonl || exit 1; \
	  $(INTERNVL_PYTHON) scripts/run_inference.py --protocol $$p --model internvl --device $(DEVICE) \
	    --limit 2 --output results/smoke/$$p/internvl.jsonl || exit 1; \
	done
	for p in clevr_confirmation clevr_nuisance; do \
	  $(QWEN_LLAVA_PYTHON) scripts/run_inference.py --protocol $$p --model llava --device $(DEVICE) \
	    --limit 2 --output results/smoke/$$p/llava.jsonl || exit 1; \
	done

inference:
	QWEN_LLAVA_PYTHON=$(QWEN_LLAVA_PYTHON) INTERNVL_PYTHON=$(INTERNVL_PYTHON) DEVICE=$(DEVICE) \
	  bash scripts/run_all_inference.sh

clean:
	rm -rf results/reproduced results/rerun results/smoke .pytest_cache
	find . -name __pycache__ -prune -exec rm -rf {} +
