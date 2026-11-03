# PathBench

PathBench evaluates whether a visual agent preserves the same structured state
when it receives the same observations in a different order. It includes frozen
model outputs, deterministic scoring code, data manifests, and scripts for
running the supported vision-language models.

## Getting started

PathBench requires Python 3.10 or later. The included model outputs can be
analyzed on a CPU with about 1 GB of memory; no model download is required.

```bash
python -m pip install -r environment/requirements-analysis.lock
python scripts/reproduce_main.py
python -m pytest -q tests
```

The analysis writes Markdown and JSON summaries to `results/reproduced/`.
To regenerate the two summary figures, run:

```bash
python scripts/make_figures.py
```

The same commands are available through `make all`. Run `make help` to list
individual targets.

## Running model inference

Full inference requires a CUDA GPU with at least 24 GB of memory. The original
runs used an NVIDIA A100 80 GB GPU with CUDA 12.1. Qwen2.5-VL and
LLaVA-OneVision use a different Transformers version from InternVL2.5, so they
must be installed in separate environments.

```bash
# Qwen2.5-VL and LLaVA-OneVision
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r environment/requirements-qwen-llava-cu121.lock

# InternVL2.5 (use a separate environment)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r environment/requirements-internvl-cu121.lock
```

Download the pinned checkpoints:

```bash
python scripts/download_models.py
python scripts/download_models.py --verify-only
```

The checkpoints are stored under `models/`. Their repository names, revisions,
and file hashes are defined in `configs/pathbench.json` and
`configs/checkpoints/`.

### Prepare external images

CLEVR crops are included. RAG-IGBench and COCO images are not redistributed;
download them from their official sources and build the required views locally.

```bash
python scripts/prepare_ragig_views.py --images-archive /path/to/images.tar.gz
python scripts/prepare_lvis_crops.py --coco-dir /path/to/val2017

# The LVIS script also accepts COCO parquet shards:
python scripts/prepare_lvis_crops.py \
  --coco-parquet val-00000-of-00002.parquet val-00001-of-00002.parquet
```

Each preparation script verifies the generated images against SHA-256 hashes in
the corresponding manifest. See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) for
sources and license terms.

### Run a protocol

```bash
python scripts/run_inference.py \
  --protocol clevr_confirmation \
  --model qwen \
  --device cuda:0
```

Available protocols are `clevr_confirmation`, `clevr_nuisance`,
`ragig_confirmation`, and `lvis_confirmation`. Supported model keys are listed
for each protocol in `configs/pathbench.json`. Use `--limit N` for a short
smoke test. Interrupted runs resume from the existing output file.

To run every configured protocol/model pair:

```bash
QWEN_LLAVA_PYTHON=/path/to/qwen-env/bin/python \
INTERNVL_PYTHON=/path/to/internvl-env/bin/python \
bash scripts/run_all_inference.sh
```

New records are written to `results/rerun/`. Analyze them with:

```bash
python scripts/reproduce_main.py --results results/rerun --no-check
```

Greedy decoding can vary across GPUs, drivers, and library versions. The
`--no-check` option computes metrics without requiring them to match the bundled
reference values.

## Protocols

| Protocol | Cases | State |
|---|---:|---|
| `clevr_confirmation` | 180 | Object attributes indexed by opaque observation ID |
| `clevr_nuisance` | 180 | CLEVR state under one-pixel and ID-relabeling controls |
| `ragig_confirmation` | 120 | Query-to-source-image mapping |
| `lvis_confirmation` | 120 | Object category indexed by opaque observation ID |
| `clevr_development` | 72 | Development records included for analysis only |

Prompts, parsers, and deterministic executors are implemented in `pathbench/`.
See [DATA.md](DATA.md) for the manifest and output schemas.

## Repository layout

```text
pathbench/          protocol, model, metric, and execution code
scripts/            analysis, inference, data preparation, and figure scripts
configs/            protocol settings and pinned model revisions
data/               case manifests and redistributable CLEVR crops
results/frozen/     bundled raw model outputs
environment/        pinned Python dependencies
tests/              parser, executor, and end-to-end checks
```

## License and citation

The code is released under the MIT License. Bundled data are released under
CC BY-SA 4.0; upstream terms still apply. See
[THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) for details.

```bibtex
@inproceedings{mo2027answer,
  title     = {Answer Agreement Is Not State Integrity: A Benchmark for Silent State Corruption in Visual Agents},
  author    = {Mo, Wenyi and Metaxas, Dimitris},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```
