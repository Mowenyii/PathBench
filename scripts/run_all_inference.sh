#!/usr/bin/env bash
# Regenerate every raw record behind Tables 1-3 (about 8 GPU-hours on one 80 GB A100-class GPU).
# Qwen/LLaVA and InternVL need different transformers versions; set the interpreters:
#   QWEN_LLAVA_PYTHON=<python with environment/requirements-qwen-llava-cu121.lock>
#   INTERNVL_PYTHON=<python with environment/requirements-internvl-cu121.lock>
# RAG-IGBench views and LVIS crops must be built first (see README, "Data preparation").
set -euo pipefail
cd "$(dirname "$0")/.."
QWEN_LLAVA_PYTHON=${QWEN_LLAVA_PYTHON:-python}
INTERNVL_PYTHON=${INTERNVL_PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
OUT=${OUT:-results/rerun}

run() {  # run <python> <protocol> <model>
  "$1" scripts/run_inference.py --protocol "$2" --model "$3" --device "$DEVICE" \
    --output "$OUT/$2/$3.jsonl"
}

for protocol in clevr_confirmation clevr_nuisance; do
  run "$QWEN_LLAVA_PYTHON" "$protocol" qwen
  run "$QWEN_LLAVA_PYTHON" "$protocol" llava
  run "$INTERNVL_PYTHON" "$protocol" internvl
done
for protocol in ragig_confirmation lvis_confirmation; do
  run "$QWEN_LLAVA_PYTHON" "$protocol" qwen
  run "$INTERNVL_PYTHON" "$protocol" internvl
done
# Figure 2A uses the 72-scene development records, which are scored from results/frozen.
mkdir -p "$OUT/clevr_development"
cp -n results/frozen/clevr_development/*.jsonl "$OUT/clevr_development/"
"$QWEN_LLAVA_PYTHON" scripts/reproduce_main.py --results "$OUT" --output-dir "$OUT/reproduced" --no-check
