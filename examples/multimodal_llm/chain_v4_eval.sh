#!/usr/bin/env bash
# Final re-evaluation of all completed v3 runs with the fixed decoding recipe.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
coco_root="${COCO2017_ROOT:-/root/autodl-tmp/datasets/coco}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$project_root"

ALL_RUNS=(
  clip_seed42 clip_seed43 clip_seed44
  mae_seed42 mae_seed43 mae_seed44
  random_seed42 random_seed43 random_seed44
  e10_clip_seed42 e10_clip_seed43 e50_clip_seed42 e50_clip_seed43
  lr1_clip_seed42 lr05_clip_seed42 lr1_mae_seed42 lr05_mae_seed42
  lr1_random_seed42 lr05_random_seed42 lora_r16_clip_seed42
)
TEST_RUNS=(
  clip_seed42 clip_seed43 clip_seed44
  mae_seed42 mae_seed43 mae_seed44
  random_seed42 random_seed43 random_seed44
  e10_clip_seed42 e10_clip_seed43 e50_clip_seed42 e50_clip_seed43
)
TEXT_PRIOR_RUNS=(
  clip_seed42 clip_seed43 clip_seed44
  mae_seed42 mae_seed43 mae_seed44
  random_seed42 random_seed43 random_seed44
)

mkdir -p logs
echo "[chain_v4_eval] start $(date -u +%FT%TZ)"
"$python_bin" -m examples.multimodal_llm.eval_final \
  --outputs-dir outputs/v3 \
  --val-manifest data/coco2017_val_select.jsonl \
  --test-manifest data/coco2017_val_test.jsonl \
  --image-root "$coco_root" \
  --instances-json "$coco_root/annotations/instances_val2017.json" \
  --limit-val 500 --limit-test 2500 --limit-chair 300 \
  --runs "${ALL_RUNS[@]}" \
  --test-runs "${TEST_RUNS[@]}" \
  --text-prior-runs "${TEXT_PRIOR_RUNS[@]}"
echo "[chain_v4_eval] done $(date -u +%FT%TZ)"
