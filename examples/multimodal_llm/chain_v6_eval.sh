#!/usr/bin/env bash
# Post-training eval driver: fills in any missing OK-VQA / GQA / POPE results
# for the 3B OK-VQA seed runs, then merges POPE summaries.  Idempotent.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
cd "$project_root"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log() { echo "[$(date +%FT%T)] $*"; }

seed_dirs=(
  outputs/okvqa3b/stage2
  outputs/okvqa3b_s43/stage2
  outputs/okvqa3b_s44/stage2
)

for dir in "${seed_dirs[@]}"; do
  name=$(basename "$(dirname "$dir")")
  if [[ ! -f "$dir/okvqa_val_full.json" ]]; then
    log "OK-VQA eval $dir"
    "$python_bin" -m examples.multimodal_llm.eval_okvqa \
      --run-dir "$dir" \
      --val-manifest data/okvqa/okvqa_val.jsonl \
      --image-root data/okvqa \
      --model-name Qwen/Qwen2.5-3B --batch-size 8 \
      --output "$dir/okvqa_val_full.json" \
      > "logs/${name}_eval_full.log" 2>&1
  else
    log "OK-VQA done $dir"
  fi

  if [[ ! -f "$dir/gqa_testdev_full.json" ]]; then
    log "GQA eval $dir"
    "$python_bin" -m examples.multimodal_llm.eval_gqa \
      --run-dir "$dir" \
      --val-manifest data/gqa/gqa_testdev.jsonl \
      --image-root data/gqa \
      --model-name Qwen/Qwen2.5-3B --batch-size 8 \
      --output "$dir/gqa_testdev_full.json" \
      > "logs/${name}_gqa_eval.log" 2>&1
  else
    log "GQA done $dir"
  fi

  if [[ ! -f "$dir/pope.json" ]]; then
    log "POPE eval $dir"
    "$python_bin" -m examples.multimodal_llm.eval_pope \
      --outputs-dir "$(dirname "$dir")" \
      --pope-manifest-dir data/pope \
      --image-root data/pope \
      --model-name Qwen/Qwen2.5-3B \
      --runs stage2 --batch-size 16 \
      > "logs/${name}_pope.log" 2>&1
  else
    log "POPE done $dir"
  fi
done

log "merging pope summaries"
"$python_bin" -m examples.multimodal_llm.merge_pope_summary \
  --pope-jsons \
    outputs/v3/qa_clip_seed42/pope.json \
    outputs/v3/qa_clip_seed43/pope.json \
    outputs/okvqa3b/stage2/pope.json \
    outputs/okvqa3b_s43/stage2/pope.json \
    outputs/okvqa3b_s44/stage2/pope.json \
  --keys qa_clip_seed42 qa_clip_seed43 okvqa3b_stage2 okvqa3b_s43_stage2 okvqa3b_s44_stage2 \
  --output outputs/pope_summary.json

log "== v6 eval driver done =="
for dir in "${seed_dirs[@]}"; do
  log "$dir okvqa=$(/root/miniconda3/bin/python3 -c "import json;print(json.load(open('$dir/okvqa_val_full.json'))['accuracy'])" 2>/dev/null || echo NA) gqa=$(/root/miniconda3/bin/python3 -c "import json;print(json.load(open('$dir/gqa_testdev_full.json'))['accuracy'])" 2>/dev/null || echo NA)"
done
