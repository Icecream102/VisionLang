#!/usr/bin/env bash
# 3B OK-VQA multi-seed + LR ablation chain (single 4090D).
#
#   seed 43: stage1 (proj, 1 ep) -> stage2 (lora, 2 ep) -> full OK-VQA eval
#   seed 44: stage1 (proj, 1 ep) -> stage2 (lora, 2 ep) -> full OK-VQA eval
#   LR low  (1.5e-4 / 1e-4): stage2 from existing seed-42 stage1 -> eval
#   LR high (6e-4  / 4e-4): stage2 from existing seed-42 stage1 -> eval
#
# Each step is idempotent via a `done` marker / existing output file, so the
# chain can be re-run after an interruption without repeating finished steps.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
cd "$project_root"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

COMMON_ARGS=(
  --image-root /root/autodl-tmp/datasets/coco
  --val-image-root data/okvqa
  --train-manifest data/okvqa/okvqa_train.jsonl
  --val-manifest data/okvqa/okvqa_val.jsonl
  --init clip
  --model-name Qwen/Qwen2.5-3B
  --lora-r 64 --lora-alpha 128 --lora-dropout 0.05
  --max-length 256 --num-workers 4
  --limit-val-eval 500 --eval-batch-size 4
  --gradient-checkpointing --vqa-eval --no-save-llm-base
)

log() { echo "[$(date +%FT%T)] $*"; }

run_stage1() {
  local seed=$1 out=$2
  if [[ -f "$out/done" ]]; then log "skip stage1 $out"; return 0; fi
  log "start stage1 seed=$seed out=$out"
  "$python_bin" -m examples.multimodal_llm.train \
    "${COMMON_ARGS[@]}" \
    --output-dir "$out" --stage projector --epochs 1 \
    --batch-size 8 --accum-steps 4 --lr-proj 3e-4 --lr-lora 2e-4 \
    --seed "$seed" \
    > "logs/okvqa3b_$(basename "$(dirname "$out")")_stage1.log" 2>&1
  touch "$out/done"
  log "done stage1 $out"
}

run_stage2() {
  local seed=$1 out=$2 init=$3 lr_proj=$4 lr_lora=$5
  if [[ -f "$out/done" ]]; then log "skip stage2 $out"; return 0; fi
  log "start stage2 seed=$seed out=$out lr=$lr_proj/$lr_lora"
  "$python_bin" -m examples.multimodal_llm.train \
    "${COMMON_ARGS[@]}" \
    --output-dir "$out" --stage lora --epochs 2 \
    --batch-size 4 --accum-steps 4 --lr-proj "$lr_proj" --lr-lora "$lr_lora" \
    --seed "$seed" --init-checkpoint "$init" \
    > "logs/okvqa3b_$(basename "$(dirname "$out")")_stage2.log" 2>&1
  touch "$out/done"
  log "done stage2 $out"
}

run_eval() {
  local out=$1
  local json_out="$out/okvqa_val_full.json"
  if [[ -f "$json_out" ]]; then log "skip eval $out"; return 0; fi
  log "start eval $out"
  "$python_bin" -m examples.multimodal_llm.eval_okvqa \
    --run-dir "$out" \
    --val-manifest data/okvqa/okvqa_val.jsonl \
    --image-root data/okvqa \
    --model-name Qwen/Qwen2.5-3B --batch-size 8 \
    --output "$json_out" \
    > "logs/okvqa3b_$(basename "$(dirname "$out")")_eval_full.log" 2>&1
  log "done eval $out"
}

log "== chain start =="

for seed in 43 44; do
  base="outputs/okvqa3b_s${seed}"
  run_stage1 "$seed" "$base/stage1"
  run_stage2 "$seed" "$base/stage2" "$base/stage1" 3e-4 2e-4
  run_eval "$base/stage2"
done

run_stage2 42 "outputs/okvqa3b_lr_low/stage2" "outputs/okvqa3b/stage1" 1.5e-4 1e-4
run_eval "outputs/okvqa3b_lr_low/stage2"
run_stage2 42 "outputs/okvqa3b_lr_high/stage2" "outputs/okvqa3b/stage1" 6e-4 4e-4
run_eval "outputs/okvqa3b_lr_high/stage2"

log "== chain done =="
for dir in \
  outputs/okvqa3b/stage2 \
  outputs/okvqa3b_s43/stage2 \
  outputs/okvqa3b_s44/stage2 \
  outputs/okvqa3b_lr_low/stage2 \
  outputs/okvqa3b_lr_high/stage2; do
  if [[ -f "$dir/okvqa_val_full.json" ]]; then
    acc=$("$python_bin" -c "import json;print(json.load(open('$dir/okvqa_val_full.json'))['accuracy'])")
    log "OK-VQA acc $dir = $acc"
  else
    log "MISSING eval $dir"
  fi
done
