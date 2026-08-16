#!/usr/bin/env bash
# 3B yes/no recognition SFT on a BALANCED manifest (yes==no), 2 epochs,
# -> same-protocol POPE.  Tests the class-imbalance hypothesis: the 2-epoch
# imbalanced run lost to 0.5B and the 4-epoch run collapsed to majority-class
# hacking; balancing should let longer training help instead of collapse.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
cd "$project_root"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

COMMON_ARGS=(
  --image-root /root/autodl-tmp/datasets/coco
  --train-manifest data/coco2017_recognition_balanced_train.jsonl
  --val-manifest data/coco2017_recognition_val.jsonl
  --init clip
  --model-name Qwen/Qwen2.5-3B
  --lora-r 64 --lora-alpha 128 --lora-dropout 0.05
  --max-length 256 --num-workers 4
  --limit-train 23658 --limit-val-eval 300
  --eval-batch-size 8
  --gradient-checkpointing --recognition-eval --no-save-llm-base
)

log() { echo "[$(date +%FT%T)] $*"; }

run_stage1() {
  local seed=$1 out=$2
  if [[ -f "$out/done" ]]; then log "skip stage1 $out"; return 0; fi
  log "start stage1(bal) seed=$seed out=$out"
  "$python_bin" -m examples.multimodal_llm.train \
    "${COMMON_ARGS[@]}" \
    --output-dir "$out" --stage projector --epochs 1 \
    --batch-size 8 --accum-steps 4 --lr-proj 3e-4 --lr-lora 2e-4 \
    --seed "$seed" \
    > "logs/okvqa3b_qabal_s${seed}_stage1.log" 2>&1
  rm -f "$out/last.pt"
  touch "$out/done"
  log "done stage1 $out"
}

run_stage2() {
  local seed=$1 out=$2 init=$3
  if [[ -f "$out/done" ]]; then log "skip stage2 $out"; return 0; fi
  log "start stage2(bal) seed=$seed out=$out"
  "$python_bin" -m examples.multimodal_llm.train \
    "${COMMON_ARGS[@]}" \
    --output-dir "$out" --stage lora --epochs 2 \
    --batch-size 4 --accum-steps 4 --lr-proj 3e-4 --lr-lora 2e-4 \
    --seed "$seed" --init-checkpoint "$init" \
    > "logs/okvqa3b_qabal_s${seed}_stage2.log" 2>&1
  rm -f "$out/last.pt"
  touch "$out/done"
  log "done stage2 $out"
}

run_pope() {
  local seed=$1 out=$2
  if [[ -f "$out/pope.json" ]]; then log "skip pope $out"; return 0; fi
  log "start pope seed=$seed out=$out"
  "$python_bin" -m examples.multimodal_llm.eval_pope \
    --outputs-dir "$(dirname "$out")" \
    --pope-manifest-dir data/pope \
    --image-root data/pope \
    --model-name Qwen/Qwen2.5-3B \
    --runs stage2 --batch-size 16 \
    > "logs/okvqa3b_qabal_s${seed}_pope.log" 2>&1
  log "done pope $out"
}

log "== 3B yes/no balanced chain start =="
for seed in 42 43; do
  base="outputs/okvqa3b_qabal_s${seed}"
  run_stage1 "$seed" "$base/stage1"
  run_stage2 "$seed" "$base/stage2" "$base/stage1"
  run_pope "$seed" "$base/stage2"
done

log "merging pope summaries (11 keys)"
"$python_bin" -m examples.multimodal_llm.merge_pope_summary \
  --pope-jsons \
    outputs/v3/qa_clip_seed42/pope.json \
    outputs/v3/qa_clip_seed43/pope.json \
    outputs/okvqa3b/stage2/pope.json \
    outputs/okvqa3b_s43/stage2/pope.json \
    outputs/okvqa3b_s44/stage2/pope.json \
    outputs/okvqa3b_qa_s42/stage2/pope.json \
    outputs/okvqa3b_qa_s43/stage2/pope.json \
    outputs/okvqa3b_qa4_s42/stage2/pope.json \
    outputs/okvqa3b_qa4_s43/stage2/pope.json \
    outputs/okvqa3b_qabal_s42/stage2/pope.json \
    outputs/okvqa3b_qabal_s43/stage2/pope.json \
  --keys qa_clip_seed42 qa_clip_seed43 okvqa3b_stage2 okvqa3b_s43_stage2 okvqa3b_s44_stage2 okvqa3b_qa_s42_stage2 okvqa3b_qa_s43_stage2 okvqa3b_qa4_s42_stage2 okvqa3b_qa4_s43_stage2 okvqa3b_qabal_s42_stage2 okvqa3b_qabal_s43_stage2 \
  --output outputs/pope_summary.json
log "== 3B yes/no balanced chain done =="
