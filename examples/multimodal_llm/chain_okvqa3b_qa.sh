#!/usr/bin/env bash
# 3B yes/no recognition SFT -> same-protocol POPE (mirrors 0.5B QA recipe).
#
# The 0.5B QA checkpoints (qa_clip_seed42/43) were trained on the COCO
# recognition yes/no manifests (limit_train=11829, stage1 + stage2 lora).
# This chain runs the identical recipe with Qwen2.5-3B + LoRA on two seeds,
# then evaluates POPE on the resulting checkpoints for a same-protocol
# 0.5B-vs-3B hallucination comparison.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
cd "$project_root"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

COMMON_ARGS=(
  --image-root /root/autodl-tmp/datasets/coco
  --train-manifest data/coco2017_recognition_train.jsonl
  --val-manifest data/coco2017_recognition_val.jsonl
  --init clip
  --model-name Qwen/Qwen2.5-3B
  --lora-r 64 --lora-alpha 128 --lora-dropout 0.05
  --max-length 256 --num-workers 4
  --limit-train 11829 --limit-val-eval 300
  --eval-batch-size 8
  --gradient-checkpointing --recognition-eval --no-save-llm-base
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
    > "logs/okvqa3b_qa_s${seed}_stage1.log" 2>&1
  rm -f "$out/last.pt"
  touch "$out/done"
  log "done stage1 $out"
}

run_stage2() {
  local seed=$1 out=$2 init=$3
  if [[ -f "$out/done" ]]; then log "skip stage2 $out"; return 0; fi
  log "start stage2 seed=$seed out=$out"
  "$python_bin" -m examples.multimodal_llm.train \
    "${COMMON_ARGS[@]}" \
    --output-dir "$out" --stage lora --epochs 2 \
    --batch-size 4 --accum-steps 4 --lr-proj 3e-4 --lr-lora 2e-4 \
    --seed "$seed" --init-checkpoint "$init" \
    > "logs/okvqa3b_qa_s${seed}_stage2.log" 2>&1
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
    > "logs/okvqa3b_qa_s${seed}_pope.log" 2>&1
  log "done pope $out"
}

log "== 3B yes/no SFT chain start =="
for seed in 42 43; do
  base="outputs/okvqa3b_qa_s${seed}"
  run_stage1 "$seed" "$base/stage1"
  run_stage2 "$seed" "$base/stage2" "$base/stage1"
  run_pope "$seed" "$base/stage2"
done
log "== 3B yes/no SFT chain done =="
