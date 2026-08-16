#!/usr/bin/env bash
# 3B yes/no recognition SFT, 4 epochs (vs 2-epoch baseline), -> same-protocol
# POPE.  Everything else identical: same recognition manifest (limit_train
# 11829), same LR (3e-4/2e-4), same stage1 init, same seeds 42/43.
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

run_stage2_4ep() {
  local seed=$1 out=$2 init=$3
  if [[ -f "$out/done" ]]; then log "skip stage2 $out"; return 0; fi
  log "start stage2(4ep) seed=$seed out=$out"
  "$python_bin" -m examples.multimodal_llm.train \
    "${COMMON_ARGS[@]}" \
    --output-dir "$out" --stage lora --epochs 4 \
    --batch-size 4 --accum-steps 4 --lr-proj 3e-4 --lr-lora 2e-4 \
    --seed "$seed" --init-checkpoint "$init" \
    > "logs/okvqa3b_qa4_s${seed}_stage2.log" 2>&1
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
    > "logs/okvqa3b_qa4_s${seed}_pope.log" 2>&1
  log "done pope $out"
}

log "== 3B yes/no 4-epoch chain start =="
for seed in 42 43; do
  base="outputs/okvqa3b_qa4_s${seed}"
  run_stage2_4ep "$seed" "$base/stage2" "outputs/okvqa3b_qa_s${seed}/stage1"
  run_pope "$seed" "$base/stage2"
done

log "merging pope summaries (9 keys)"
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
  --keys qa_clip_seed42 qa_clip_seed43 okvqa3b_stage2 okvqa3b_s43_stage2 okvqa3b_s44_stage2 okvqa3b_qa_s42_stage2 okvqa3b_qa_s43_stage2 okvqa3b_qa4_s42_stage2 okvqa3b_qa4_s43_stage2 \
  --output outputs/pope_summary.json
log "== 3B yes/no 4-epoch chain done =="
