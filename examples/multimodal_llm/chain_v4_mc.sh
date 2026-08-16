#!/usr/bin/env bash
# Multiple-choice visual-recognition task: build a 4-option COCO task from
# instances annotations, SFT a CLIP-initialized VLM on 10% data (1 seed) and
# evaluate.  Shows task breadth beyond binary yes/no.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
coco_root="${COCO2017_ROOT:-/root/autodl-tmp/datasets/coco}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$project_root"

run_dir=outputs/v4/mc_clip_seed42
mkdir -p logs outputs/v4
echo "[chain_v4_mc] start $(date -u +%FT%TZ)"

"$python_bin" -m examples.multimodal_llm.build_mc_manifest \
  --instances-json "$coco_root/annotations/instances_train2017.json" \
  --image-prefix train2017 --output data/coco2017_mc_train.jsonl \
  --seed 42 --limit-images 11829
"$python_bin" -m examples.multimodal_llm.build_mc_manifest \
  --instances-json "$coco_root/annotations/instances_val2017.json" \
  --image-prefix val2017 --output data/coco2017_mc_val.jsonl \
  --seed 42 --limit-images 500

if [[ ! -f "$run_dir/done" ]]; then
  "$python_bin" -m examples.multimodal_llm.train \
    --image-root "$coco_root" --train-manifest data/coco2017_mc_train.jsonl \
    --val-manifest data/coco2017_mc_val.jsonl --output-dir "$run_dir/stage1" \
    --init clip --stage projector --epochs 1 --batch-size 8 --accum-steps 4 \
    --lr-proj 3e-4 --lr-lora 2e-4 --seed 42 --limit-train 11829 \
    --limit-val-eval 300 --mc-eval --gradient-checkpointing
  "$python_bin" -m examples.multimodal_llm.train \
    --image-root "$coco_root" --train-manifest data/coco2017_mc_train.jsonl \
    --val-manifest data/coco2017_mc_val.jsonl --output-dir "$run_dir" \
    --init clip --stage lora --epochs 2 --batch-size 8 --accum-steps 4 \
    --lr-proj 3e-4 --lr-lora 2e-4 --seed 42 --limit-train 11829 \
    --limit-val-eval 300 --mc-eval --gradient-checkpointing \
    --init-checkpoint "$run_dir/stage1"
  touch "$run_dir/done"
  rm -rf "$run_dir/stage1"
  rm -f "$run_dir/last.pt"
fi

cat "$run_dir/metrics.jsonl"
echo "[chain_v4_mc] done $(date -u +%FT%TZ)"
