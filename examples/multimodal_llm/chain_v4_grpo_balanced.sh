#!/usr/bin/env bash
# Balanced-prompt GRPO variant: equal yes/no prompts in the rule-reward
# training subset, demonstrating reward-design iteration on class imbalance.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
coco_root="${COCO2017_ROOT:-/root/autodl-tmp/datasets/coco}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$project_root"

mkdir -p logs outputs/v4
echo "[chain_v4_grpo_balanced] start $(date -u +%FT%TZ)"
"$python_bin" -m examples.multimodal_llm.grpo_qa \
  --init-checkpoint outputs/v3/qa_clip_seed42 \
  --qa-train-manifest data/coco2017_recognition_train.jsonl \
  --qa-val-manifest data/coco2017_recognition_val.jsonl \
  --image-root "$coco_root" \
  --output-dir outputs/v4/grpo_qa_balanced_clip_seed42 \
  --prompts 600 --group-size 8 --batch-prompts 4 --epochs 1 \
  --max-new-tokens 8 --temperature 1.2 --top-p 0.95 \
  --beta 0.05 --lr 2e-5 --eval-limit 300 --seed 42 \
  --log-every 10 --balanced
echo "[chain_v4_grpo_balanced] done $(date -u +%FT%TZ)"
