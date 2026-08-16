#!/usr/bin/env bash
# Text-only baseline: same Qwen2-0.5B + LoRA recipe, image removed from the
# prompt, trained on the 10% caption subset (2 seeds).  Quantifies how much of
# the VLM's caption quality comes from the visual pathway.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
coco_root="${COCO2017_ROOT:-/root/autodl-tmp/datasets/coco}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$project_root"

train_manifest=data/coco2017_train.jsonl
val_manifest=data/coco2017_val_select.jsonl

run_textonly() {
  local run_dir="$1" seed="$2" limit_train="$3"
  if [[ -f "$run_dir/done" ]]; then
    echo "Skipping completed run: $run_dir"
    return
  fi
  local extra=()
  if [[ -n "$limit_train" ]]; then
    extra+=(--limit-train "$limit_train")
  fi
  "$python_bin" -m examples.multimodal_llm.train \
    --image-root "$coco_root" --train-manifest "$train_manifest" \
    --val-manifest "$val_manifest" --output-dir "$run_dir" \
    --init clip --stage lora --epochs 2 \
    --batch-size 8 --accum-steps 4 --lr-proj 3e-4 --lr-lora 2e-4 \
    --seed "$seed" "${extra[@]}" --limit-val-eval 500 \
    --gradient-checkpointing --text-only
  touch "$run_dir/done"
  rm -f "$run_dir/last.pt"
  echo "Completed: $run_dir"
}

mkdir -p logs outputs/v4
echo "[chain_v4_textonly] start $(date -u +%FT%TZ)"
run_textonly outputs/v4/textonly_e10_clip_seed42 42 11829
run_textonly outputs/v4/textonly_e10_clip_seed43 43 11829

"$python_bin" -m examples.multimodal_llm.eval_final \
  --outputs-dir outputs/v4 \
  --val-manifest "$val_manifest" \
  --test-manifest data/coco2017_val_test.jsonl \
  --image-root "$coco_root" \
  --instances-json "$coco_root/annotations/instances_val2017.json" \
  --limit-val 500 --limit-test 2500 --limit-chair 300 \
  --runs textonly_e10_clip_seed42 textonly_e10_clip_seed43 \
  --test-runs textonly_e10_clip_seed42 textonly_e10_clip_seed43
echo "[chain_v4_textonly] done $(date -u +%FT%TZ)"
