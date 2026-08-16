#!/usr/bin/env bash
# COCO2017-only controlled MAE-transfer experiment with a fixed val/test split.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
coco_root="${COCO2017_ROOT:-/root/autodl-tmp/datasets/coco}"
train_manifest="${TRAIN_MANIFEST:-data/coco2017_train.jsonl}"
source_val_manifest="${SOURCE_VAL_MANIFEST:-data/coco2017_val.jsonl}"
val_manifest="${VAL_MANIFEST:-data/coco2017_val_select.jsonl}"
test_manifest="${TEST_MANIFEST:-data/coco2017_val_test.jsonl}"
mae_checkpoint="${MAE_CHECKPOINT:?Set MAE_CHECKPOINT to a completed MAE last.pt}"
seeds="${SEEDS:-42,43,44}"
epochs="${EPOCHS:-20}"
output_prefix="${OUTPUT_PREFIX:-coco2017_v2}"
# Prefer the AutoDL mirror when a model is not already in the local cache.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$project_root"
"$python_bin" -m examples.representation_learning.split_retrieval_manifest \
  --input "$source_val_manifest" --val-output "$val_manifest" \
  --test-output "$test_manifest" --test-fraction 0.5 --seed 42

IFS=',' read -r -a seed_values <<< "$seeds"
for mode in random mae_frozen mae_finetune; do
  for seed in "${seed_values[@]}"; do
    output_dir="outputs/${output_prefix}_${mode}_seed${seed}"
    # A completed run has a held-out test result.  This makes the driver safe
    # to restart after a disk expansion without rerunning finished seeds.
    if [[ -f "$output_dir/test_metrics.json" ]]; then
      echo "Skipping completed run: $output_dir"
      continue
    fi
    args=(
      --image-root "$coco_root" --train-manifest "$train_manifest"
      --val-manifest "$val_manifest" --test-manifest "$test_manifest"
      --output-dir "$output_dir"
      --model-size tiny --epochs "$epochs" --batch-size 64 --num-workers 8
      --text-backbone bert --text-model-name bert-base-uncased
      --lr 1e-4 --vision-lr 1e-5 --text-lr 2e-5 --warmup-epochs 1
      --min-lr-ratio 0.1 --grad-clip-norm 1.0 --itm-loss-weight 0.1 --amp --seed "$seed"
    )
    if [[ "$mode" != "random" ]]; then
      args+=(--mae-checkpoint "$mae_checkpoint")
    fi
    if [[ "$mode" == "mae_frozen" ]]; then
      args+=(--freeze-vision-epochs "$epochs")
    fi
    "$python_bin" -m examples.representation_learning.mae_dual_encoder "${args[@]}"
  done
done
