#!/usr/bin/env bash
# Train the controlled MAE-transfer ablation under the COCO Karpathy protocol.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
image_root="${COCO_IMAGE_ROOT:?Set COCO_IMAGE_ROOT to the COCO-2014 root}"
train_manifest="${KARPATHY_TRAIN_MANIFEST:?Set KARPATHY_TRAIN_MANIFEST}"
val_manifest="${KARPATHY_VAL_MANIFEST:?Set KARPATHY_VAL_MANIFEST}"
test_manifest="${KARPATHY_TEST_MANIFEST:?Set KARPATHY_TEST_MANIFEST}"
mae_checkpoint="${MAE_CHECKPOINT:?Set MAE_CHECKPOINT}"
seeds="${SEEDS:-42,43,44}"
epochs="${EPOCHS:-20}"

cd "$project_root"
IFS=',' read -r -a seed_values <<< "$seeds"
for mode in random mae_frozen mae_finetune; do
  for seed in "${seed_values[@]}"; do
    args=(
      --image-root "$image_root" --train-manifest "$train_manifest"
      --val-manifest "$val_manifest" --test-manifest "$test_manifest"
      --output-dir "outputs/karpathy_${mode}_seed${seed}"
      --model-size tiny --epochs "$epochs" --batch-size 64 --num-workers 8
      --text-backbone bert --text-model-name bert-base-uncased
      --itm-loss-weight 0.1 --amp --seed "$seed"
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
