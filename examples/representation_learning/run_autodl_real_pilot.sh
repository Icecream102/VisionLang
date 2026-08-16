#!/usr/bin/env bash
# Run a reproducible, real-data pilot on the AutoDL public COCO2017/ImageNet100 sets.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
imagenet_root="${IMAGENET100_ROOT:-/root/autodl-tmp/datasets/imagenet100/imagenet100}"
coco_root="${COCO2017_ROOT:-/root/autodl-tmp/datasets/coco}"
mae_epochs="${MAE_EPOCHS:-3}"
probe_epochs="${PROBE_EPOCHS:-5}"
retrieval_epochs="${RETRIEVAL_EPOCHS:-3}"
train_limit="${TRAIN_LIMIT_IMAGES:-1200}"

cd "$project_root"

"$python_bin" -m examples.representation_learning.mae_pretrain \
  --data-root "$imagenet_root" \
  --val-fraction 0.1 \
  --output-dir outputs/real_imagenet100_mae_tiny_r75 \
  --model-size tiny --image-size 224 --mask-ratio 0.75 \
  --epochs "$mae_epochs" --batch-size 128 --num-workers 8 --amp --seed 42

"$python_bin" -m examples.representation_learning.mae_linear_probe \
  --train-root "$imagenet_root" --val-fraction 0.1 \
  --checkpoint outputs/real_imagenet100_mae_tiny_r75/last.pt \
  --output outputs/real_imagenet100_mae_tiny_r75/linear_probe.json \
  --epochs "$probe_epochs" --batch-size 128 --num-workers 8 --device cuda --seed 42

for mode in random mae_frozen mae_finetune; do
  args=(
    --image-root "$coco_root"
    --train-manifest data/coco2017_train.jsonl
    --val-manifest data/coco2017_val.jsonl
    --output-dir "outputs/real_coco2017_1pct_${mode}"
    --train-limit-images "$train_limit"
    --model-size tiny --image-size 224
    --epochs "$retrieval_epochs" --batch-size 64 --num-workers 8 --amp --seed 42
  )
  if [[ "$mode" != "random" ]]; then
    args+=(--mae-checkpoint outputs/real_imagenet100_mae_tiny_r75/last.pt)
  fi
  if [[ "$mode" == "mae_frozen" ]]; then
    args+=(--freeze-vision-epochs "$retrieval_epochs")
  fi
  "$python_bin" -m examples.representation_learning.mae_dual_encoder "${args[@]}"
done
