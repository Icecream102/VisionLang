#!/usr/bin/env bash
# Independent v2 reproduction: MAE -> linear probe -> CLIP -> 3x3 retrieval ablation.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
imagenet_root="${IMAGENET100_ROOT:-/root/autodl-tmp/datasets/imagenet100/imagenet100}"
coco_root="${COCO2017_ROOT:-/root/autodl-tmp/datasets/coco}"
mae_epochs="${MAE_EPOCHS:-100}"
probe_epochs="${PROBE_EPOCHS:-50}"
retrieval_epochs="${RETRIEVAL_EPOCHS:-20}"
mae_output="outputs/v2_imagenet100_mae_tiny_r75"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$project_root"

"$python_bin" -m examples.representation_learning.mae_pretrain \
  --data-root "$imagenet_root" --val-fraction 0.1 --output-dir "$mae_output" \
  --model-size tiny --image-size 224 --mask-ratio 0.75 --epochs "$mae_epochs" \
  --batch-size 128 --num-workers 8 --amp --seed 42

"$python_bin" -m examples.representation_learning.mae_linear_probe \
  --train-root "$imagenet_root" --val-fraction 0.1 \
  --checkpoint "$mae_output/last.pt" --output "$mae_output/linear_probe.json" \
  --epochs "$probe_epochs" --batch-size 128 --num-workers 8 --device cuda --seed 42

"$python_bin" -m examples.representation_learning.split_retrieval_manifest \
  --input data/coco2017_val.jsonl --val-output data/coco2017_val_select.jsonl \
  --test-output data/coco2017_val_test.jsonl --test-fraction 0.5 --seed 42

"$python_bin" -m examples.representation_learning.clip_retrieval_baseline \
  --image-root "$coco_root" --manifest data/coco2017_val_test.jsonl \
  --output outputs/clip_vit_b32_coco2017_v2_test.json --batch-size 128 --num-workers 8

COCO2017_ROOT="$coco_root" MAE_CHECKPOINT="$mae_output/last.pt" \
OUTPUT_PREFIX=coco2017_v2 EPOCHS="$retrieval_epochs" \
  bash examples/representation_learning/run_autodl_coco2017_ablations.sh
