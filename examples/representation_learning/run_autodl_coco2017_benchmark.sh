#!/usr/bin/env bash
# Complete COCO2017-only benchmark: fixed split, CLIP reference, MAE ablations.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
coco_root="${COCO2017_ROOT:-/root/autodl-tmp/datasets/coco}"
mae_checkpoint="${MAE_CHECKPOINT:-outputs/full_imagenet100_mae_tiny_r75/last.pt}"
source_val_manifest="${SOURCE_VAL_MANIFEST:-data/coco2017_val.jsonl}"
val_manifest="${VAL_MANIFEST:-data/coco2017_val_select.jsonl}"
test_manifest="${TEST_MANIFEST:-data/coco2017_val_test.jsonl}"
# AutoDL instances frequently cannot reach huggingface.co directly.  Callers
# can override this with a local mirror or a pre-populated Hugging Face cache.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$project_root"
"$python_bin" -m examples.representation_learning.split_retrieval_manifest \
  --input "$source_val_manifest" --val-output "$val_manifest" \
  --test-output "$test_manifest" --test-fraction 0.5 --seed 42

"$python_bin" -m examples.representation_learning.clip_retrieval_baseline \
  --image-root "$coco_root" --manifest "$test_manifest" \
  --output outputs/clip_vit_b32_coco2017_test.json --batch-size 128 --num-workers 8

COCO2017_ROOT="$coco_root" \
SOURCE_VAL_MANIFEST="$source_val_manifest" \
VAL_MANIFEST="$val_manifest" \
TEST_MANIFEST="$test_manifest" \
MAE_CHECKPOINT="$mae_checkpoint" \
  bash examples/representation_learning/run_autodl_coco2017_ablations.sh
