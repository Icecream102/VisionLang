#!/usr/bin/env bash
# v3 final experiment driver: model warmup, smoke test, LR sensitivity, main matrix.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
coco_root="${COCO2017_ROOT:-/root/autodl-tmp/datasets/coco}"
train_manifest="${TRAIN_MANIFEST:-$project_root/data/coco2017_train.jsonl}"
val_manifest="${VAL_MANIFEST:-$project_root/data/coco2017_val_select.jsonl}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mode="${1:?Usage: $0 smoke|lr_sweep|main}"
cd "$project_root"

mkdir -p outputs/v3

install_deps() {
  "$python_bin" -m pip install -q peft accelerate einops pycocoevalcap pycocotools
  if ! command -v java >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq --no-install-recommends default-jre-headless
  fi
}

warm_models() {
  "$python_bin" - <<'PY'
from transformers import AutoConfig, AutoTokenizer, CLIPVisionModel, Qwen2ForCausalLM, ViTMAEModel
AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype="auto")
ViTMAEModel.from_pretrained("facebook/vit-mae-base")
CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
print("models ready")
PY
}

check_manifest() {
  if [[ ! -f "$train_manifest" ]] || [[ ! -f "$val_manifest" ]]; then
    echo "Missing manifest: $train_manifest / $val_manifest" >&2
    exit 1
  fi
}

run_pair() {
  # run_pair <run_dir> <init> <seed> <stage1_epochs> <stage2_epochs> \
  #            <limit_train> <limit_val> <batch> <accum> <lr_proj> <lr_lora>
  local run_dir="$1" init="$2" seed="$3"
  local s1_epochs="$4" s2_epochs="$5" limit_train="$6" limit_val="$7"
  local batch="$8" accum="$9" lr_proj="${10}" lr_lora="${11}"
  local extra_args=()
  local run_extra=("${@:12}")
  if [[ -f "$run_dir/done" ]]; then
    echo "Skipping completed run: $run_dir"
    return
  fi
  local s1_dir="$run_dir/stage1"
  mkdir -p "$s1_dir"
  if [[ -n "$limit_train" ]]; then
    extra_args+=(--limit-train "$limit_train")
  fi
  extra_args+=(--gradient-checkpointing)
  extra_args+=("${run_extra[@]}")
  "$python_bin" -m examples.multimodal_llm.train \
    --image-root "$coco_root" --train-manifest "$train_manifest" \
    --val-manifest "$val_manifest" --output-dir "$s1_dir" \
    --init "$init" --stage projector --epochs "$s1_epochs" \
    --batch-size "$batch" --accum-steps "$accum" \
    --lr-proj "$lr_proj" --lr-lora "$lr_lora" \
    --seed "$seed" "${extra_args[@]}" --limit-val-eval "$limit_val"
  "$python_bin" -m examples.multimodal_llm.train \
    --image-root "$coco_root" --train-manifest "$train_manifest" \
    --val-manifest "$val_manifest" --output-dir "$run_dir" \
    --init "$init" --stage lora --epochs "$s2_epochs" \
    --batch-size "$batch" --accum-steps "$accum" \
    --lr-proj "$lr_proj" --lr-lora "$lr_lora" \
    --seed "$seed" "${extra_args[@]}" --limit-val-eval "$limit_val" \
    --init-checkpoint "$s1_dir"
  touch "$run_dir/done"
  echo "Completed: $run_dir"
  # Free disk: stage-1 artifacts and resume-only checkpoints are no longer needed.
  rm -rf "$run_dir/stage1"
  rm -f "$run_dir/last.pt"
}

run_qa_pair() {
  # run_qa_pair <run_dir> <init> <seed> <epochs> <limit_train> <limit_val> <batch> <accum>
  local run_dir="$1" init="$2" seed="$3"
  local epochs="$4" limit_train="$5" limit_val="$6" batch="$7" accum="$8"
  if [[ -f "$run_dir/done" ]]; then
    echo "Skipping completed QA run: $run_dir"
    return
  fi
  local s1_dir="$run_dir/stage1"
  mkdir -p "$s1_dir"
  "$python_bin" -m examples.multimodal_llm.train \
    --image-root "$coco_root" --train-manifest "$qa_train_manifest" \
    --val-manifest "$qa_val_manifest" --output-dir "$s1_dir" \
    --init "$init" --stage projector --epochs 1 \
    --batch-size "$batch" --accum-steps "$accum" \
    --lr-proj 3e-4 --lr-lora 2e-4 \
    --seed "$seed" --limit-train "$limit_train" --limit-val-eval "$limit_val" \
    --recognition-eval --gradient-checkpointing
  "$python_bin" -m examples.multimodal_llm.train \
    --image-root "$coco_root" --train-manifest "$qa_train_manifest" \
    --val-manifest "$qa_val_manifest" --output-dir "$run_dir" \
    --init "$init" --stage lora --epochs "$epochs" \
    --batch-size "$batch" --accum-steps "$accum" \
    --lr-proj 3e-4 --lr-lora 2e-4 \
    --seed "$seed" --limit-train "$limit_train" --limit-val-eval "$limit_val" \
    --recognition-eval --gradient-checkpointing --init-checkpoint "$s1_dir"
  touch "$run_dir/done"
  echo "Completed QA run: $run_dir"
  rm -rf "$run_dir/stage1"
  rm -f "$run_dir/last.pt"
}

check_manifest

case "$mode" in
  smoke)
    install_deps
    warm_models
    run_pair outputs/v3/smoke_clip_seed42 clip 42 1 1 200 50 8 1 3e-4 2e-4
    ;;
  lr_sweep)
    install_deps
    warm_models
    for init in random mae clip; do
      run_pair "outputs/v3/lr1_${init}_seed42" "$init" 42 1 2 5000 200 8 4 3e-4 2e-4
      run_pair "outputs/v3/lr05_${init}_seed42" "$init" 42 1 2 5000 200 8 4 3e-4 1e-4
    done
    ;;
  main)
    install_deps
    warm_models
    for init in random mae clip; do
      for seed in 42 43 44; do
        run_pair "outputs/v3/${init}_seed${seed}" "$init" "$seed" 1 2 "" 500 8 4 3e-4 2e-4
      done
    done
    ;;
  ae)
    install_deps
    warm_models
    # A: core 3x3 initialization comparison on the full COCO2017 train split.
    for init in random mae clip; do
      for seed in 42 43 44; do
        run_pair "outputs/v3/${init}_seed${seed}" "$init" "$seed" 1 2 "" 500 8 4 3e-4 2e-4
      done
    done
    # E: data-scaling curve (10%/50%); the 100% point reuses the CLIP runs above.
    for seed in 42 43; do
      run_pair "outputs/v3/e10_clip_seed${seed}" clip "$seed" 1 2 11829 500 8 4 3e-4 2e-4
      run_pair "outputs/v3/e50_clip_seed${seed}" clip "$seed" 1 2 59144 500 8 4 3e-4 2e-4
    done
    ;;
  extras)
    install_deps
    warm_models
    # PEFT depth: LoRA rank ablation on 10% data (CLIP init).
    run_pair "outputs/v3/lora_r16_clip_seed42" clip 42 1 2 11829 500 8 4 3e-4 2e-4 \
      --lora-r 16 --lora-alpha 32
    # Task breadth: yes/no visual-recognition (COCO instances) on 10% data.
    qa_train_manifest="data/coco2017_recognition_train.jsonl"
    qa_val_manifest="data/coco2017_recognition_val.jsonl"
    "$python_bin" -m examples.multimodal_llm.build_recognition_manifest \
      --instances-json "$coco_root/annotations/instances_train2017.json" \
      --image-prefix train2017 --output "$qa_train_manifest" --seed 42 \
      --limit-images 11829
    "$python_bin" -m examples.multimodal_llm.build_recognition_manifest \
      --instances-json "$coco_root/annotations/instances_val2017.json" \
      --image-prefix val2017 --output "$qa_val_manifest" --seed 42 --limit-images 500
    for seed in 42 43; do
      run_qa_pair "outputs/v3/qa_clip_seed${seed}" clip "$seed" 2 11829 300 8 4
    done
    # Hallucination evaluation (CHAIR) over all completed runs.
    "$python_bin" -m examples.multimodal_llm.eval_chair \
      --outputs-dir outputs/v3 --val-manifest "$val_manifest" \
      --image-root "$coco_root" \
      --instances-json "$coco_root/annotations/instances_val2017.json" \
      --limit-images 300
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    exit 1
    ;;
esac

echo "v3 pipeline mode=$mode finished"
