#!/usr/bin/env bash
# v8 orchestration: wait for the 3B balanced yes/no chain, then run GRPO on
# open-ended OK-VQA generation (the "RL beyond toy tasks" fix).
set -uo pipefail

cd /root/autodl-tmp/multimodal
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[$(date +%FT%T)] v8 orchestrate: waiting for balanced chain" > logs/grpo_okvqa.log
while ! grep -q "== 3B yes/no balanced chain done ==" logs/chain_okvqa3b_qabal.log 2>/dev/null; do
  sleep 60
done
echo "[$(date +%FT%T)] balanced chain done; starting GRPO OK-VQA" >> logs/grpo_okvqa.log

nohup /root/miniconda3/bin/python3 -m examples.multimodal_llm.grpo_okvqa \
  --init-checkpoint outputs/okvqa3b/stage2 \
  --qa-val-manifest data/okvqa/okvqa_val.jsonl \
  --image-root data/okvqa \
  --output-dir outputs/okvqa3b_grpo \
  --model-name Qwen/Qwen2.5-3B \
  --prompts 300 --eval-limit 1000 \
  --group-size 4 --batch-prompts 2 --grad-chunk 2 \
  --epochs 1 --max-new-tokens 32 --temperature 1.2 --top-p 0.95 \
  --beta 0.05 --lr 2e-5 --seed 42 --log-every 10 \
  >> logs/grpo_okvqa.log 2>&1 &
GRPO_PID=$!
echo "[$(date +%FT%T)] grpo pid=$GRPO_PID" >> logs/grpo_okvqa.log
