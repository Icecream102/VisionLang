#!/usr/bin/env bash
# v6 overnight orchestration:
#   1. wait for the 3B train chain to report "== chain done =="
#   2. run the eval driver (OK-VQA/GQA/POPE for all seed runs, idempotent)
#   3. run finalize_v6 to write the summary JSON + markdown
set -uo pipefail

cd /root/autodl-tmp/multimodal
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[$(date +%FT%T)] orchestrate: waiting for train chain" > logs/chain_v6_eval.log
while ! grep -q "== chain done ==" logs/chain_okvqa3b_seeds.log 2>/dev/null; do
  sleep 60
done
echo "[$(date +%FT%T)] train chain done; starting eval driver" >> logs/chain_v6_eval.log

bash examples/multimodal_llm/chain_v6_eval.sh >> logs/chain_v6_eval.log 2>&1
eval_exit=$?
echo "[$(date +%FT%T)] eval driver exit=$eval_exit" >> logs/chain_v6_eval.log

/root/miniconda3/bin/python3 -m examples.multimodal_llm.finalize_v6 \
  --output-json outputs/v6_scale_summary.json \
  --output-md docs/results_v6_okvqa_scale.md >> logs/chain_v6_eval.log 2>&1
echo "[$(date +%FT%T)] finalize exit=$?" >> logs/chain_v6_eval.log
