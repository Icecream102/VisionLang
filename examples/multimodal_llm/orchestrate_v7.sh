#!/usr/bin/env bash
# v7 orchestration: wait for the 3B yes/no SFT chain, then merge POPE
# summaries (0.5B QA x2 + 3B OK-VQA x3 + 3B QA x2).
set -uo pipefail

cd /root/autodl-tmp/multimodal
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[$(date +%FT%T)] v7 orchestrate: waiting for QA chain" > logs/chain_v7.log
while ! grep -q "== 3B yes/no SFT chain done ==" logs/chain_okvqa3b_qa.log 2>/dev/null; do
  sleep 60
done
echo "[$(date +%FT%T)] QA chain done; merging pope summaries" >> logs/chain_v7.log

/root/miniconda3/bin/python3 -m examples.multimodal_llm.merge_pope_summary \
  --pope-jsons \
    outputs/v3/qa_clip_seed42/pope.json \
    outputs/v3/qa_clip_seed43/pope.json \
    outputs/okvqa3b/stage2/pope.json \
    outputs/okvqa3b_s43/stage2/pope.json \
    outputs/okvqa3b_s44/stage2/pope.json \
    outputs/okvqa3b_qa_s42/stage2/pope.json \
    outputs/okvqa3b_qa_s43/stage2/pope.json \
  --keys qa_clip_seed42 qa_clip_seed43 okvqa3b_stage2 okvqa3b_s43_stage2 okvqa3b_s44_stage2 okvqa3b_qa_s42_stage2 okvqa3b_qa_s43_stage2 \
  --output outputs/pope_summary.json >> logs/chain_v7.log 2>&1
echo "[$(date +%FT%T)] merge exit=$?" >> logs/chain_v7.log
