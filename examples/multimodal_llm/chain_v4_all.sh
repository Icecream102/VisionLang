#!/usr/bin/env bash
# Full v4 chain on a single GPU, run in background with nohup:
#   nohup bash examples/multimodal_llm/chain_v4_all.sh > logs/v4_chain.log 2>&1 &
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
cd "$project_root"
mkdir -p logs

PIDFILE=logs/v4_chain.pid
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "chain_v4_all already running (pid $(cat "$PIDFILE")); refusing to start a duplicate."
  exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

echo "== v4 chain start $(date -u +%FT%TZ)"

echo "== [1/4] final re-evaluation of v3 runs (new decoding, val+test)"
bash examples/multimodal_llm/chain_v4_eval.sh 2>&1 | tee logs/v4_eval.log

echo "== [2/4] text-only baseline training"
bash examples/multimodal_llm/chain_v4_textonly.sh 2>&1 | tee logs/v4_textonly.log

echo "== [3/4] GRPO alignment"
bash examples/multimodal_llm/chain_v4_grpo.sh 2>&1 | tee logs/v4_grpo.log

echo "== [4/4] multiple-choice task"
bash examples/multimodal_llm/chain_v4_mc.sh 2>&1 | tee logs/v4_mc.log

echo "== v4 chain done $(date -u +%FT%TZ)"
