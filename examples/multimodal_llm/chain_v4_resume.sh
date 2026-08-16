#!/usr/bin/env bash
# Resume the v4 chain after the GPU switch: GRPO -> MC -> finalize.
# Stages 1 (final eval) and 2 (text-only) are already complete.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
cd "$project_root"
mkdir -p logs

PIDFILE=logs/v4_resume.pid
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "chain_v4_resume already running (pid $(cat "$PIDFILE")); refusing to start a duplicate."
  exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

echo "== v4 resume start $(date -u +%FT%TZ)"
echo "== [1/3] GRPO alignment"
bash examples/multimodal_llm/chain_v4_grpo.sh 2>&1 | tee logs/v4_grpo.log
echo "== [2/3] multiple-choice task"
bash examples/multimodal_llm/chain_v4_mc.sh 2>&1 | tee logs/v4_mc.log
echo "== [3/3] finalize report"
bash examples/multimodal_llm/chain_v4_finalize.sh 2>&1 | tee logs/v4_finalize.log
echo "== v4 resume done $(date -u +%FT%TZ)"
