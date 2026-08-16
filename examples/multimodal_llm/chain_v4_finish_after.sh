#!/usr/bin/env bash
# Auto-finisher: waits for the current resume chain to exit, then runs the
# balanced GRPO variant and regenerates the final report with BOTH GRPO runs.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
cd "$project_root"
mkdir -p logs

PIDFILE=logs/v4_finish_after.pid
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "chain_v4_finish_after already running; refusing duplicate."
  exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

echo "== finish_after start $(date -u +%FT%TZ)"
for i in $(seq 1 120); do
  if [[ ! -f logs/v4_resume.pid ]] || ! kill -0 "$(cat logs/v4_resume.pid 2>/dev/null)" 2>/dev/null; then
    echo "resume chain finished after ${i} checks; proceeding"
    break
  fi
  sleep 30
done

echo "== [1/2] balanced GRPO"
bash examples/multimodal_llm/chain_v4_grpo_balanced.sh 2>&1 | tee logs/v4_grpo_balanced.log
echo "== [2/2] finalize (both GRPO runs)"
bash examples/multimodal_llm/chain_v4_finalize.sh 2>&1 | tee logs/v4_finalize.log
echo "== finish_after done $(date -u +%FT%TZ)"
