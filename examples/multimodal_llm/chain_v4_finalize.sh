#!/usr/bin/env bash
# Generate the final results markdown + paired significance after the v4 chain.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/multimodal}"
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
cd "$project_root"

summary=outputs/v4_final_summary.json
grpo_summary=outputs/v4/grpo_qa_clip_seed42/summary.json
grpo_balanced_summary=outputs/v4/grpo_qa_balanced_clip_seed42/summary.json
paired_out=outputs/v4_paired.json
md_out=docs/results_v3_final.md

echo "== finalize start $(date -u +%FT%TZ)"
"$python_bin" -m examples.multimodal_llm.paired_tests \
  --summary "$summary" --output "$paired_out"
GRPO_ARGS=(--grpo-json "$grpo_summary")
if [[ -f "$grpo_balanced_summary" ]]; then
  GRPO_ARGS+=("$grpo_balanced_summary")
fi
"$python_bin" -m examples.multimodal_llm.finalize_v4 \
  --summary "$summary" \
  "${GRPO_ARGS[@]}" \
  --paired-json "$paired_out" \
  --output-md "$md_out"
echo "== wrote $md_out"
echo "== finalize done $(date -u +%FT%TZ)"
