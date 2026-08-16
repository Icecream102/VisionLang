#!/usr/bin/env bash
# Re-run the finalize step once the extras pipeline actually finishes.
# The first chain_finalize timed out after 24h and produced a partial draft;
# this one waits up to 72h and overwrites it with the complete results.
set -u

cd /root/autodl-tmp/multimodal

for i in $(seq 1 4320); do
  if grep -q "mode=extras finished" outputs/v3_extras.log 2>/dev/null; then
    echo "extras pipeline finished after ${i} polls"
    break
  fi
  sleep 60
done

tar -xzf v3_multimodal_llm.tar.gz
/root/miniconda3/bin/python3 -m examples.multimodal_llm.finalize \
  --outputs-dir outputs/v3 --output-md docs/results_v3.md \
  --summary-json outputs/v3/summary.json
echo "finalize2 done"
