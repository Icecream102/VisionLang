#!/usr/bin/env bash
# After the extras pipeline finishes, render the results draft into docs/.
set -u

cd /root/autodl-tmp/multimodal

for i in $(seq 1 1440); do
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
echo "finalize done"
