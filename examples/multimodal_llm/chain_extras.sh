#!/usr/bin/env bash
# Wait for the A+E pipeline to finish, then run the position-oriented extras:
# LoRA rank ablation, visual-recognition task, and CHAIR hallucination eval.
set -u

cd /root/autodl-tmp/multimodal

for i in $(seq 1 3600); do
  if grep -q "mode=ae finished" outputs/v3_ae.log 2>/dev/null; then
    echo "ae pipeline finished after ${i} polls"
    break
  fi
  sleep 60
done

tar -xzf v3_multimodal_llm.tar.gz
echo "launching extras pipeline"
nohup bash examples/multimodal_llm/run_autodl_v3_pipeline.sh extras \
  > outputs/v3_extras.log 2>&1 &
echo "extras pipeline launched"
