#!/usr/bin/env bash
# Wait for the LR sweep to finish, then launch the A+E priority pipeline.
set -u

cd /root/autodl-tmp/multimodal

for i in $(seq 1 180); do
  if grep -q "mode=lr_sweep finished" outputs/v3_lr_sweep.log 2>/dev/null; then
    echo "lr_sweep finished after ${i} polls"
    break
  fi
  sleep 60
done

tar -xzf v3_multimodal_llm.tar.gz
echo "launching ae pipeline"
nohup bash examples/multimodal_llm/run_autodl_v3_pipeline.sh ae \
  > outputs/v3_ae.log 2>&1 &
echo "ae pipeline launched"
