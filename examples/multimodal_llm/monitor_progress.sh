#!/usr/bin/env bash
# Watch for completed v3 runs and analyze each one into outputs/v3/PROGRESS.md.
set -u

cd /root/autodl-tmp/multimodal
python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python3}"

for i in $(seq 1 4320); do
  for log in outputs/v3_smoke.log outputs/v3_lr_sweep.log \
             outputs/v3_ae.log outputs/v3_extras.log; do
    [[ -f "$log" ]] || continue
    grep -a "Completed: outputs/v3/" "$log" | sed 's/.*Completed: //' | while read -r run; do
      if [[ -d "$run" ]] && [[ ! -f "$run/.analyzed" ]]; then
        "$python_bin" -m examples.multimodal_llm.analyze_run \
          --outputs-dir outputs/v3 --run-dir "$run" \
          --progress-md outputs/v3/PROGRESS.md || true
        touch "$run/.analyzed"
      fi
    done
  done
  sleep 120
done
