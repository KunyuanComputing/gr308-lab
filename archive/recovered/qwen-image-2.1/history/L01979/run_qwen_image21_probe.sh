#!/usr/bin/env bash
set -euo pipefail
BASE=/data/infra/qwen-image-2.1-probe
IMAGE=sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc
STEPS=${STEPS:-4}
DTYPE=${DTYPE:-bfloat16}
docker rm -f qwen-image21-probe >/dev/null 2>&1 || true
docker run --name qwen-image21-probe --privileged --network host --shm-size 16g \
  -e CUDA_VISIBLE_DEVICES=0 -e FANTASY_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=8 \
  -e PYTHONPATH=/work/deps -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e STEPS="$STEPS" -e DTYPE="$DTYPE" -e WIDTH=1024 -e HEIGHT=1024 \
  -v "$BASE:/work" \
  -v /usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro \
  -v /usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro \
  -v /usr/local/fant/include:/usr/local/fant/include:ro \
  "$IMAGE" python3 /work/probe.py 2>&1 | tee "$BASE/logs/probe-${DTYPE}-${STEPS}steps.log"
