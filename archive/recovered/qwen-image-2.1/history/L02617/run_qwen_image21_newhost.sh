#!/usr/bin/env bash
set -euo pipefail
BASE=/data/infra/qwen-image-2.1-probe-20260921
IMAGE=fantasy/vllm-ftuca:0.18.0-ftuca20260920174921-torch2.7.1-py310-ubuntu22.04-amd64
STEPS=${STEPS:-4}
DTYPE=${DTYPE:-bfloat16}
docker rm -f qwen-image21-probe >/dev/null 2>&1 || true
docker run --name qwen-image21-probe --privileged --network host --shm-size 16g \
  -e CUDA_VISIBLE_DEVICES=0 -e FANTASY_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=8 \
  -e PYTHONPATH=/work/deps -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e STEPS="$STEPS" -e DTYPE="$DTYPE" -e WIDTH=1024 -e HEIGHT=1024 \
  -v "$BASE:/work" -v /etc/fant:/etc/fant:ro \
  -v /usr/local/fant:/usr/local/fant:ro \
  -v /usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro \
  "$IMAGE" python3 /work/probe.py 2>&1 | tee "$BASE/logs/probe-${DTYPE}-${STEPS}steps.log"
