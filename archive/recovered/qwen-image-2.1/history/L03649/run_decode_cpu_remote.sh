#!/usr/bin/env bash
set -euo pipefail
BASE=/data/infra/qwen-image-2.1-probe-20260921
IMAGE=fantasy/vllm-ftuca:0.18.0-ftuca20260920174921-torch2.7.1-py310-ubuntu22.04-amd64
docker rm -f qwen-image21-decode-cpu >/dev/null 2>&1 || true
docker run --name qwen-image21-decode-cpu --network host --shm-size 32g \
  -e OMP_NUM_THREADS=64 -e MKL_NUM_THREADS=64 -e PYTHONPATH=/work/deps_cpu:/work/deps \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -v "$BASE:/work" \
  "$IMAGE" python3 /work/decode_qwen_cpu.py
