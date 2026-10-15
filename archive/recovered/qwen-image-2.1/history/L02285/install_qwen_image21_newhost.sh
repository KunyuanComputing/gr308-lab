#!/usr/bin/env bash
set -euo pipefail
BASE=/data/infra/qwen-image-2.1-probe-20260921
IMAGE=fantasy/vllm-ftuca:0.18.0-ftuca20260920174921-torch2.7.1-py310-ubuntu22.04-amd64
rm -rf "$BASE/deps"/*
docker run --rm --network host -v "$BASE:/work" "$IMAGE" bash -lc '
set -e
python3 -m pip install --no-cache-dir --no-deps --target /work/deps \
 /work/src/diffusers transformers==5.17.0 accelerate==1.15.0 huggingface_hub==1.32.0 \
 tokenizers==0.23.2 safetensors sentencepiece pillow hf-xet filelock fsspec httpx packaging \
 pyyaml regex requests tqdm typer psutil importlib_metadata typing_extensions numpy==1.26.4 \
 protobuf tiktoken 2>&1 | tee /work/logs/deps-install.log
'
docker run --rm --privileged --network host -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/work/deps -v "$BASE:/work" -v /etc/fant:/etc/fant:ro \
  -v /usr/local/fant:/usr/local/fant:ro -v /usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro \
  "$IMAGE" python3 -c 'import torch,torch_fant,diffusers,transformers,accelerate,huggingface_hub; from diffusers import QwenImage21Pipeline; print("torch",torch.__version__); print("torch_fant",torch_fant.__version__); print("diffusers",diffusers.__version__); print("transformers",transformers.__version__); print("accelerate",accelerate.__version__); print("hub",huggingface_hub.__version__); print("pipeline_import=OK")' | tee "$BASE/logs/version-check.txt"
