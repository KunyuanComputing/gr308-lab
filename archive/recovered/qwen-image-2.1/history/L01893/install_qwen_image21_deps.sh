#!/usr/bin/env bash
set -euo pipefail
BASE=/data/infra/qwen-image-2.1-probe
IMAGE=sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc
docker run --rm --network host -v "$BASE:/work" "$IMAGE" bash -lc '
set -e
python3 -m pip install --no-cache-dir --no-deps --target /work/deps \
  /work/src/diffusers transformers==5.17.0 accelerate==1.15.0 huggingface_hub==1.32.0 \
  tokenizers==0.23.2 safetensors sentencepiece pillow hf-xet \
  filelock fsspec httpx packaging pyyaml regex requests tqdm typer psutil \
  importlib_metadata typing_extensions numpy==1.26.4 2>&1 | tee /work/logs/deps-install-clean.log
PYTHONPATH=/work/deps python3 - <<"PY" | tee /work/logs/version-check.txt
import torch, torch_fant
import diffusers, transformers, accelerate, huggingface_hub
print("torch", torch.__version__, "torch_fant", getattr(torch_fant,"__version__","unknown"))
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("huggingface_hub", huggingface_hub.__version__)
from diffusers import QwenImage21Pipeline
print("QwenImage21Pipeline import OK")
PY
'
