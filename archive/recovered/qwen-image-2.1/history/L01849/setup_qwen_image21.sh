#!/usr/bin/env bash
set -euo pipefail
BASE=/data/infra/qwen-image-2.1-probe
IMAGE=sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc
mkdir -p "$BASE"/{deps,model,output,logs}
cat > "$BASE/requirements.txt" <<'REQ'
git+https://github.com/huggingface/diffusers.git
transformers>=5.17
accelerate
huggingface_hub[hf_xet]
safetensors
sentencepiece
pillow
REQ
docker run --rm --network host \
  -v "$BASE:/work" \
  "$IMAGE" \
  bash -lc 'python3 -m pip install --no-cache-dir --upgrade --target /work/deps -r /work/requirements.txt 2>&1 | tee /work/logs/deps-install.log'
PYTHONPATH="$BASE/deps" python3 - <<'PY' > "$BASE/logs/version-check.txt"
import diffusers, transformers, accelerate, huggingface_hub
print('diffusers', diffusers.__version__)
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
print('huggingface_hub', huggingface_hub.__version__)
from diffusers import QwenImage21Pipeline
print('QwenImage21Pipeline import OK')
PY
