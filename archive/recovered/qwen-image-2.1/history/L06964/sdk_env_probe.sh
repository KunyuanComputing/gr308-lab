#!/usr/bin/env bash
set -eu
command -v hipcc || true
hipcc --version 2>/dev/null | head -5 || true
python - <<'PY'
import torch.utils.cpp_extension as e
print('CUDA_HOME', e.CUDA_HOME)
print('ROCM_HOME', e.ROCM_HOME)
print('IS_HIP_EXTENSION', e.IS_HIP_EXTENSION)
PY
