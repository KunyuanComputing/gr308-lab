#!/usr/bin/env bash
set -euo pipefail

sudo docker run -d \
  --name wan22-sglang-baseline-short \
  --privileged \
  --network host \
  --shm-size 4g \
  -e SGLANG_DIFFUSION_PLATFORM_OVERRIDE=fant \
  -e CUDA_VISIBLE_DEVICES=2,4 \
  -e OMP_NUM_THREADS=4 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e PYTHONPATH=/src/python:/kitchen:/deps:/work \
  -v /data/infra/benchmarks/gr308-h3-20260905/sglang-da76fa073f8e7df4bf30d7049dd4609d9ca3f23c:/src:ro \
  -v /data/infra/benchmarks/gr308-wan-sglang-20260917/fantasy_parallel_state.py:/src/python/sglang/multimodal_gen/runtime/distributed/parallel_state.py:ro \
  -v /data/infra/benchmarks/gr308-wan-sglang-20260917/fantasy_visual_embedding.py:/src/python/sglang/multimodal_gen/runtime/layers/visual_embedding.py:ro \
  -v /data/infra/benchmarks/gr308-wan-sglang-20260917/baseline-short/runtime-fantHipSettings.cfg:/etc/fant/fantHipSettings.cfg:ro \
  -v /data/infra/benchmarks/gr308-h3-20260905/comfy-kitchen-b678fdf63378409676aa5596721445d33794d0ea:/kitchen:ro \
  -v /data/infra/benchmarks/gr308-h3-20260905/runtime-deps:/deps:ro \
  -v /data/infra/benchmarks/gr308-wan-sglang-20260917/fantasy_safetensors_mmap.py:/src/python/sglang/multimodal_gen/runtime/loader/weight_readers/safetensors_mmap.py:ro \
  -v /data/infra/benchmarks/gr308-wan-sglang-20260917/fantasy_t5.py:/src/python/sglang/multimodal_gen/runtime/models/encoders/t5.py:ro \
  -v /data/infra/benchmarks/gr308-wan-sglang-20260917/fantasy_wanvideo.py:/src/python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py:ro \
  -v /data/infra/benchmarks/gr308-wan-sglang-20260917/fantasy_wanvae.py:/src/python/sglang/multimodal_gen/runtime/models/vaes/wanvae.py:ro \
  -v /data/infra/benchmarks/gr308-wan-sglang-20260917/baseline-short-1f:/out \
  -v /usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro \
  -v /data/infra/benchmarks/gr308-h3-20260905/fantasy_platform_registry.py:/src/python/sglang/multimodal_gen/runtime/platforms/__init__.py:ro \
  -v /data/models/Wan2.2-TI2V-5B-Diffusers-local:/model:ro \
  -v /data/infra/benchmarks/gr308-h3-20260905:/work:ro \
  -v /usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro \
  -v /usr/local/fant/include:/usr/local/fant/include:ro \
  sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc \
  timeout --signal=TERM --kill-after=20 1800 \
  python3 -c 'from sglang.cli.main import main; main()' generate \
  --model-path /model \
  --model-id Wan2.2-TI2V-5B-Diffusers \
  --num-gpus 2 \
  --tp-size 2 \
  --cfg-parallel-size 1 \
  --performance-mode manual \
  --attention-backend torch_sdpa \
  --vae-cpu-offload \
  --pin-cpu-memory=false \
  --prompt 'A small wooden sailboat glides slowly across a calm blue mountain lake. Gentle ripples spread behind the boat. Pine-covered mountains and soft morning sunlight, realistic cinematic footage, steady camera.' \
  --negative-prompt 'Bright tones, overexposed, static, blurred details, subtitles, paintings, still image, low quality, JPEG artifacts, distorted objects, messy background.' \
  --width 1280 \
  --height 704 \
  --num-frames 1 \
  --num-inference-steps 3 \
  --guidance-scale 5 \
  --seed 42 \
  --fps 24 \
  --save-output \
  --output-path /out \
  --perf-dump-path /out/perf.json
