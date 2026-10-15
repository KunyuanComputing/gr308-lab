#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 14 || $# -gt 30 ]]; then
  echo "usage: $0 NAME DEVICES NUM_GPUS TP SP EXCHANGE_DEGREE CFG OUTDIR MASTER_PORT SCHED_PORT SERVICE_PORT EXCHANGE_MODE VAE_CPU_OFFLOAD ENCODER_PARALLEL [WANVAE_COMPAT_PATH] [VAE_PRECISION] [VAE_DECODE_PRECISION] [VAE_PARALLEL_DECODE_MODE] [FP16_UP_BLOCKS] [FP16_FINAL_RESNETS] [NUM_FRAMES] [NUM_INFERENCE_STEPS] [FLOW_SHIFT] [TIMEOUT_SECONDS] [DIT_LAYERWISE_OFFLOAD] [TEXT_ENCODER_CPU_OFFLOAD] [DIT_OFFLOAD_PREFETCH_SIZE] [ENABLE_SEQUENCE_SHARD] [ATTENTION_BACKEND] [FP16_CONV_ONLY_FINAL_RESNETS]" >&2
  exit 2
fi

name=$1
devices=$2
num_gpus=$3
tp_size=$4
sp_degree=$5
ulysses_degree=$6
cfg_degree=$7
outdir=$8
master_port=$9
scheduler_port=${10}
service_port=${11}
exchange_mode=${12}
vae_cpu_offload=${13}
encoder_parallel=${14}
wanvae_compat_path=${15:-/data/infra/benchmarks/gr308-wan-sglang-20260917/fantasy_wanvae.py}
vae_precision=${16:-fp32}
vae_decode_precision=${17:-bf16}
vae_parallel_decode_mode=${18:-auto}
fp16_up_blocks=${19:-}
fp16_final_resnets=${20:-}
num_frames=${21:-1}
num_inference_steps=${22:-3}
flow_shift=${23:-}
timeout_seconds=${24:-2400}
dit_layerwise_offload=${25:-false}
text_encoder_cpu_offload=${26:-false}
dit_offload_prefetch_size=${27:-0.0}
enable_sequence_shard=${28:-false}
attention_backend=${29:-torch_sdpa}
fp16_conv_only_final_resnets=${30:-}

mixed_env_args=()
if [[ -n "$fp16_up_blocks" ]]; then
  mixed_env_args+=(-e "SGLANG_WAN_VAE_FP16_UP_BLOCKS=$fp16_up_blocks")
fi
if [[ -n "$fp16_final_resnets" ]]; then
  mixed_env_args+=(-e "SGLANG_WAN_VAE_FP16_FINAL_RESNETS=$fp16_final_resnets")
fi
if [[ -n "$fp16_conv_only_final_resnets" ]]; then
  mixed_env_args+=(-e "SGLANG_WAN_VAE_FP16_CONV_ONLY_FINAL_RESNETS=$fp16_conv_only_final_resnets")
fi

flow_shift_args=()
if [[ -n "$flow_shift" ]]; then
  flow_shift_args=(--flow-shift "$flow_shift")
fi

case "$exchange_mode" in
  ulysses)
    exchange_args=(--ulysses-degree "$ulysses_degree" --ring-degree 1)
    ;;
  kv)
    exchange_args=(--kv-gather-degree "$ulysses_degree" --ring-degree 1)
    ;;
  ring)
    exchange_args=(--ulysses-degree 1 --ring-degree "$ulysses_degree")
    ;;
  *)
    echo "EXCHANGE_MODE must be 'ulysses', 'kv', or 'ring'" >&2
    exit 2
    ;;
esac

mkdir -p "$outdir"

sudo docker run -d \
  --name "$name" \
  --privileged \
  --network host \
  --shm-size 4g \
  -e SGLANG_DIFFUSION_PLATFORM_OVERRIDE=fant \
  -e CUDA_VISIBLE_DEVICES="$devices" \
  -e OMP_NUM_THREADS=4 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e PYTORCH_FANT_ALLOC_CONF=max_split_size_mb:64 \
  "${mixed_env_args[@]}" \
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
  -v "$wanvae_compat_path":/src/python/sglang/multimodal_gen/runtime/models/vaes/wanvae.py:ro \
  -v "$outdir":/out \
  -v /usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro \
  -v /data/infra/benchmarks/gr308-h3-20260905/fantasy_platform_registry.py:/src/python/sglang/multimodal_gen/runtime/platforms/__init__.py:ro \
  -v /data/models/Wan2.2-TI2V-5B-Diffusers-local:/model:ro \
  -v /data/infra/benchmarks/gr308-h3-20260905:/work:ro \
  -v /usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro \
  -v /usr/local/fant/include:/usr/local/fant/include:ro \
  sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc \
  timeout --signal=TERM --kill-after=20 "$timeout_seconds" \
  python3 -c 'from sglang.cli.main import main; main()' generate \
  --model-path /model \
  --model-id Wan2.2-TI2V-5B-Diffusers \
  --num-gpus "$num_gpus" \
  --tp-size "$tp_size" \
  --sp-degree "$sp_degree" \
  "${exchange_args[@]}" \
  --cfg-parallel-size "$cfg_degree" \
  --performance-mode manual \
  --dit-layerwise-offload="$dit_layerwise_offload" \
  --text-encoder-cpu-offload="$text_encoder_cpu_offload" \
  --dit-offload-prefetch-size "$dit_offload_prefetch_size" \
  "${flow_shift_args[@]}" \
  --encoder-parallel "$encoder_parallel" \
  --attention-backend "$attention_backend" \
  --vae-cpu-offload="$vae_cpu_offload" \
  --vae-precision "$vae_precision" \
  --vae-decode-precision "$vae_decode_precision" \
  --vae-config.use-parallel-decode true \
  --vae-config.parallel-decode-mode "$vae_parallel_decode_mode" \
  --pin-cpu-memory=false \
  --master-port "$master_port" \
  --scheduler-port "$scheduler_port" \
  --port "$service_port" \
  --prompt 'A small wooden sailboat glides slowly across a calm blue mountain lake. Gentle ripples spread behind the boat. Pine-covered mountains and soft morning sunlight, realistic cinematic footage, steady camera.' \
  --negative-prompt 'Bright tones, overexposed, static, blurred details, subtitles, paintings, still image, low quality, JPEG artifacts, distorted objects, messy background.' \
  --width 1280 \
  --height 704 \
  --num-frames "$num_frames" \
  --num-inference-steps "$num_inference_steps" \
  --enable-sequence-shard="$enable_sequence_shard" \
  --guidance-scale 5 \
  --seed 42 \
  --fps 24 \
  --save-output \
  --output-path /out \
  --perf-dump-path /out/perf.json
