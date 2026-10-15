#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

RUN_ROOT=${1:?usage: run_sglang_baseline_rapid.sh RUN_ROOT}
case "$RUN_ROOT" in
  /data/infra/benchmarks/gr308-sglang-baseline-20??????T??????Z) ;;
  *) echo "unsafe run root: $RUN_ROOT" >&2; exit 90 ;;
esac

CONTAINER="$(basename "$RUN_ROOT")-engine"
MODEL_PATH=/data/models/Qwen3.8-27B
MODEL_NAME=qwen3.8-27b
CLIENT_SRC=/data/infra/benchmarks/gr308-throughput-20260905/closed_loop_safe_client.py
SENTINEL_SRC=/data/infra/benchmarks/gr308-iteration-v2-20260905/run_sentinels.py
CASES_SRC=/data/infra/benchmarks/gr308-iteration-v2-20260905/GR308_Iteration_v2_20260905/sentinel_cases.PROPOSED.json
IMAGE='fantasy/vllm@sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc'
ENDPOINT=http://127.0.0.1:30000

status() {
  printf '{"ts":"%s","stage":"%s","status":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >> "$RUN_ROOT/STATUS.events.jsonl"
}

mkdir -p "$RUN_ROOT"/{manifest,tools/GR308_Iteration_v2_20260905,sglang/{startup,rapid},analysis}
touch "$RUN_ROOT/.gr308-sglang-baseline-root"
status init running

cp -n "$CLIENT_SRC" "$RUN_ROOT/tools/closed_loop_safe_client.py"
cp -n "$SENTINEL_SRC" "$RUN_ROOT/tools/run_sentinels.py"
cp -n "$CASES_SRC" "$RUN_ROOT/tools/sentinel_cases.json"
cp -n /data/infra/benchmarks/gr308-iteration-v2-20260905/GR308_Iteration_v2_20260905/sentinel_screen.py \
  "$RUN_ROOT/tools/GR308_Iteration_v2_20260905/sentinel_screen.py"
sha256sum "$RUN_ROOT"/tools/* > "$RUN_ROOT/manifest/tools.sha256"
sha256sum "$MODEL_PATH"/{config.json,tokenizer.json,tokenizer_config.json,chat_template.jinja,model.safetensors.index.json} \
  > "$RUN_ROOT/manifest/model-code-and-config.sha256"

{
  date -u --iso-8601=seconds
  uname -a
  lscpu
  free -h
  df -h / /data
  fantgpu-smi --query-gpu timestamp,index,pci.bus_id,serial_number,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format csv,noheader,nounits
  docker ps --no-trunc
} > "$RUN_ROOT/manifest/host-before.txt" 2>&1
docker image inspect "$IMAGE" > "$RUN_ROOT/manifest/image.json"

docker rm -f sglang-fant-real-tp4 >/dev/null 2>&1 || true
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
status startup running
date +%s%N > "$RUN_ROOT/sglang/startup/start-ns.txt"
docker run -d --name "$CONTAINER" \
  --label gr308.baseline.run="$(basename "$RUN_ROOT")" \
  --runtime=fantasy --privileged --security-opt label=disable --network host \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e CUDA_VISIBLE_DEVICES=0,2,4,6 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/adaptation/cache/baseline-rapid \
  -v /etc/fant:/etc/fant:ro \
  -v /usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro \
  -v /usr/local/fant/include:/usr/local/fant/include:ro \
  -v /usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro \
  -v /home/USER/levius/sglang-upstream:/sglang:ro \
  -v /home/USER/levius/sglang-fant-platform:/fant-platform:ro \
  -v /home/USER/levius/sglang-runtime-overlay/plugin:/plugin:ro \
  -v /home/USER/levius/sglang-runtime-overlay/site-packages:/sgdeps:ro \
  -v /home/USER/levius/.venvs/sglang:/home/USER/levius/.venvs/sglang:ro \
  -v "$MODEL_PATH":/model:ro \
  -v /data/infra/sglang/gr308-adaptation:/adaptation \
  "$IMAGE" python /fant-platform/launch_sglang_fant.py \
  --model-path /model --served-model-name "$MODEL_NAME" --trust-remote-code \
  --tp-size 4 --dtype bfloat16 --context-length 32768 \
  --mem-fraction-static 0.40 --disable-radix-cache \
  --attention-backend torch_native --language-model-only \
  --host 127.0.0.1 --port 30000 \
  > "$RUN_ROOT/sglang/startup/container-id.txt"

ready=0
for _ in $(seq 1 180); do
  if curl -fsS --max-time 3 "$ENDPOINT/health" >/dev/null; then
    ready=1
    break
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" != true ]]; then
    break
  fi
  sleep 5
done
date +%s%N > "$RUN_ROOT/sglang/startup/ready-ns.txt"
docker inspect "$CONTAINER" > "$RUN_ROOT/sglang/startup/container-ready.json"
docker logs --timestamps "$CONTAINER" > "$RUN_ROOT/sglang/startup/server-ready.log" 2>&1
if [[ "$ready" != 1 ]]; then
  status startup failed
  exit 20
fi
curl -fsS "$ENDPOINT/health" > "$RUN_ROOT/sglang/startup/health.txt"
curl -fsS "$ENDPOINT/v1/models" > "$RUN_ROOT/sglang/startup/models.json"
status startup complete

status correctness_pre running
set +e
python3 "$RUN_ROOT/tools/run_sentinels.py" \
  --endpoint "$ENDPOINT/v1/chat/completions" --model "$MODEL_NAME" \
  --cases "$RUN_ROOT/tools/sentinel_cases.json" \
  --output "$RUN_ROOT/sglang/correctness-pre" --max-tokens 64 --timeout-s 180 \
  > "$RUN_ROOT/sglang/correctness-pre.run.log" 2>&1
sentinel_rc=$?
set -e
printf '%s\n' "$sentinel_rc" > "$RUN_ROOT/sglang/correctness-pre.rc"
if [[ "$sentinel_rc" != 0 ]]; then
  docker logs --timestamps "$CONTAINER" > "$RUN_ROOT/sglang/correctness-pre.server.log" 2>&1
  status correctness_pre failed
  exit 30
fi
status correctness_pre complete

status rapid running
stdbuf -oL fantgpu-smi \
  --query-gpu timestamp,index,pci.bus_id,serial_number,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
  --format csv,noheader,nounits -lms 1000 \
  > "$RUN_ROOT/sglang/rapid/telemetry.csv" 2> "$RUN_ROOT/sglang/rapid/telemetry.err" &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" > "$RUN_ROOT/sglang/rapid/telemetry.pid"
cleanup_telemetry() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup_telemetry EXIT INT TERM

for pair in 1:8 8:16 16:32 32:64 64:128; do
  concurrency=${pair%%:*}
  requests=${pair##*:}
  case_dir="$RUN_ROOT/sglang/rapid/c${concurrency}"
  mkdir "$case_dir"
  status "rapid_c${concurrency}_warmup" running
  set +e
  python3 "$RUN_ROOT/tools/closed_loop_safe_client.py" \
    --endpoint "$ENDPOINT/v1/completions" --model "$MODEL_NAME" \
    --deployment-id "sglang-upstream-gr308-rapid-c${concurrency}-warmup" \
    --input-tokens 128 --output-tokens 64 \
    --concurrency "$concurrency" --requests "$concurrency" --warmup 0 \
    --prompt-token-id 64 --timeout-s 1200 \
    --output "$case_dir/warmup-requests.jsonl" --summary "$case_dir/warmup-summary.json" \
    > "$case_dir/warmup-client.log" 2>&1
  warmup_rc=$?
  set -e
  printf '%s\n' "$warmup_rc" > "$case_dir/warmup-client.rc"
  if [[ "$warmup_rc" != 0 ]]; then
    status "rapid_c${concurrency}_warmup" failed
    exit 40
  fi
  status "rapid_c${concurrency}_warmup" complete
  sleep 10
  curl -fsS --max-time 3 "$ENDPOINT/health" >/dev/null

  status "rapid_c${concurrency}" running
  set +e
  timeout --signal=INT --kill-after=30s 1800s \
    python3 "$RUN_ROOT/tools/closed_loop_safe_client.py" \
    --endpoint "$ENDPOINT/v1/completions" --model "$MODEL_NAME" \
    --deployment-id "sglang-upstream-gr308-rapid-c${concurrency}" \
    --input-tokens 128 --output-tokens 64 \
    --concurrency "$concurrency" --requests "$requests" --warmup 0 \
    --prompt-token-id 64 --timeout-s 1200 \
    --output "$case_dir/requests.jsonl" --summary "$case_dir/summary.json" \
    > "$case_dir/client.log" 2>&1
  case_rc=$?
  set -e
  printf '%s\n' "$case_rc" > "$case_dir/client.rc"
  docker logs --timestamps "$CONTAINER" > "$case_dir/server-through-c${concurrency}.log" 2>&1
  if [[ "$case_rc" != 0 ]]; then
    status "rapid_c${concurrency}" failed
    exit 50
  fi
  status "rapid_c${concurrency}" complete
done

cleanup_telemetry
trap - EXIT INT TERM
docker logs --timestamps "$CONTAINER" > "$RUN_ROOT/sglang/rapid/server-final.log" 2>&1
status rapid complete
status rapid_pipeline complete
