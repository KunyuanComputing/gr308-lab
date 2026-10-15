#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

RUN_ROOT=${1:?usage: run_sglang_post.sh RUN_ROOT}
case "$RUN_ROOT" in
  /data/infra/benchmarks/gr308-sglang-baseline-20??????T??????Z) ;;
  *) echo "unsafe run root: $RUN_ROOT" >&2; exit 90 ;;
esac
test -f "$RUN_ROOT/.gr308-sglang-baseline-root"

RUN_ID=${RUN_ROOT##*/}
MODEL_PATH=/data/models/Qwen3.8-27B
MODEL_NAME=qwen3.8-27b
CLIENT="$RUN_ROOT/tools/closed_loop_safe_client.py"
IMAGE='fantasy/vllm@sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc'
ENDPOINT=http://127.0.0.1:30000
POST_ROOT="$RUN_ROOT/sglang/stability"
LONG_ROOT="$RUN_ROOT/sglang/stability-long-context-1024x256"
CONTAINER="${RUN_ID}-stability"
TELEMETRY_PID=

status() {
  printf '{"ts":"%s","stage":"%s","status":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >> "$RUN_ROOT/STATUS.events.jsonl"
}
cleanup_telemetry() {
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=
  fi
}
cleanup_error() {
  cleanup_telemetry
  docker logs --timestamps "$CONTAINER" > "$POST_ROOT/last-failed-container.log" 2>&1 || true
  docker stop --time 120 "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup_error ERR INT TERM

validate_case() {
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json, sys
summary_path, expected, records_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
expected_input, expected_output = int(sys.argv[4]), int(sys.argv[5])
with open(summary_path, encoding="utf-8") as f: summary=json.load(f)
with open(records_path, encoding="utf-8") as f: rows=[json.loads(x) for x in f if x.strip()]
assert summary.get("statuses",{}).get("ok")==expected
assert len(rows)==expected
for row in rows:
    assert row.get("status")=="ok"
    assert row.get("input_tokens")==expected_input
    assert row.get("output_tokens")==expected_output
    assert row.get("http_status")==200
    assert row.get("stream_done") is True
    assert row.get("finish_reason")=="length"
PY
}

if [[ -f "$POST_ROOT/.complete" ]]; then
  read -r prior_input prior_output < <(python3 - "$POST_ROOT/c64/summary.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
print(x.get("input_tokens"), x.get("output_tokens"))
PY
)
  if [[ "$prior_input" == 1024 && "$prior_output" == 256 ]]; then
    test ! -e "$LONG_ROOT"
    mv "$POST_ROOT" "$LONG_ROOT"
    if [[ -d "$RUN_ROOT/sglang/correctness-post" ]]; then
      test ! -e "$RUN_ROOT/sglang/correctness-post-long-context-1024x256"
      mv "$RUN_ROOT/sglang/correctness-post" "$RUN_ROOT/sglang/correctness-post-long-context-1024x256"
    fi
    status stability_protocol_correction archived_long_context_attempt
  else
    status stability already_complete
    exit 0
  fi
fi
if [[ -e "$POST_ROOT" ]]; then
  mv "$POST_ROOT" "$RUN_ROOT/sglang/stability.failed.$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$POST_ROOT/startup" "$POST_ROOT/c64"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
  if ! ss -ltnH | awk '{print $4}' | grep -Eq ':30000$'; then break; fi
  sleep 2
done
if ss -ltnH | awk '{print $4}' | grep -Eq ':30000$'; then exit 93; fi

status stability_startup running
date +%s%N > "$POST_ROOT/startup/start-ns.txt"
docker run -d --name "$CONTAINER" \
  --label gr308.baseline.run="$RUN_ID" --label gr308.baseline.repeat=stability \
  --runtime=fantasy --privileged --security-opt label=disable --network host --restart=no \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e CUDA_VISIBLE_DEVICES=0,2,4,6 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/adaptation/cache/baseline-formal \
  -v /etc/fant:/etc/fant:ro \
  -v /usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro \
  -v /usr/local/fant/include:/usr/local/fant/include:ro \
  -v /usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro \
  -v /home/USER/levius/sglang-upstream:/sglang:ro \
  -v /home/USER/levius/sglang-fant-platform:/fant-platform:ro \
  -v /home/USER/levius/sglang-runtime-overlay/plugin:/plugin:ro \
  -v /home/USER/levius/sglang-runtime-overlay/site-packages:/sgdeps:ro \
  -v /home/USER/levius/.venvs/sglang:/home/USER/levius/.venvs/sglang:ro \
  -v "$MODEL_PATH":/model:ro -v /data/infra/sglang/gr308-adaptation:/adaptation \
  "$IMAGE" python /fant-platform/launch_sglang_fant.py \
  --model-path /model --served-model-name "$MODEL_NAME" --trust-remote-code \
  --tp-size 4 --dtype bfloat16 --context-length 32768 \
  --mem-fraction-static 0.40 --disable-radix-cache \
  --attention-backend torch_native --language-model-only \
  --host 127.0.0.1 --port 30000 > "$POST_ROOT/startup/container-id.txt"

ready=0
for _ in $(seq 1 180); do
  if curl -fsS --max-time 3 "$ENDPOINT/health" >/dev/null; then ready=1; break; fi
  test "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" = true || break
  sleep 5
done
date +%s%N > "$POST_ROOT/startup/ready-ns.txt"
docker inspect "$CONTAINER" > "$POST_ROOT/startup/container-ready.json"
docker logs --timestamps "$CONTAINER" > "$POST_ROOT/startup/server-ready.log" 2>&1
if [[ "$ready" != 1 ]]; then status stability_startup failed; exit 20; fi
curl -fsS "$ENDPOINT/v1/models" > "$POST_ROOT/startup/models.json"
status stability_startup complete

case_dir="$POST_ROOT/c64"
status stability_c64_warmup running
stdbuf -oL fantgpu-smi \
  --query-gpu timestamp,index,pci.bus_id,serial_number,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
  --format csv,noheader,nounits -lms 1000 > "$case_dir/telemetry.csv" 2> "$case_dir/telemetry.err" &
TELEMETRY_PID=$!
printf '%s\n' "$TELEMETRY_PID" > "$case_dir/telemetry.pid"

set +e
timeout --signal=INT --kill-after=30s 3600s python3 "$CLIENT" \
  --endpoint "$ENDPOINT/v1/completions" --model "$MODEL_NAME" \
  --deployment-id sglang-upstream-gr308-stability-c64-warmup \
  --input-tokens 128 --output-tokens 64 --concurrency 64 --requests 64 --warmup 0 \
  --prompt-token-id 64 --timeout-s 3600 \
  --output "$case_dir/warmup-requests.jsonl" --summary "$case_dir/warmup-summary.json" \
  > "$case_dir/warmup-client.log" 2>&1
warmup_rc=$?
set -e
printf '%s\n' "$warmup_rc" > "$case_dir/warmup-client.rc"
if [[ "$warmup_rc" != 0 ]]; then cleanup_telemetry; status stability_c64_warmup failed; exit 40; fi
status stability_c64_warmup complete
sleep 10

status stability_c64 running
set +e
timeout --signal=INT --kill-after=30s 14400s python3 "$CLIENT" \
  --endpoint "$ENDPOINT/v1/completions" --model "$MODEL_NAME" \
  --deployment-id sglang-upstream-gr308-stability-c64 \
  --input-tokens 128 --output-tokens 64 --concurrency 64 --requests 1024 --warmup 0 \
  --prompt-token-id 64 --timeout-s 14400 \
  --output "$case_dir/requests.jsonl" --summary "$case_dir/summary.json" \
  > "$case_dir/client.log" 2>&1
case_rc=$?
set -e
printf '%s\n' "$case_rc" > "$case_dir/client.rc"
cleanup_telemetry
docker logs --timestamps "$CONTAINER" > "$case_dir/server-through-case.log" 2>&1
if [[ "$case_rc" != 0 ]] || ! validate_case "$case_dir/summary.json" 1024 "$case_dir/requests.jsonl" 128 64; then
  status stability_c64 failed; exit 50
fi
status stability_c64 complete

measurement_duration_s=$(python3 - "$case_dir/summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["measurement_duration_s"])
PY
)
if python3 - "$measurement_duration_s" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) < 3600 else 1)
PY
then
  status stability_c64_extension running
  set +e
  timeout --signal=INT --kill-after=30s 10800s python3 "$CLIENT" \
    --endpoint "$ENDPOINT/v1/completions" --model "$MODEL_NAME" \
    --deployment-id sglang-upstream-gr308-stability-c64-extension \
    --input-tokens 128 --output-tokens 64 --concurrency 64 --requests 512 --warmup 0 \
    --prompt-token-id 64 --timeout-s 10800 \
    --output "$case_dir/extension-requests.jsonl" --summary "$case_dir/extension-summary.json" \
    > "$case_dir/extension-client.log" 2>&1
  extension_rc=$?
  set -e
  printf '%s\n' "$extension_rc" > "$case_dir/extension-client.rc"
  if [[ "$extension_rc" != 0 ]] || ! validate_case "$case_dir/extension-summary.json" 512 "$case_dir/extension-requests.jsonl" 128 64; then
    status stability_c64_extension failed; exit 55
  fi
  status stability_c64_extension complete
else
  printf '{"reason":"main measurement duration >= 3600 s","measurement_duration_s":%s}\n' "$measurement_duration_s" > "$case_dir/extension-skipped.json"
  status stability_c64_extension skipped_duration_met
fi

status correctness_post running
set +e
python3 "$RUN_ROOT/tools/run_sentinels.py" \
  --endpoint "$ENDPOINT/v1/chat/completions" --model "$MODEL_NAME" \
  --cases "$RUN_ROOT/tools/sentinel_cases.json" \
  --output "$RUN_ROOT/sglang/correctness-post" --max-tokens 64 --timeout-s 180 \
  > "$RUN_ROOT/sglang/correctness-post.run.log" 2>&1
sentinel_rc=$?
set -e
printf '%s\n' "$sentinel_rc" > "$RUN_ROOT/sglang/correctness-post.rc"
if [[ "$sentinel_rc" != 0 ]]; then
  docker logs --timestamps "$CONTAINER" > "$RUN_ROOT/sglang/correctness-post.server.log" 2>&1
  status correctness_post failed; exit 60
fi
status correctness_post complete

docker logs --timestamps "$CONTAINER" > "$POST_ROOT/server-final.log" 2>&1
docker inspect "$CONTAINER" > "$POST_ROOT/container-final.json"
docker stop --time 120 "$CONTAINER" > "$POST_ROOT/docker-stop.txt"
touch "$POST_ROOT/.complete"
status stability complete
trap - ERR INT TERM
