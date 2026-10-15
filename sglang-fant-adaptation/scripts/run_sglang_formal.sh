#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

RUN_ROOT=${1:?usage: run_sglang_formal.sh RUN_ROOT}
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
FORMAL_ROOT="$RUN_ROOT/sglang/formal"
RAPID_CONTAINER="${RUN_ID}-engine"
ACTIVE_CONTAINER=
TELEMETRY_PID=

status() {
  printf '{"ts":"%s","stage":"%s","status":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >> "$RUN_ROOT/STATUS.events.jsonl"
}

cleanup_case() {
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=
  fi
}

cleanup_error() {
  cleanup_case
  if [[ -n "$ACTIVE_CONTAINER" ]]; then
    docker logs --timestamps "$ACTIVE_CONTAINER" \
      > "$FORMAL_ROOT/last-failed-container.log" 2>&1 || true
    docker stop --time 120 "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
  fi
}
trap cleanup_error ERR INT TERM

validate_case() {
  python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

summary_path, expected, records_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(summary_path, encoding="utf-8") as f:
    summary = json.load(f)
with open(records_path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
assert summary.get("statuses", {}).get("ok") == expected
assert len(rows) == expected
for row in rows:
    assert row.get("status") == "ok"
    assert row.get("input_tokens") == 1024
    assert row.get("output_tokens") == 256
    assert row.get("http_status") == 200
    assert row.get("stream_done") is True
    assert row.get("finish_reason") == "length"
PY
}

mkdir -p "$FORMAL_ROOT"
status formal_schedule freezing
python3 - "$RUN_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
requests = {1: 12, 8: 16, 16: 32, 32: 64, 64: 128}
rapid = {}
budget = {}
for c, n in requests.items():
    data = json.loads((root / "sglang" / "rapid" / f"c{c}" / "summary.json").read_text())
    q = data["raw_completed_output_tokens_per_s"]
    rapid[str(c)] = q
    measurement_s = n * 256 / (0.70 * q)
    budget[str(c)] = {
        "requests": n,
        "estimated_measurement_s": measurement_s,
        "reserved_s": 1.20 * measurement_s + 600,
    }
out = {
    "schema_version": 1,
    "basis": "rapid output throughput with 0.70 shape factor",
    "rapid_output_tokens_per_s": rapid,
    "per_concurrency": budget,
    "repeat_orders": {
        "r1": [1, 8, 64, 16, 32],
        "r3": [16, 32, 8, 64, 1],
        "r5": [64, 1, 32, 8, 16],
    },
}
(root / "analysis" / "formal-time-budget.json").write_text(
    json.dumps(out, indent=2) + "\n", encoding="utf-8"
)
PY
status formal_schedule complete

if docker inspect "$RAPID_CONTAINER" >/dev/null 2>&1; then
  test "$(docker inspect -f '{{ index .Config.Labels "gr308.baseline.run" }}' "$RAPID_CONTAINER")" = "$RUN_ID"
  docker logs --timestamps "$RAPID_CONTAINER" > "$RUN_ROOT/sglang/startup/server-final.log" 2>&1
  docker inspect "$RAPID_CONTAINER" > "$RUN_ROOT/sglang/startup/container-final.json"
  docker stop --time 120 "$RAPID_CONTAINER" > "$RUN_ROOT/sglang/startup/docker-stop.txt"
fi
for _ in $(seq 1 60); do
  if ! ss -ltnH | awk '{print $4}' | grep -Eq ':30000$'; then break; fi
  sleep 2
done
if ss -ltnH | awk '{print $4}' | grep -Eq ':30000$'; then exit 93; fi

status formal running
for repeat in r1 r3 r5; do
  case "$repeat" in
    r1) order='1 8 64 16 32' ;;
    r3) order='16 32 8 64 1' ;;
    r5) order='64 1 32 8 16' ;;
  esac
  repeat_root="$FORMAL_ROOT/$repeat"
  if [[ -f "$repeat_root/.complete" ]]; then
    status "formal_${repeat}" already_complete
    continue
  fi
  if [[ -e "$repeat_root" ]]; then
    mv "$repeat_root" "$FORMAL_ROOT/${repeat}.failed.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  mkdir -p "$repeat_root/startup"
  container="${RUN_ID}-formal-${repeat}"
  ACTIVE_CONTAINER="$container"
  docker rm -f "$container" >/dev/null 2>&1 || true
  status "formal_${repeat}_startup" running
  date +%s%N > "$repeat_root/startup/start-ns.txt"
  docker run -d --name "$container" \
    --label gr308.baseline.run="$RUN_ID" --label gr308.baseline.repeat="$repeat" \
    --runtime=fantasy --privileged --security-opt label=disable --network host \
    --restart=no \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e CUDA_VISIBLE_DEVICES=0,2,4,6 \
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
    -v "$MODEL_PATH":/model:ro \
    -v /data/infra/sglang/gr308-adaptation:/adaptation \
    "$IMAGE" python /fant-platform/launch_sglang_fant.py \
    --model-path /model --served-model-name "$MODEL_NAME" --trust-remote-code \
    --tp-size 4 --dtype bfloat16 --context-length 32768 \
    --mem-fraction-static 0.40 --disable-radix-cache \
    --attention-backend torch_native --language-model-only \
    --host 127.0.0.1 --port 30000 \
    > "$repeat_root/startup/container-id.txt"

  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 3 "$ENDPOINT/health" >/dev/null; then ready=1; break; fi
    test "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" = true || break
    sleep 5
  done
  date +%s%N > "$repeat_root/startup/ready-ns.txt"
  docker inspect "$container" > "$repeat_root/startup/container-ready.json"
  docker logs --timestamps "$container" > "$repeat_root/startup/server-ready.log" 2>&1
  if [[ "$ready" != 1 ]]; then status "formal_${repeat}_startup" failed; exit 20; fi
  curl -fsS "$ENDPOINT/v1/models" > "$repeat_root/startup/models.json"
  status "formal_${repeat}_startup" complete

  for concurrency in $order; do
    case "$concurrency" in
      1) requests=12 ;;
      8) requests=16 ;;
      16) requests=32 ;;
      32) requests=64 ;;
      64) requests=128 ;;
    esac
    case_dir="$repeat_root/c${concurrency}"
    mkdir "$case_dir"
    status "formal_${repeat}_c${concurrency}_warmup" running
    stdbuf -oL fantgpu-smi \
      --query-gpu timestamp,index,pci.bus_id,serial_number,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
      --format csv,noheader,nounits -lms 1000 \
      > "$case_dir/telemetry.csv" 2> "$case_dir/telemetry.err" &
    TELEMETRY_PID=$!
    printf '%s\n' "$TELEMETRY_PID" > "$case_dir/telemetry.pid"

    set +e
    timeout --signal=INT --kill-after=30s 3600s python3 "$CLIENT" \
      --endpoint "$ENDPOINT/v1/completions" --model "$MODEL_NAME" \
      --deployment-id "sglang-upstream-gr308-${repeat}-c${concurrency}-warmup" \
      --input-tokens 1024 --output-tokens 64 \
      --concurrency "$concurrency" --requests "$concurrency" --warmup 0 \
      --prompt-token-id 64 --timeout-s 3600 \
      --output "$case_dir/warmup-requests.jsonl" --summary "$case_dir/warmup-summary.json" \
      > "$case_dir/warmup-client.log" 2>&1
    warmup_rc=$?
    set -e
    printf '%s\n' "$warmup_rc" > "$case_dir/warmup-client.rc"
    if [[ "$warmup_rc" != 0 ]]; then
      cleanup_case; status "formal_${repeat}_c${concurrency}_warmup" failed; exit 40
    fi
    status "formal_${repeat}_c${concurrency}_warmup" complete
    sleep 10

    status "formal_${repeat}_c${concurrency}" running
    set +e
    timeout --signal=INT --kill-after=30s 7200s python3 "$CLIENT" \
      --endpoint "$ENDPOINT/v1/completions" --model "$MODEL_NAME" \
      --deployment-id "sglang-upstream-gr308-${repeat}-c${concurrency}" \
      --input-tokens 1024 --output-tokens 256 \
      --concurrency "$concurrency" --requests "$requests" --warmup 0 \
      --prompt-token-id 64 --timeout-s 7200 \
      --output "$case_dir/requests.jsonl" --summary "$case_dir/summary.json" \
      > "$case_dir/client.log" 2>&1
    case_rc=$?
    set -e
    printf '%s\n' "$case_rc" > "$case_dir/client.rc"
    cleanup_case
    docker logs --timestamps "$container" > "$case_dir/server-through-case.log" 2>&1
    if [[ "$case_rc" != 0 ]] || ! validate_case "$case_dir/summary.json" "$requests" "$case_dir/requests.jsonl"; then
      status "formal_${repeat}_c${concurrency}" failed
      exit 50
    fi
    status "formal_${repeat}_c${concurrency}" complete
  done

  docker logs --timestamps "$container" > "$repeat_root/server-final.log" 2>&1
  docker inspect "$container" > "$repeat_root/container-final.json"
  docker stop --time 120 "$container" > "$repeat_root/docker-stop.txt"
  ACTIVE_CONTAINER=
  touch "$repeat_root/.complete"
  status "formal_${repeat}" complete
done

status formal complete
trap - ERR INT TERM
