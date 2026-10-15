#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

RUN_ROOT=${1:?usage: continue_sglang_rapid.sh RUN_ROOT}
case "$RUN_ROOT" in
  /data/infra/benchmarks/gr308-sglang-baseline-20??????T??????Z) ;;
  *) echo "unsafe run root: $RUN_ROOT" >&2; exit 90 ;;
esac
test -f "$RUN_ROOT/.gr308-sglang-baseline-root"

CONTAINER="$(basename "$RUN_ROOT")-engine"
ENDPOINT=http://127.0.0.1:30000
MODEL_NAME=qwen3.8-27b
RAPID_ROOT="$RUN_ROOT/sglang/rapid"
CLIENT="$RUN_ROOT/tools/closed_loop_safe_client.py"

status() {
  printf '{"ts":"%s","stage":"%s","status":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >> "$RUN_ROOT/STATUS.events.jsonl"
}

case_is_valid() {
  python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

summary_path, expected, records_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    with open(records_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if summary.get("statuses", {}).get("ok", 0) != expected or len(rows) != expected:
    raise SystemExit(1)
if any(
    row.get("status") != "ok"
    or row.get("input_tokens") != 128
    or row.get("output_tokens") != 64
    or not row.get("stream_done")
    or row.get("finish_reason") != "length"
    for row in rows
):
    raise SystemExit(1)
PY
}

curl -fsS --max-time 3 "$ENDPOINT/health" >/dev/null
test "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" = true
status correctness_pre complete
status rapid running

if [[ -e "$RAPID_ROOT/telemetry.csv" ]]; then
  mv "$RAPID_ROOT/telemetry.csv" "$RAPID_ROOT/telemetry.previous.$(date -u +%Y%m%dT%H%M%SZ).csv"
fi
stdbuf -oL fantgpu-smi \
  --query-gpu timestamp,index,pci.bus_id,serial_number,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
  --format csv,noheader,nounits -lms 1000 \
  > "$RAPID_ROOT/telemetry.csv" 2> "$RAPID_ROOT/telemetry.err" &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" > "$RAPID_ROOT/telemetry.pid"
cleanup() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for pair in 1:8 8:16 16:32 32:64 64:128; do
  concurrency=${pair%%:*}
  requests=${pair##*:}
  case_dir="$RAPID_ROOT/c${concurrency}"

  if [[ -f "$case_dir/client.rc" ]] && [[ "$(cat "$case_dir/client.rc")" = 0 ]] \
      && [[ -f "$case_dir/summary.json" ]] \
      && case_is_valid "$case_dir/summary.json" "$requests" "$case_dir/requests.jsonl"; then
    status "rapid_c${concurrency}" already_complete
    continue
  fi
  if [[ -e "$case_dir" ]]; then
    mv "$case_dir" "$RAPID_ROOT/c${concurrency}.failed.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  mkdir "$case_dir"

  status "rapid_c${concurrency}_warmup" running
  set +e
  python3 "$CLIENT" \
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
    python3 "$CLIENT" \
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
  if [[ "$case_rc" != 0 ]] \
      || ! case_is_valid "$case_dir/summary.json" "$requests" "$case_dir/requests.jsonl"; then
    status "rapid_c${concurrency}" failed
    exit 50
  fi
  status "rapid_c${concurrency}" complete
done

cleanup
trap - EXIT INT TERM
docker logs --timestamps "$CONTAINER" > "$RAPID_ROOT/server-final.log" 2>&1
status rapid complete
status rapid_pipeline complete
