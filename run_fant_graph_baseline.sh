#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

# Full, resumable FANT Graph32 baseline for Qwen3.8-27B on 4 physical GR308s.
# C64 is outside the frozen graph capture buckets and is reported as stress only.

IMAGE="fantasy/vllm:1.1.0.20-py310-ubuntu22.04"
MODEL_HOST="/data/models/Qwen3.8-27B"
MODEL_NAME="qwen3.8-27b"
VISIBLE_DEVICES="0,2,4,6"
PORT=8000
ENDPOINT="http://127.0.0.1:${PORT}"
SOURCE_BASELINE="/data/infra/benchmarks/gr308-vllm-baseline-20260907T095637Z"
CONCURRENCIES=(1 8 16 32 64)
RAPID_REQUESTS=(8 16 32 64 128)
FORMAL_REQUESTS=(12 16 32 64 128)
FORMAL_REPEATS=(r1 r3 r5)
ADAPTIVE_REPEATS=(r2 r4)
MAX_BLOCK_ATTEMPTS=3
ACTIVE_CONTAINER=""
TELEMETRY_PID=""

timestamp() { date -u +%FT%TZ; }

event() {
  printf '{"timestamp":"%s","stage":"%s","status":"%s"}\n' "$(timestamp)" "$1" "$2" >> "${RUN_ROOT}/STATUS.events.jsonl"
}

cleanup() {
  if [[ -n "${TELEMETRY_PID}" ]]; then
    kill "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
    TELEMETRY_PID=""
  fi
  if [[ -n "${ACTIVE_CONTAINER}" ]]; then
    docker rm -f "${ACTIVE_CONTAINER}" >/dev/null 2>&1 || true
    ACTIVE_CONTAINER=""
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

init_run() {
  local active_link="/data/infra/benchmarks/gr308-vllm-fant-graph-active"
  if [[ -n "${RUN_ROOT:-}" ]]; then
    RUN_ROOT="$(readlink -f -- "${RUN_ROOT}")"
  elif [[ -L "${active_link}" ]]; then
    RUN_ROOT="$(readlink -f -- "${active_link}")"
  else
    RUN_ID="gr308-vllm-fant-graph-$(date -u +%Y%m%dT%H%M%SZ)"
    RUN_ROOT="/data/infra/benchmarks/${RUN_ID}"
    mkdir -p "${RUN_ROOT}"/{manifest,tools,graph32,analysis}
    printf '%s\n' "${RUN_ID}" > "${RUN_ROOT}/.gr308-vllm-fant-graph-root"
    ln -s "${RUN_ROOT}" "${active_link}"
  fi
  RUN_ID="${RUN_ROOT##*/}"
  case "${RUN_ROOT}" in
    /data/infra/benchmarks/gr308-vllm-fant-graph-20??????T??????Z) ;;
    *) printf 'unsafe run root: %s\n' "${RUN_ROOT}" >&2; exit 90 ;;
  esac
  test -f "${RUN_ROOT}/.gr308-vllm-fant-graph-root"
  mkdir -p "${RUN_ROOT}"/{manifest,tools,graph32,analysis}

  if [[ ! -f "${RUN_ROOT}/manifest/run-manifest.json" ]]; then
    printf '{"schema_version":1,"run_id":"%s","framework":"FANT vendor vLLM Graph32","image":"%s","model":"%s","devices":"%s","authorized":true,"desired_post_state":"all-stopped","production_promotion_allowed":false}\n' \
      "${RUN_ID}" "${IMAGE}" "${MODEL_HOST}" "${VISIBLE_DEVICES}" > "${RUN_ROOT}/manifest/run-manifest.json"
  fi
  [[ -f "${RUN_ROOT}/tools/closed_loop_safe_client.py" ]] || cp "${SOURCE_BASELINE}/tools/closed_loop_safe_client.py" "${RUN_ROOT}/tools/"
  [[ -f "${RUN_ROOT}/tools/run_sentinels.py" ]] || cp "${SOURCE_BASELINE}/tools/run_sentinels.py" "${RUN_ROOT}/tools/"
  [[ -f "${RUN_ROOT}/tools/sentinel_cases.json" ]] || cp "${SOURCE_BASELINE}/tools/sentinel_cases.json" "${RUN_ROOT}/tools/"
  [[ -f "${RUN_ROOT}/tools/sentinel_screen.py" ]] || cp "${SOURCE_BASELINE}/tools/GR308_Iteration_v2_20260905/sentinel_screen.py" "${RUN_ROOT}/tools/sentinel_screen.py"
  [[ -f "${RUN_ROOT}/tools/aggregate_graph_baseline.py" ]] || cp /home/USER/levius/aggregate_graph_baseline.py "${RUN_ROOT}/tools/aggregate_graph_baseline.py"
}

preflight() {
  test -d "${MODEL_HOST}"
  test -f "${MODEL_HOST}/config.json"
  test "$(sha256sum "${MODEL_HOST}/config.json" | awk '{print $1}')" = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
  docker image inspect "${IMAGE}" > "${RUN_ROOT}/manifest/image.json"
  if [[ -n "$(docker ps -q)" ]]; then
    printf 'refusing to co-run with another container\n' >&2
    docker ps --format '{{.Names}} {{.Image}} {{.Status}}' >&2
    exit 91
  fi
  if ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
    printf 'port %s is already in use\n' "${PORT}" >&2
    exit 92
  fi
  {
    timestamp
    uname -a
    lscpu
    free -h
    df -h / /data
    fantgpu-smi
  } > "${RUN_ROOT}/manifest/host-before.txt"
  sha256sum "${RUN_ROOT}/tools/"* > "${RUN_ROOT}/manifest/tools.sha256"
  event preflight complete
}

start_server() {
  local block_dir="$1" suffix="$2"
  local startup="${block_dir}/startup"
  mkdir -p "${startup}"
  ACTIVE_CONTAINER="${RUN_ID}-graph-${suffix}"
  docker rm -f "${ACTIVE_CONTAINER}" >/dev/null 2>&1 || true
  printf '%s\n' "${ACTIVE_CONTAINER}" > "${startup}/container-intent.txt"
  date +%s%N > "${startup}/start-ns.txt"

  docker run -d \
    --name "${ACTIVE_CONTAINER}" --hostname user \
    --label "gr308.baseline.run=${RUN_ID}" \
    --privileged --runtime=fantasy --ipc=host --network=host --restart=no \
    --security-opt label=disable --log-opt max-size=100m --log-opt max-file=3 \
    -e "CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v /usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro \
    -v /usr/local/fant/include:/usr/local/fant/include:ro \
    -v /usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro \
    -v "${MODEL_HOST}:/model:ro" \
    "${IMAGE}" \
    vllm serve /model \
      --host 127.0.0.1 --port "${PORT}" --served-model-name "${MODEL_NAME}" \
      --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 32768 \
      --gpu-memory-utilization 0.40 --block-size 800 \
      --language-model-only --trust-remote-code --no-enable-prefix-caching \
      --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
      > "${startup}/container-id.txt"

  local ready=0
  for _ in $(seq 1 540); do
    if curl -fsS --max-time 3 "${ENDPOINT}/health" >/dev/null; then
      ready=1
      break
    fi
    [[ "$(docker inspect -f '{{.State.Running}}' "${ACTIVE_CONTAINER}" 2>/dev/null || true)" == true ]] || break
    sleep 5
  done
  docker logs --timestamps "${ACTIVE_CONTAINER}" > "${startup}/server-ready.log" 2>&1 || true
  [[ "${ready}" -eq 1 ]] || return 1
  date +%s%N > "${startup}/ready-ns.txt"
  curl -fsS "${ENDPOINT}/v1/models" > "${startup}/models.json"
  docker inspect "${ACTIVE_CONTAINER}" > "${startup}/container-ready.json"
  python3 - "${startup}/container-ready.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))[0]
cmd=d['Config']['Cmd']
joined=' '.join(cmd)
assert '--enforce-eager' not in cmd
assert 'FULL_DECODE_ONLY' in joined
assert 'cudagraph_capture_sizes' in joined
assert '[1,2,4,8,16,32]' in joined
assert d['RestartCount'] == 0
PY
}

stop_server() {
  local block_dir="$1"
  if [[ -n "${ACTIVE_CONTAINER}" ]]; then
    docker logs --timestamps "${ACTIVE_CONTAINER}" > "${block_dir}/startup/server-final.log" 2>&1 || true
    docker inspect "${ACTIVE_CONTAINER}" > "${block_dir}/startup/container-final.json" || true
    docker rm -f "${ACTIVE_CONTAINER}" > "${block_dir}/startup/docker-stop.txt" 2>&1 || true
    ACTIVE_CONTAINER=""
  fi
}

run_sentinels() {
  local output="$1"
  python3 "${RUN_ROOT}/tools/run_sentinels.py" \
    --endpoint "${ENDPOINT}/v1/chat/completions" --model "${MODEL_NAME}" \
    --cases "${RUN_ROOT}/tools/sentinel_cases.json" --output "${output}" \
    --max-tokens 64 --timeout-s 90 > "${output}.run.log" 2>&1 || return 1
  python3 - "${output}/summary.json" <<'PY' || return 1
import json,sys
d=json.load(open(sys.argv[1]))
assert d.get('all_cases_pass') is True and d.get('passed') == 12
PY
}

start_telemetry() {
  local output="$1"
  stdbuf -oL fantgpu-smi \
    --query-gpu timestamp,index,pci.bus_id,serial_number,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
    --format=csv,noheader,nounits -lms 1000 > "${output}/telemetry.csv" 2> "${output}/telemetry.err" &
  TELEMETRY_PID=$!
}

stop_telemetry() {
  if [[ -n "${TELEMETRY_PID}" ]]; then
    kill "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
    TELEMETRY_PID=""
  fi
}

validate_case() {
  python3 - "$1/summary.json" "$1/client.rc" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
rc=int(open(sys.argv[2]).read().strip())
assert rc == 0
assert d['statuses'].get('ok', 0) == d['requests']
assert d['raw_completed_output_tokens_per_s'] > 0
PY
}

run_case() {
  local output="$1" phase="$2" repeat="$3" concurrency="$4"
  local input_tokens="$5" output_tokens="$6" requests="$7" timeout_s="$8" warmup_output_tokens="$9"
  mkdir -p "${output}"
  python3 "${RUN_ROOT}/tools/closed_loop_safe_client.py" \
    --endpoint "${ENDPOINT}/v1/completions" --model "${MODEL_NAME}" \
    --deployment-id "fant-graph-${phase}-${repeat}-c${concurrency}-warmup" \
    --input-tokens "${input_tokens}" --output-tokens "${warmup_output_tokens}" \
    --concurrency "${concurrency}" --requests "${concurrency}" --warmup 0 \
    --prompt-token-id 64 --timeout-s "${timeout_s}" \
    --output "${output}/warmup-requests.jsonl" --summary "${output}/warmup-summary.json" \
    > "${output}/warmup-client.log" 2>&1 || return 1
  sleep 5
  curl -fsS --max-time 3 "${ENDPOINT}/health" >/dev/null || return 1
  start_telemetry "${output}"
  curl -fsS "${ENDPOINT}/metrics" > "${output}/metrics-before.prom" || true
  local rc=0
  timeout --signal=INT --kill-after=30s "${timeout_s}s" \
    python3 "${RUN_ROOT}/tools/closed_loop_safe_client.py" \
      --endpoint "${ENDPOINT}/v1/completions" --model "${MODEL_NAME}" \
      --deployment-id "fant-graph-${phase}-${repeat}-c${concurrency}" \
      --input-tokens "${input_tokens}" --output-tokens "${output_tokens}" \
      --concurrency "${concurrency}" --requests "${requests}" --warmup 0 \
      --prompt-token-id 64 --timeout-s "${timeout_s}" \
      --output "${output}/requests.jsonl" --summary "${output}/summary.json" \
      > "${output}/client.log" 2>&1 || rc=$?
  printf '%s\n' "${rc}" > "${output}/client.rc"
  stop_telemetry
  curl -fsS "${ENDPOINT}/metrics" > "${output}/metrics-after.prom" || true
  validate_case "${output}" || return 1
  touch "${output}/case_complete"
}

run_rapid_attempt() {
  local attempt="$1" root="${RUN_ROOT}/graph32/rapid/${attempt}"
  mkdir -p "${root}"
  start_server "${root}" "rapid-${attempt}" || return 1
  run_sentinels "${root}/correctness-pre" || return 1
  for i in "${!CONCURRENCIES[@]}"; do
    local c="${CONCURRENCIES[$i]}" n="${RAPID_REQUESTS[$i]}"
    run_case "${root}/c${c}" rapid rapid "${c}" 128 64 "${n}" 3600 64 || return 1
  done
  stop_server "${root}"
}

run_formal_attempt() {
  local repeat="$1" attempt="$2" root="${RUN_ROOT}/graph32/formal/${repeat}/${attempt}"
  mkdir -p "${root}"
  start_server "${root}" "formal-${repeat}-${attempt}" || return 1
  for i in "${!CONCURRENCIES[@]}"; do
    local c="${CONCURRENCIES[$i]}" n="${FORMAL_REQUESTS[$i]}"
    run_case "${root}/c${c}" formal "${repeat}" "${c}" 1024 256 "${n}" 9000 256 || return 1
  done
  stop_server "${root}"
}

run_stability_attempt() {
  local attempt="$1" root="${RUN_ROOT}/graph32/stability/${attempt}" case_dir
  case_dir="${root}/c64"
  mkdir -p "${root}"
  start_server "${root}" "stability-${attempt}" || return 1
  run_case "${case_dir}" stability stability 64 128 64 1024 9000 64 || return 1
  local duration
  duration="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["measurement_duration_s"])' "${case_dir}/summary.json")"
  if python3 -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) < 3600 else 1)' "${duration}"; then
    local rc=0
    timeout --signal=INT --kill-after=30s 4500s \
      python3 "${RUN_ROOT}/tools/closed_loop_safe_client.py" \
        --endpoint "${ENDPOINT}/v1/completions" --model "${MODEL_NAME}" \
        --deployment-id fant-graph-stability-c64-extension \
        --input-tokens 128 --output-tokens 64 --concurrency 64 --requests 512 \
        --warmup 0 --prompt-token-id 64 --timeout-s 4500 \
        --output "${case_dir}/extension-requests.jsonl" --summary "${case_dir}/extension-summary.json" \
        > "${case_dir}/extension-client.log" 2>&1 || rc=$?
    printf '%s\n' "${rc}" > "${case_dir}/extension-client.rc"
    python3 - "${case_dir}/extension-summary.json" "${case_dir}/extension-client.rc" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); rc=int(open(sys.argv[2]).read())
assert rc == 0 and d['requests'] == 512 and d['statuses'].get('ok',0) == 512
PY
  fi
  run_sentinels "${root}/correctness-post" || return 1
  stop_server "${root}"
}

formal_needs_another_block() {
  python3 - "${RUN_ROOT}" <<'PY'
import json, statistics, sys
from pathlib import Path
root=Path(sys.argv[1])/'graph32'/'formal'
values={c:[] for c in (1,8,16,32,64)}
for repeat in ('r1','r3','r5','r2','r4'):
    selected=root/repeat/'selected'
    if not selected.exists():
        continue
    for c in values:
        d=json.load(open(selected/f'c{c}'/'summary.json'))
        values[c].append(float(d['raw_completed_output_tokens_per_s']))
need=False
for points in values.values():
    if len(points) < 3:
        need=True
    elif statistics.stdev(points)/statistics.mean(points)*100 > 5:
        need=True
raise SystemExit(0 if need else 1)
PY
}

run_block_with_retries() {
  local kind="$1" repeat="${2:-}" parent selected attempt
  if [[ "${kind}" == formal ]]; then
    parent="${RUN_ROOT}/graph32/formal/${repeat}"
  else
    parent="${RUN_ROOT}/graph32/${kind}"
  fi
  mkdir -p "${parent}"
  selected="${parent}/selected"
  [[ -L "${selected}" ]] && return 0
  for index in $(seq 1 "${MAX_BLOCK_ATTEMPTS}"); do
    attempt="attempt-$(printf '%03d' "${index}")"
    [[ -e "${parent}/${attempt}" ]] && continue
    event "${kind}${repeat:+-${repeat}}-${attempt}" started
    if [[ "${kind}" == rapid ]]; then
      if run_rapid_attempt "${attempt}"; then :; else cleanup; event "${kind}-${attempt}" failed; continue; fi
    elif [[ "${kind}" == formal ]]; then
      if run_formal_attempt "${repeat}" "${attempt}"; then :; else cleanup; event "${kind}-${repeat}-${attempt}" failed; continue; fi
    else
      if run_stability_attempt "${attempt}"; then :; else cleanup; event "${kind}-${attempt}" failed; continue; fi
    fi
    ln -s "${attempt}" "${selected}"
    event "${kind}${repeat:+-${repeat}}-${attempt}" complete
    return 0
  done
  printf 'all attempts failed for %s %s\n' "${kind}" "${repeat}" >&2
  return 1
}

finalize() {
  python3 "${RUN_ROOT}/tools/aggregate_graph_baseline.py" "${RUN_ROOT}"
  {
    timestamp
    docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
    fantgpu-smi
    df -h / /data
  } > "${RUN_ROOT}/manifest/host-after.txt"
  event baseline complete
  (cd "${RUN_ROOT}" && find . -type f ! -name FINAL_SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > FINAL_SHA256SUMS)
  rm -f /data/infra/benchmarks/gr308-vllm-fant-graph-active
}

main() {
  init_run
  if [[ ! -f "${RUN_ROOT}/manifest/preflight.complete" ]]; then
    preflight
    touch "${RUN_ROOT}/manifest/preflight.complete"
  fi
  run_block_with_retries rapid
  for repeat in "${FORMAL_REPEATS[@]}"; do
    run_block_with_retries formal "${repeat}"
  done
  for repeat in "${ADAPTIVE_REPEATS[@]}"; do
    if formal_needs_another_block; then
      run_block_with_retries formal "${repeat}"
    fi
  done
  run_block_with_retries stability
  finalize
}

main "$@"
