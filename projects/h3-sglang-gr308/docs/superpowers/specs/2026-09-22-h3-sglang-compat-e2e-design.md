# MiniMax H3 on GR308 through SGLang Compatibility Mode — Design

## Goal

Run MiniMax H3 end to end through SGLang on `192.0.2.11` and produce one
valid audio-video MP4. Correctness and observability come first. This phase may
use the already validated eager/FP32-tiled compatibility implementation for
W4A8 operations; it does not claim native GR308 INT8 performance.

## Scope and Success Criteria

The first target is deterministic T2VA generation from the locally available
`MiniMax-H3-INT4-W4A8` checkpoint, using the existing FL2VA configuration at
`/data/infra/benchmarks/gr308-h3-20260905/base-config/FL2VA`. The canonical
smoke request uses short edge 768, aspect ratio 16:9, duration 5 seconds,
24 FPS, one output, and a fixed seed. A run succeeds only when all of the
following are true:

1. SGLang starts with the Fantasy diffusion platform on logical devices
   `0,2,4,6` without changing or stopping the Qwen services on `1,3,5`.
2. The Qwen3-VL text encoder, 50-layer H3 DiT, video VAE, and audio VAE all load
   from the existing local checkpoint in offline mode.
3. A canonical T2VA request completes with finite intermediate tensors.
4. The result is an MP4 with a decodable video stream and a decodable stereo
   32 kHz audio stream; duration and audio-video drift satisfy the H3 request
   contract.
5. The run directory contains the launch command, environment identities,
   server log, request record, GPU telemetry, media metadata, and checksums.

This phase does not publish a throughput or acceleration claim. The existing
5% CPU numerical gate remains recorded as an unresolved quality issue rather
than being silently weakened.

## Architecture

The original SGLang checkout and the existing Qwen Fantasy adapter remain
read-only. New behavior lives in the existing `h3-sglang-fant` overlay and is
mounted into a uniquely named H3 container.

The overlay has four responsibilities:

1. **Fantasy platform registration.** Reuse the existing diffusion platform
   class and drift-checked registry overlay so SGLang selects `fant` without an
   in-place upstream edit.
2. **Compatibility runtime.** Install scoped monkey patches only inside the H3
   process. Reuse the proven GPU-resident W4A8 helpers for row-wise
   quantization, INT4-to-INT8 dequantization, exact bounded FP32-tiled matrix
   multiplication, rounding, and missing `index_fill_` behavior. Original
   functions are restored when the context exits.
3. **Guarded four-device launcher.** Start SGLang on logical devices
   `0,2,4,6`, use FHCCL for distributed collectives, mount all source/model
   inputs read-only, and write only to a new H3 run directory. The launcher
   refuses to start if any selected logical device has a compute process or if
   another owned H3 run holds the lock.
4. **Deterministic request and verifier.** Submit one minimal canonical T2VA
   request with a fixed prompt and seed, poll with a bounded timeout, download
   the result, and validate it with `ffprobe` plus a decode smoke test.

## Device and Parallelism Plan

The service uses four logical devices, one function from each physical GR308
card: `0,2,4,6`. This avoids the active Qwen workers on `1,3,5` and matches the
four-way sequence-parallel structure expected by upstream H3.

The first launch uses `num_gpus=4`, `sp_degree=4`, `ulysses_degree=4`,
`ring_degree=1`, and `tp_size=1`. Torch compile and CUDA-graph paths remain
disabled. Torch SDPA is the only accepted attention backend. If full component
residency exceeds memory, the next permitted change is SGLang's synchronous
layerwise offload for the specific failing component; the degree layout is not
changed in the same experiment.

## Execution Flow

The work proceeds through ordered gates:

1. Validate the four selected devices are free and capture the initial system
   state.
2. Start a disposable container and prove platform selection, tensor
   allocation, synchronization, FHCCL initialization, and Torch SDPA.
3. Start the H3 server and validate each component load boundary separately.
4. Submit a smallest legal fixed-seed T2VA request and record stage timings.
5. Validate the resulting media, archive evidence, and stop only the owned H3
   container.

If a gate fails, later gates do not run. The smallest failing reproducer and
full traceback are retained before the next code change.

## Error Handling and Isolation

- Existing containers and processes are never stopped, killed, renamed, or
  modified.
- The launcher owns only container names beginning with a unique
  `gr308-h3-sglang-compat-` prefix and removes only the exact owned name.
- A trap records logs and GPU state on normal exit, failure, or interruption.
- Unsupported Fantasy-operator warnings become errors during compatibility
  bring-up so CPU fallbacks cannot remain hidden.
- All checkpoint, upstream SGLang, dependency, and vendor-library mounts are
  read-only. New outputs live under a timestamped directory in
  `/data/infra/benchmarks/`.
- The SSH password is never placed in source, scripts, logs, manifests, or
  shell history.

## Testing Strategy

Implementation follows test-first development:

1. Unit tests define compatibility-context installation, restoration, device
   rejection, and counter reporting before the runtime module is written.
2. Launcher contract tests assert exact device selection, unique ownership,
   read-only mounts, lock handling, offline mode, and absence of commands that
   can remove unrelated containers.
3. Request/verifier tests exercise the real H3 payload schema and validate
   representative `ffprobe` JSON, including rejection of missing audio, wrong
   sample rate/channel count, invalid video, excessive drift, and empty files.
4. A container smoke test proves platform/FHCCL/SDPA behavior before model
   loading.
5. The final acceptance test is the real deterministic request and media
   decode on `192.0.2.11`.

## Evidence and Reporting

The run directory records compatibility counters so the generated artifact is
not mistaken for native INT8 execution. The final report states:

- whether SGLang served H3 end to end;
- the exact logical/physical devices used;
- whether every model component loaded;
- request latency and stage timing as diagnostic measurements;
- media stream metadata and checksums;
- that `native_int8_acceleration=false` until a real Fantasy W4A8 kernel is
  introduced;
- the existing CPU-relative audio error and its unresolved 5% gate failure.

Native-kernel optimization and multi-concurrency performance testing are
separate follow-up phases after this compatibility-mode acceptance passes.
