# MiniMax H3 on GR308 with SGLang — Design

## Goal

Deploy and benchmark MiniMax H3 through SGLang on server `192.0.2.11`, using the existing Fantasy/GR308 runtime while leaving the established Qwen baseline untouched.

## Scope

The work covers one H3 deployment only. It does not compare H3 with Qwen, change Qwen launch scripts, modify the clean SGLang source tree in place, or present CPU/eager fallback results as GR308 acceleration.

The first target is the locally available `MiniMax-H3-INT4-W4A8` FL2VA checkpoint. A successful result must use GR308 devices for the H3 diffusion pipeline and must generate a valid audio-video file. If a native W4A8 execution path cannot be established, the result is an explicit blocked capability report with reproducible evidence rather than a misleading benchmark.

## Architecture

The existing Qwen Fantasy adapter remains read-only. H3 support is added as a separate overlay mounted into the vendor container at launch time:

1. A Fantasy diffusion platform class implements the SGLang `multimodal_gen` platform contract for device selection, synchronization, attention selection, distributed backend choice, and supported dtypes.
2. A narrow registry overlay adds `fant` to `SGLANG_DIFFUSION_PLATFORM_OVERRIDE` without modifying the upstream SGLang checkout.
3. A W4A8 capability probe identifies the exact GR308 operator used by Comfy Kitchen. The deployment gate requires a native PrivateUse1/Fantasy implementation; the known `_int_mm` fallback is accepted only for correctness diagnosis and is excluded from performance reporting.
4. A dedicated launcher mounts the current SGLang source, H3 dependencies, H3 adapter, checkpoint, and output directory into the pinned Fantasy container image.
5. A benchmark harness records launch configuration, package and image identities, GPU topology, telemetry, request timing, generated media metadata, logs, and checksums in a new H3-only run directory.

## Isolation and Data Safety

- Local implementation lives on branch `codex/h3-sglang-gr308` in an isolated worktree.
- Remote H3 code and outputs use new directories under `/home/USER/levius/` and `/data/infra/benchmarks/`.
- `/home/USER/levius/sglang-upstream`, `/home/USER/levius/sglang-fant-platform`, all Qwen scripts, model files, and vendor libraries are mounted read-only.
- Containers receive unique H3 names and are removed after failed probes. Existing containers are never removed unless their name exactly matches this task's H3 prefix.
- The SSH password is never written to a file, command history, manifest, or log.

## Validation Stages

### 1. Platform contract

Inside the pinned container image, SGLang must report the SRT platform as Fantasy and the diffusion platform as the new Fantasy diffusion class. A tensor allocation, device synchronization, stream query, and one torch-native SDPA call must pass on a GR308 logical device.

### 2. W4A8 operator gate

A representative H3 linear shape is loaded or constructed using the checkpoint metadata. The probe records selected Comfy Kitchen backend, operator dispatch table, elapsed time, output finiteness, and error against a CPU reference. Native dispatch is mandatory before a performance run.

### 3. Model correctness

The FL2VA transformer, Qwen3-VL text encoder, video VAE, and audio VAE must load from local files with strict offline mode. A small deterministic DiT forward must be finite and meet recorded relative-error thresholds. Then a minimal supported H3 request must produce an MP4 containing H.264 video at 24 FPS and stereo audio at 32 kHz.

### 4. H3 performance run

After correctness passes, run a warm-up followed by at least three identical measured requests at concurrency one. Record server-side stage timings, client end-to-end latency, physical-card telemetry, peak device memory, and output checksums. Report median and individual request values. No Qwen data is read or included.

## Failure Handling

Each stage is a gate. A failure captures the exact command, return code, logs, environment manifest, GPU state, and the smallest reproducer. Later performance stages do not run after a correctness or native-dispatch failure. Temporary containers are cleaned up, while evidence files are retained.

## Deliverables

- H3 Fantasy diffusion adapter and automated tests.
- Reproducible H3 launcher and benchmark script.
- One H3-only evidence directory on the server.
- A concise Chinese report stating either verified H3 correctness and performance, or the exact unsupported operator/API blocking native execution.
