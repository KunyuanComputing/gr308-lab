# Qwen3.6-27B vLLM Throughput Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible, throughput-first Qwen3.6-27B tuning workflow for GR308 that compares TP2 BF16 Eager/Graph, MTP, scheduler, FHCCL, and dual-replica profiles without requiring semantic output parity.

**Architecture:** A small Python package owns immutable profile schemas, command construction, streaming benchmark metrics, safe Docker lifecycle, analysis, and reporting. Local unit tests use recorded/fake inputs; server trials run the frozen vendor image and write only to a unique `/data/infra/benchmarks/gr308-qwen36-27b-tuning-<UTC>` evidence root. This plan deliberately ends with a profiling handoff: an FTUCA kernel is designed only after measured evidence identifies the exact Top-1 operation and shapes.

**Tech Stack:** Python 3.11, dataclasses, Pydantic 2, httpx/asyncio, pytest, Docker, vendor vLLM 0.18.0 + vllm_fant 1.1.0.20, torch_fant 1.1.0, Qwen3.6-27B, FANT Graph, FHCCL, JSON/JSONL/CSV.

---

## Scope and frozen identities

- Design source: `docs/superpowers/specs/2026-09-10-qwen36-vllm-throughput-tuning-design.md`.
- Model: `/data/models/Qwen3.6-27B`; no silent substitution.
- Base image digest: `sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc`.
- Physical-device-safe logical sets: TP2 replica A=`0,2`; TP2 replica B=`4,6`; TP4=`0,2,4,6`.
- Primary comparable lane: TP2, BF16, 3072 input tokens, 1024 output tokens, Batch 1 and 16.
- Secondary C8 lane: 1024 input tokens, 1024 output tokens, exactly 8 requests and concurrency 8.
- Saturation lane: 1024 input/output tokens with one wave at C1/C2/C4/C8/C16/C32/C64.
- Accuracy policy: `semantic_accuracy_not_evaluated=true`, `token_consistency_not_required=true`.
- Validity boundary: every successful request must traverse the real model and report exactly the requested completion length; cached, fabricated, empty-model, skipped-layer, or CPU-generated tokens are forbidden.
- Existing baseline directories are read-only and never become output roots.

## File map

Create a focused package instead of adding tuning behavior to the serving API:

```text
python/grserve_tuning/
├── __init__.py          package version
├── schema.py            immutable identities, workloads, profiles and results
├── metrics.py           request and run metric calculations
├── workloads.py         frozen Batch1/16, ModelZoo-C8 and C1-C64 request generation
├── profiles.py          bounded candidate catalog and pruning order
├── vllm_command.py      exact vendor vLLM Docker/CLI construction
├── preflight.py         server/model/device/disk/port safety checks
├── benchmark.py         streaming HTTP client and request-level JSONL
├── runtime.py           owned-container lifecycle and cleanup
├── analyze.py           run aggregation, A/B comparison and selection
├── report.py            Chinese Markdown and machine-readable summary
└── cli.py               render/derive, preflight, trial/block/dual, analyze/report entry points

tests/python/tuning/
├── test_schema.py
├── test_metrics.py
├── test_workloads.py
├── test_profiles.py
├── test_vllm_command.py
├── test_preflight.py
├── test_benchmark.py
├── test_runtime.py
├── test_analyze.py
└── test_report.py

server/qwen36/
├── Dockerfile.profile
└── README.md
```

Do not modify `tools/evidence.py`; its privacy/redaction contract is for the Mac MVP and has a different schema.

### Task 1: Add the tuning package and immutable schema

**Files:**
- Modify: `pyproject.toml`
- Create: `python/grserve_tuning/__init__.py`
- Create: `python/grserve_tuning/schema.py`
- Create: `tests/python/tuning/test_schema.py`

- [ ] **Step 1: Write failing schema tests**

```python
from pydantic import ValidationError
import pytest

from grserve_tuning.schema import ExperimentIdentity, Profile, Workload


def test_primary_workload_requires_frozen_lengths() -> None:
    workload = Workload.primary(batch_size=16)
    assert workload.input_tokens == 3072
    assert workload.output_tokens == 1024
    assert workload.requests == 16
    assert workload.concurrency == 16
    assert workload.semantic_accuracy_not_evaluated is True
    assert workload.token_consistency_not_required is True


def test_profile_rejects_same_physical_gpu_functions() -> None:
    with pytest.raises(ValidationError, match="distinct physical"):
        Profile(id="bad", tp=2, logical_devices=(0, 1))


def test_identity_rejects_wrong_model_path() -> None:
    with pytest.raises(ValidationError, match="Qwen3.6-27B"):
        ExperimentIdentity(model_path="/data/models/Qwen3.8-27B")
```

- [ ] **Step 2: Run the test and verify collection fails**

Run: `uv run pytest tests/python/tuning/test_schema.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'grserve_tuning'`.

- [ ] **Step 3: Register the package and implement the schema**

Change the wheel package list in `pyproject.toml` to:

```toml
[tool.hatch.build.targets.wheel]
packages = ["python/grserve_api", "python/grserve_tuning"]
```

Create `python/grserve_tuning/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `python/grserve_tuning/schema.py` with frozen Pydantic models:

```python
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_PATH = "/data/models/Qwen3.6-27B"
BASE_IMAGE = "sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc"
PHYSICAL_BY_LOGICAL = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExperimentIdentity(FrozenModel):
    model_path: str = MODEL_PATH
    base_image: str = BASE_IMAGE
    served_model_name: str = "qwen3.6-27b"
    semantic_accuracy_not_evaluated: bool = True
    token_consistency_not_required: bool = True

    @model_validator(mode="after")
    def exact_identity(self) -> "ExperimentIdentity":
        if self.model_path != MODEL_PATH:
            raise ValueError("model path must be Qwen3.6-27B")
        if self.base_image != BASE_IMAGE:
            raise ValueError("base image digest mismatch")
        return self


class Workload(FrozenModel):
    id: str
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=1)
    requests: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    temperature: float = 0.0
    ignore_eos: bool = True
    semantic_accuracy_not_evaluated: bool = True
    token_consistency_not_required: bool = True

    @classmethod
    def primary(cls, batch_size: Literal[1, 16]) -> "Workload":
        return cls(id=f"primary-b{batch_size}", input_tokens=3072,
                   output_tokens=1024, requests=batch_size,
                   concurrency=batch_size)

    @classmethod
    def modelzoo_c8(cls) -> "Workload":
        return cls(id="modelzoo-c8", input_tokens=1024,
                   output_tokens=1024, requests=8, concurrency=8)

    @classmethod
    def saturation(cls, concurrency: Literal[1, 2, 4, 8, 16, 32, 64]) -> "Workload":
        return cls(id=f"saturation-c{concurrency}", input_tokens=1024,
                   output_tokens=1024, requests=concurrency,
                   concurrency=concurrency)


class Profile(FrozenModel):
    id: str
    tp: Literal[1, 2, 4] = 2
    logical_devices: tuple[int, ...] = (0, 2)
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    graph: bool = False
    graph_capture_sizes: tuple[int, ...] = ()
    speculative_tokens: Literal[0, 1, 2] = 0
    gpu_memory_utilization: float = Field(default=0.70, gt=0.0, lt=1.0)
    max_num_seqs: int | None = Field(default=None, gt=0)
    max_num_batched_tokens: int | None = Field(default=None, gt=0)
    async_scheduling: bool = False
    enable_chunked_prefill: bool | None = None
    fhccl_channels: int | None = Field(default=None, ge=1, le=30)
    fhccl_buffer_bytes: int | None = Field(default=None, gt=0)
    fhccl_protocol: Literal["LL", "Simple"] | None = None

    @model_validator(mode="after")
    def validate_topology(self) -> "Profile":
        if len(self.logical_devices) != self.tp:
            raise ValueError("device count must equal TP")
        physical = [PHYSICAL_BY_LOGICAL[d] for d in self.logical_devices]
        if len(set(physical)) != len(physical):
            raise ValueError("TP ranks require distinct physical GPUs")
        if self.graph and not self.graph_capture_sizes:
            raise ValueError("graph profile requires capture sizes")
        if not self.graph and self.graph_capture_sizes:
            raise ValueError("eager profile cannot declare graph sizes")
        return self
```

- [ ] **Step 4: Run schema tests**

Run: `uv run pytest tests/python/tuning/test_schema.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml python/grserve_tuning tests/python/tuning/test_schema.py
git commit -m "feat: add Qwen tuning schemas"
```

### Task 2: Implement request and aggregate throughput metrics

**Files:**
- Create: `python/grserve_tuning/metrics.py`
- Create: `tests/python/tuning/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

```python
import pytest

from grserve_tuning.metrics import RequestTiming, summarize_run


def test_decode_tps_excludes_first_token_interval() -> None:
    item = RequestTiming("r1", 0, 1_000_000_000, 11_000_000_000,
                         101, True, expected_output_tokens=101)
    assert item.ttft_ms == 1000.0
    assert item.decode_tps == pytest.approx(10.0)


def test_aggregate_throughput_keeps_failed_time_in_denominator() -> None:
    rows = [
        RequestTiming("ok", 0, 1, 10_000_000_000, 101, True,
                      expected_output_tokens=101),
        RequestTiming("bad", 0, 1, 20_000_000_000, 50, False,
                      expected_output_tokens=101),
    ]
    summary = summarize_run(rows, wall_duration_ns=20_000_000_000)
    assert summary.validated_output_tok_s == pytest.approx(5.05)
    assert summary.completed == 1
    assert summary.failed == 1


def test_incomplete_output_is_not_validated() -> None:
    row = RequestTiming("short", 0, 1, 2, 1023, True,
                        expected_output_tokens=1024)
    assert row.valid is False


def test_run_with_no_valid_requests_fails_closed() -> None:
    rows = [RequestTiming("bad", 0, 1, 2, 0, False,
                          expected_output_tokens=1024)]
    with pytest.raises(ValueError, match="no valid requests"):
        summarize_run(rows, wall_duration_ns=2)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/python/tuning/test_metrics.py -q`

Expected: FAIL because `grserve_tuning.metrics` does not exist.

- [ ] **Step 3: Implement metrics with nanosecond timestamps**

```python
from __future__ import annotations

from dataclasses import dataclass
import statistics


@dataclass(frozen=True)
class RequestTiming:
    request_id: str
    started_ns: int
    first_token_ns: int
    finished_ns: int
    completion_tokens: int
    transport_success: bool
    expected_output_tokens: int

    @property
    def valid(self) -> bool:
        return (self.transport_success and
                self.completion_tokens == self.expected_output_tokens and
                self.started_ns <= self.first_token_ns <= self.finished_ns)

    @property
    def ttft_ms(self) -> float:
        return (self.first_token_ns - self.started_ns) / 1_000_000

    @property
    def decode_tps(self) -> float:
        intervals = self.completion_tokens - 1
        duration = (self.finished_ns - self.first_token_ns) / 1_000_000_000
        return intervals / duration


@dataclass(frozen=True)
class RunMetrics:
    completed: int
    failed: int
    validated_output_tok_s: float
    decode_avg_tps_per_request: float
    median_ttft_ms: float


def summarize_run(rows: list[RequestTiming], wall_duration_ns: int) -> RunMetrics:
    if wall_duration_ns <= 0:
        raise ValueError("wall duration must be positive")
    valid = [row for row in rows if row.valid]
    if not valid:
        raise ValueError("no valid requests")
    return RunMetrics(
        completed=len(valid),
        failed=len(rows) - len(valid),
        validated_output_tok_s=(sum(r.completion_tokens for r in valid) /
                                (wall_duration_ns / 1_000_000_000)),
        decode_avg_tps_per_request=statistics.fmean(r.decode_tps for r in valid),
        median_ttft_ms=statistics.median(r.ttft_ms for r in valid),
    )
```

- [ ] **Step 4: Run metric tests**

Run: `uv run pytest tests/python/tuning/test_metrics.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_tuning/metrics.py tests/python/tuning/test_metrics.py
git commit -m "feat: compute Qwen throughput metrics"
```

### Task 3: Freeze deterministic token-ID workloads

**Files:**
- Create: `python/grserve_tuning/workloads.py`
- Create: `tests/python/tuning/test_workloads.py`

- [ ] **Step 1: Write failing workload tests**

```python
from grserve_tuning.schema import Workload
from grserve_tuning.workloads import completion_payloads


def test_primary_batch_is_one_wave_with_exact_lengths() -> None:
    workload = Workload.primary(16)
    payloads = completion_payloads(workload, model="qwen3.6-27b")
    assert len(payloads) == 16
    assert all(len(p["prompt"]) == 3072 for p in payloads)
    assert all(p["max_tokens"] == 1024 for p in payloads)
    assert all(p["ignore_eos"] is True for p in payloads)
    assert all(p["stream"] is True for p in payloads)
    assert len({tuple(p["prompt"]) for p in payloads}) == 16


def test_modelzoo_c8_is_exactly_eight_requests() -> None:
    payloads = completion_payloads(Workload.modelzoo_c8(), "qwen3.6-27b")
    assert len(payloads) == 8


def test_saturation_workloads_cover_c1_through_c64() -> None:
    levels = (1, 2, 4, 8, 16, 32, 64)
    workloads = [Workload.saturation(c) for c in levels]
    assert [w.id for w in workloads] == [f"saturation-c{c}" for c in levels]
    assert all(w.requests == w.concurrency for w in workloads)
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/python/tuning/test_workloads.py -q`

Expected: FAIL because the workload generator is absent.

- [ ] **Step 3: Implement bounded token-ID generation**

```python
from __future__ import annotations

from typing import Any
from .schema import Workload


def _prompt(length: int, request_index: int) -> list[int]:
    # IDs remain below the model vocabulary and avoid EOS/PAD IDs.
    return [100 + ((request_index * 131 + position) % 1000)
            for position in range(length)]


def completion_payloads(workload: Workload, model: str) -> list[dict[str, Any]]:
    return [
        {
            "model": model,
            "prompt": _prompt(workload.input_tokens, index),
            "max_tokens": workload.output_tokens,
            "temperature": workload.temperature,
            "ignore_eos": workload.ignore_eos,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        for index in range(workload.requests)
    ]
```

- [ ] **Step 4: Run workload tests**

Run: `uv run pytest tests/python/tuning/test_workloads.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_tuning/workloads.py tests/python/tuning/test_workloads.py
git commit -m "feat: freeze Qwen tuning workloads"
```

### Task 4: Define the bounded candidate catalog

**Files:**
- Create: `python/grserve_tuning/profiles.py`
- Create: `tests/python/tuning/test_profiles.py`

- [ ] **Step 1: Write tests for exact candidate order and size**

```python
import pytest

from grserve_tuning.profiles import candidate_profiles, derive_profile, load_profile_file


def test_catalog_starts_with_comparable_baselines() -> None:
    ids = [profile.id for profile in candidate_profiles()]
    assert ids[:2] == ["tp2-bf16-eager-mtp0", "tp2-bf16-graph16-mtp0"]
    assert len(ids) <= 20


def test_every_candidate_uses_distinct_physical_devices() -> None:
    assert all(p.logical_devices in {(0, 2), (0, 2, 4, 6)}
               for p in candidate_profiles())


def test_extreme_lanes_are_not_labelled_comparable() -> None:
    extreme = [p for p in candidate_profiles() if p.dtype == "float16"]
    assert extreme
    assert all(p.id.startswith("extreme-") for p in extreme)


def test_derived_profile_uses_only_allowlisted_overrides(tmp_path) -> None:
    base = candidate_profiles()[1]
    derived = derive_profile(base, "derived-bf16-winner",
                             {"speculative_tokens": 2, "fhccl_channels": 30})
    path = tmp_path / "profile.json"
    path.write_text(derived.model_dump_json(), encoding="utf-8")
    assert load_profile_file(path) == derived
    with pytest.raises(ValueError, match="not derivable"):
        derive_profile(base, "derived-bad", {"logical_devices": [0, 4]})
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/python/tuning/test_profiles.py -q`

Expected: FAIL because the catalog is absent.

- [ ] **Step 3: Implement an explicit, non-combinatorial catalog**

```python
from __future__ import annotations

import json
from pathlib import Path

from .schema import Profile


DERIVABLE_FIELDS = frozenset({
    "graph", "graph_capture_sizes", "speculative_tokens",
    "gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens",
    "async_scheduling", "enable_chunked_prefill",
    "fhccl_channels", "fhccl_buffer_bytes", "fhccl_protocol",
})


def derive_profile(base: Profile, derived_id: str,
                   overrides: dict[str, object]) -> Profile:
    if not derived_id.startswith("derived-"):
        raise ValueError("derived profile id must start with derived-")
    invalid = set(overrides) - DERIVABLE_FIELDS
    if invalid:
        raise ValueError(f"fields are not derivable: {sorted(invalid)}")
    values = base.model_dump()
    values.update(overrides)
    values["id"] = derived_id
    return Profile.model_validate(values)


def load_profile_file(path: Path) -> Profile:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("profile file must be an absolute regular file")
    return Profile.model_validate(json.loads(path.read_text(encoding="utf-8")))


def candidate_profiles() -> tuple[Profile, ...]:
    return (
        Profile(id="tp2-bf16-eager-mtp0"),
        Profile(id="tp2-bf16-graph16-mtp0", graph=True,
                graph_capture_sizes=(1, 2, 4, 8, 16)),
        Profile(id="tp2-bf16-eager-mtp1", speculative_tokens=1),
        Profile(id="tp2-bf16-eager-mtp2", speculative_tokens=2),
        Profile(id="tp2-bf16-graph16-mtp1", graph=True,
                graph_capture_sizes=(1, 2, 4, 8, 16), speculative_tokens=1),
        Profile(id="tp2-bf16-graph16-mtp2", graph=True,
                graph_capture_sizes=(1, 2, 4, 8, 16), speculative_tokens=2),
        Profile(id="tp2-bf16-graph64-mtp0", graph=True,
                graph_capture_sizes=(1, 2, 4, 8, 16, 32, 64)),
        Profile(id="tp2-bf16-graph16-mtp0-mem80", graph=True,
                graph_capture_sizes=(1, 2, 4, 8, 16),
                gpu_memory_utilization=0.80),
        Profile(id="tp2-bf16-graph16-mtp0-seq16", graph=True,
                graph_capture_sizes=(1, 2, 4, 8, 16), max_num_seqs=16),
        Profile(id="tp2-bf16-graph16-mtp0-bt16384", graph=True,
                graph_capture_sizes=(1, 2, 4, 8, 16),
                max_num_batched_tokens=16384),
        Profile(id="tp2-bf16-graph16-mtp0-async", graph=True,
                graph_capture_sizes=(1, 2, 4, 8, 16),
                async_scheduling=True),
        Profile(id="tp2-bf16-graph16-mtp0-chunked", graph=True,
                graph_capture_sizes=(1, 2, 4, 8, 16),
                enable_chunked_prefill=True),
        Profile(id="tp2-bf16-eager-fhccl30", fhccl_channels=30,
                fhccl_buffer_bytes=16 * 1024 * 1024),
        Profile(id="tp2-bf16-eager-fhccl30-simple", fhccl_channels=30,
                fhccl_buffer_bytes=16 * 1024 * 1024,
                fhccl_protocol="Simple"),
        Profile(id="tp2-bf16-eager-fhccl30-ll", fhccl_channels=30,
                fhccl_buffer_bytes=16 * 1024 * 1024,
                fhccl_protocol="LL"),
        Profile(id="tp4-bf16-graph16-mtp0", tp=4,
                logical_devices=(0, 2, 4, 6), graph=True,
                graph_capture_sizes=(1, 2, 4, 8, 16)),
        Profile(id="extreme-tp2-fp16-graph16-mtp0", dtype="float16",
                graph=True, graph_capture_sizes=(1, 2, 4, 8, 16)),
        Profile(id="extreme-tp2-fp16-graph16-mtp2", dtype="float16",
                graph=True, graph_capture_sizes=(1, 2, 4, 8, 16),
                speculative_tokens=2),
    )
```

- [ ] **Step 4: Run catalog tests**

Run: `uv run pytest tests/python/tuning/test_profiles.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_tuning/profiles.py tests/python/tuning/test_profiles.py
git commit -m "feat: define bounded Qwen tuning profiles"
```

### Task 5: Render exact Docker and vLLM commands

**Files:**
- Create: `python/grserve_tuning/vllm_command.py`
- Create: `tests/python/tuning/test_vllm_command.py`

- [ ] **Step 1: Write command rendering tests**

```python
import json

from grserve_tuning.schema import ExperimentIdentity, Profile
from grserve_tuning.vllm_command import docker_command


def test_eager_command_is_digest_pinned_and_text_only() -> None:
    command = docker_command("run-1", ExperimentIdentity(), Profile(id="eager"), 18000)
    joined = " ".join(command)
    assert "sha256:a153d4c" in joined
    assert "--enforce-eager" in command
    assert "--language-model-only" in command
    assert "CUDA_VISIBLE_DEVICES=0,2" in command
    assert "/data/models/Qwen3.6-27B:/model:ro" in command
    assert "--restart=no" in command
    assert "/data/infra/benchmarks/.grserve-qwen36-work/runtime-cache:/cache" in command
    assert "max-size=50m" in command


def test_graph_and_mtp_are_structured_json() -> None:
    profile = Profile(id="g", graph=True,
                      graph_capture_sizes=(1, 2, 4, 8, 16),
                      speculative_tokens=2)
    command = docker_command("run-2", ExperimentIdentity(), profile, 18001)
    compilation = json.loads(command[command.index("--compilation-config") + 1])
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    assert compilation["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert compilation["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16]
    assert speculative == {"method": "qwen3_next_mtp", "num_speculative_tokens": 2}
    assert "--enforce-eager" not in command
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/python/tuning/test_vllm_command.py -q`

Expected: FAIL because command construction is absent.

- [ ] **Step 3: Implement argument-list construction without shell interpolation**

```python
from __future__ import annotations

import json
from .schema import ExperimentIdentity, Profile


def docker_command(run_id: str, identity: ExperimentIdentity,
                   profile: Profile, port: int) -> tuple[str, ...]:
    name = f"qwen36-{run_id}-{profile.id}"
    env = [
        "-e", f"CUDA_VISIBLE_DEVICES={','.join(map(str, profile.logical_devices))}",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "TRANSFORMERS_OFFLINE=1",
        "-e", "HF_HOME=/cache/huggingface",
        "-e", "XDG_CACHE_HOME=/cache/xdg",
        "-e", "TORCH_EXTENSIONS_DIR=/cache/torch_extensions",
    ]
    if profile.fhccl_channels is not None:
        env += ["-e", f"NCCL_MIN_NCHANNELS={profile.fhccl_channels}",
                "-e", f"NCCL_MAX_NCHANNELS={profile.fhccl_channels}"]
    if profile.fhccl_buffer_bytes is not None:
        env += ["-e", f"NCCL_BUFFSIZE={profile.fhccl_buffer_bytes}"]
    if profile.fhccl_protocol is not None:
        env += ["-e", f"NCCL_PROTO={profile.fhccl_protocol}"]

    command = [
        "docker", "run", "-d", "--name", name, "--hostname", "user",
        "--label", f"grserve.qwen36.run={run_id}", "--privileged",
        "--runtime=fantasy", "--ipc=host", "--network=host", "--restart=no",
        "--log-driver", "local", "--log-opt", "max-size=50m", "--log-opt", "max-file=2",
        "--security-opt", "label=disable", *env,
        "-v", "/usr/lib/x86_64-linux-gnu/fantgpu-fh3:/usr/lib/x86_64-linux-gnu/fantgpu-fh3:ro",
        "-v", "/usr/local/fant/include:/usr/local/fant/include:ro",
        "-v", "/usr/local/fant/lib/llvm:/usr/local/fant/lib/llvm:ro",
        "-v", "/data/infra/benchmarks/.grserve-qwen36-work/runtime-cache:/cache",
        "-v", f"{identity.model_path}:/model:ro", identity.base_image,
        "vllm", "serve", "/model", "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", identity.served_model_name,
        "--tensor-parallel-size", str(profile.tp), "--dtype", profile.dtype,
        "--max-model-len", "32768", "--gpu-memory-utilization",
        str(profile.gpu_memory_utilization), "--block-size", "800",
        "--language-model-only", "--trust-remote-code",
        "--no-enable-log-requests", "--no-enable-prefix-caching",
    ]
    if profile.graph:
        command += ["--compilation-config", json.dumps({
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_num_of_warmups": 1,
            "cudagraph_capture_sizes": list(profile.graph_capture_sizes),
        }, separators=(",", ":"))]
    else:
        command.append("--enforce-eager")
    if profile.speculative_tokens:
        command += ["--speculative-config", json.dumps({
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": profile.speculative_tokens,
        }, separators=(",", ":"))]
    if profile.max_num_seqs is not None:
        command += ["--max-num-seqs", str(profile.max_num_seqs)]
    if profile.max_num_batched_tokens is not None:
        command += ["--max-num-batched-tokens", str(profile.max_num_batched_tokens)]
    command.append("--async-scheduling" if profile.async_scheduling
                   else "--no-async-scheduling")
    if profile.enable_chunked_prefill is not None:
        command.append("--enable-chunked-prefill" if profile.enable_chunked_prefill
                       else "--no-enable-chunked-prefill")
    return tuple(command)
```

- [ ] **Step 4: Run command tests**

Run: `uv run pytest tests/python/tuning/test_vllm_command.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_tuning/vllm_command.py tests/python/tuning/test_vllm_command.py
git commit -m "feat: render pinned Qwen vLLM launches"
```

### Task 6: Add fail-closed server preflight

**Files:**
- Create: `python/grserve_tuning/preflight.py`
- Create: `tests/python/tuning/test_preflight.py`

- [ ] **Step 1: Write failing preflight tests**

```python
from grserve_tuning.preflight import ProbeSnapshot, validate_preflight


def healthy() -> ProbeSnapshot:
    return ProbeSnapshot(model_config=True, image_digest=True, running_containers=(),
                         gpu_processes=(), ports_in_use=(), root_free_bytes=9_000_000_000,
                         data_free_bytes=2_000_000_000_000,
                         logical_to_serial={0: "a", 1: "a", 2: "b", 3: "b",
                                            4: "c", 5: "c", 6: "d", 7: "d"})


def test_preflight_accepts_idle_server() -> None:
    assert validate_preflight(healthy(), required_ports=(18000,)) == ()


def test_preflight_rejects_gpu_process_and_busy_port() -> None:
    snapshot = healthy().model_copy(update={"gpu_processes": (123,),
                                            "ports_in_use": (18000,)})
    assert validate_preflight(snapshot, (18000,)) == ("GPU_BUSY", "PORT_BUSY:18000")


def test_preflight_rejects_alias_devices_on_same_card() -> None:
    snapshot = healthy().model_copy(update={"logical_to_serial": {0: "a", 2: "a"}})
    assert "INVALID_DEVICE_TOPOLOGY" in validate_preflight(snapshot, (18000,))


def test_preflight_rejects_low_root_or_data_space() -> None:
    snapshot = healthy().model_copy(update={"root_free_bytes": 1_000_000_000,
                                            "data_free_bytes": 100_000_000_000})
    assert validate_preflight(snapshot, (18000,))[-2:] == (
        "ROOT_SPACE_LOW", "DATA_SPACE_LOW")
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/python/tuning/test_preflight.py -q`

Expected: FAIL because preflight models are absent.

- [ ] **Step 3: Implement pure validation and a probe command manifest**

Implement `ProbeSnapshot` as a frozen Pydantic model and `validate_preflight()` returning ordered
stable reason codes. Add `probe_commands()` returning argument tuples for `docker ps`,
`docker image inspect`, `ss -ltn`, `fantgpu-smi lms`, `df -B1 /`, `df -B1 /data`, model config
existence, and model/hash inventory. The implementation must never call `kill`, `docker stop`,
`rm`, `hipDeviceReset`, or write into a baseline directory.

```python
def validate_preflight(snapshot: ProbeSnapshot,
                       required_ports: tuple[int, ...]) -> tuple[str, ...]:
    reasons: list[str] = []
    if not snapshot.model_config:
        reasons.append("MODEL_MISSING")
    if not snapshot.image_digest:
        reasons.append("IMAGE_MISSING")
    if snapshot.running_containers:
        reasons.append("CONTAINERS_RUNNING")
    if snapshot.gpu_processes:
        reasons.append("GPU_BUSY")
    reasons.extend(f"PORT_BUSY:{p}" for p in required_ports
                   if p in snapshot.ports_in_use)
    serials = [snapshot.logical_to_serial.get(d) for d in (0, 2, 4, 6)]
    if None in serials or len(set(serials)) != 4:
        reasons.append("INVALID_DEVICE_TOPOLOGY")
    if snapshot.root_free_bytes < 4_000_000_000:
        reasons.append("ROOT_SPACE_LOW")
    if snapshot.data_free_bytes < 200_000_000_000:
        reasons.append("DATA_SPACE_LOW")
    return tuple(reasons)
```

- [ ] **Step 4: Run preflight tests**

Run: `uv run pytest tests/python/tuning/test_preflight.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_tuning/preflight.py tests/python/tuning/test_preflight.py
git commit -m "feat: add fail-closed Qwen preflight"
```

### Task 7: Build the streaming benchmark client

**Files:**
- Create: `python/grserve_tuning/benchmark.py`
- Create: `tests/python/tuning/test_benchmark.py`

- [ ] **Step 1: Write async tests against a fake SSE transport**

```python
import pytest

from grserve_tuning.benchmark import parse_sse_events


@pytest.mark.asyncio
async def test_parser_uses_final_usage_and_done() -> None:
    lines = [
        'data: {"choices":[{"text":"a"}]}', "",
        'data: {"choices":[{"text":"","finish_reason":"length"}],'
        '"usage":{"completion_tokens":1024}}', "",
        "data: [DONE]", "",
    ]
    result = await parse_sse_events(lines, expected_tokens=1024)
    assert result.completion_tokens == 1024
    assert result.done_seen is True
    assert result.transport_success is True


@pytest.mark.asyncio
async def test_parser_rejects_missing_usage() -> None:
    result = await parse_sse_events(["data: [DONE]", ""], expected_tokens=1024)
    assert result.transport_success is False
    assert result.error == "MISSING_USAGE"
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/python/tuning/test_benchmark.py -q`

Expected: FAIL because the benchmark client is absent.

- [ ] **Step 3: Implement bounded SSE parsing and one-wave concurrency**

Implement:

```python
@dataclass(frozen=True)
class StreamResult:
    completion_tokens: int
    first_token_ns: int | None
    finished_ns: int
    done_seen: bool
    finish_reason: str | None
    transport_success: bool
    error: str | None


async def run_workload(base_url: str, workload: Workload,
                       model: str, output: Path) -> RunMetrics:
    payloads = completion_payloads(workload, model)
    semaphore = asyncio.Semaphore(workload.concurrency)
    start_gate = asyncio.Event()
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [asyncio.create_task(_one_request(
            client, base_url, payload, str(i), semaphore, start_gate,
            workload.output_tokens)) for i, payload in enumerate(payloads)]
        wall_start = time.monotonic_ns()
        start_gate.set()
        rows = await asyncio.gather(*tasks)
        wall_end = time.monotonic_ns()
    output.write_text("".join(json.dumps(asdict(row), separators=(",", ":")) + "\n"
                              for row in rows), encoding="utf-8")
    return summarize_run(rows, wall_end - wall_start)
```

`_one_request()` must cap any individual SSE line at 1 MiB, record monotonic start/first-nonempty-token/
end (metadata-only chunks do not end TTFT), require exact final usage, `finish_reason="length"`, and
`[DONE]`, sanitize HTTP errors, and never store generated text or prompts. It opens request JSONL with
exclusive creation and flushes every row so interruption preserves completed-request evidence.

- [ ] **Step 4: Run benchmark tests**

Run: `uv run pytest tests/python/tuning/test_benchmark.py -q`

Expected: PASS, including malformed JSON, oversized line, missing `[DONE]`, timeout and HTTP 500 cases.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_tuning/benchmark.py tests/python/tuning/test_benchmark.py
git commit -m "feat: add one-wave streaming benchmark"
```

### Task 8: Own container lifecycle and preserve failed evidence

**Files:**
- Create: `python/grserve_tuning/runtime.py`
- Create: `tests/python/tuning/test_runtime.py`

- [ ] **Step 1: Write lifecycle tests with a fake command runner**

```python
def test_cleanup_targets_only_recorded_container(tmp_path: Path) -> None:
    runner = FakeRunner()
    runtime = TrialRuntime(runner, tmp_path, "run-1", "owned-container")
    runtime.cleanup()
    assert runner.commands == [
        ("docker", "inspect", "owned-container"),
        ("docker", "stop", "--time", "120", "owned-container"),
        ("docker", "rm", "owned-container"),
    ]


def test_start_failure_is_written_before_cleanup(tmp_path: Path) -> None:
    runner = FakeRunner(start_exit=1)
    outcome = run_trial(runner, tmp_path, identity(), profile(), workload())
    assert outcome.status == "START_FAILED"
    assert (tmp_path / "startup" / "docker-run.rc").read_text() == "1\n"
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/python/tuning/test_runtime.py -q`

Expected: FAIL because runtime orchestration is absent.

- [ ] **Step 3: Implement exact ownership, health wait and cleanup**

Implement `CommandRunner` with argument arrays, output SHA256 and bounded capture. `TrialRuntime`
must:

1. create a case directory with mode `0700` and `O_EXCL` marker;
2. write intent before `docker run`;
3. validate `grserve.qwen36.run=<run_id>` after creation;
4. wait at most 45 minutes for `/health`, stopping early if the container exits;
5. collect `docker inspect`, logs, restart count and GPU telemetry;
6. stop only the recorded container after revalidating its label;
7. remove the stopped owned container after inspect/log capture to protect the nearly-full root disk;
8. retain all failed case directories and their captured evidence.

The cleanup implementation must be structurally equivalent to:

```python
def cleanup(self) -> None:
    inspect = self.runner(("docker", "inspect", self.container_name))
    if inspect.exit_code != 0:
        return
    payload = json.loads(inspect.stdout)[0]
    labels = payload["Config"]["Labels"] or {}
    if labels.get("grserve.qwen36.run") != self.run_id:
        raise RuntimeError("CONTAINER_OWNERSHIP_MISMATCH")
    self.runner(("docker", "stop", "--time", "120", self.container_name))
    self.runner(("docker", "rm", self.container_name))
```

- [ ] **Step 4: Run runtime tests**

Run: `uv run pytest tests/python/tuning/test_runtime.py -q`

Expected: PASS for start, readiness timeout, exited container, SIGINT cleanup, ownership mismatch,
and preserved-failure tests.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_tuning/runtime.py tests/python/tuning/test_runtime.py
git commit -m "feat: run owned Qwen tuning trials"
```

### Task 9: Implement candidate pruning and A/B analysis

**Files:**
- Create: `python/grserve_tuning/analyze.py`
- Create: `tests/python/tuning/test_analyze.py`

- [ ] **Step 1: Write selection tests**

```python
from grserve_tuning.analyze import CandidateResult, compose_selected_profile, select_candidates
from grserve_tuning.profiles import candidate_profiles


def test_candidate_below_five_percent_is_pruned() -> None:
    baseline = CandidateResult("base", (100.0, 101.0, 99.0), 0)
    weak = CandidateResult("weak", (103.0, 104.0, 102.0), 0)
    assert select_candidates(baseline, [weak]) == ()


def test_failed_candidate_never_wins() -> None:
    baseline = CandidateResult("base", (100.0, 100.0, 100.0), 0)
    fast = CandidateResult("fast", (140.0, 141.0, 139.0), 1)
    assert select_candidates(baseline, [fast]) == ()


def test_selection_orders_by_geomean_gain() -> None:
    baseline = CandidateResult("base", (100.0, 100.0, 100.0), 0)
    a = CandidateResult("a", (115.0, 116.0, 114.0), 0)
    b = CandidateResult("b", (130.0, 129.0, 131.0), 0)
    assert [x.id for x in select_candidates(baseline, [a, b])] == ["b", "a"]


def test_composition_copies_only_category_fields() -> None:
    catalog = {p.id: p for p in candidate_profiles()}
    combined = compose_selected_profile(
        catalog["tp2-bf16-graph16-mtp0"],
        mtp=catalog["tp2-bf16-graph16-mtp2"],
        graph_scheduler=catalog["tp2-bf16-graph64-mtp0"],
        fhccl=catalog["tp2-bf16-eager-fhccl30"],
    )
    assert combined.id == "derived-bf16-selected"
    assert combined.speculative_tokens == 2
    assert combined.graph_capture_sizes[-1] == 64
    assert combined.fhccl_channels == 30
    assert combined.dtype == "bfloat16"
    assert combined.logical_devices == (0, 2)
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/python/tuning/test_analyze.py -q`

Expected: FAIL because analysis is absent.

- [ ] **Step 3: Implement statistics and fail-closed selection**

`CandidateResult` contains profile ID, three validated-throughput values and total failures.
`select_candidates()` rejects fewer than three blocks, any failure, CV above 10%, or geometric mean
gain below 5%. It returns remaining candidates ordered by geometric mean gain. Add helpers for mean,
median, sample standard deviation, CV and descriptive t interval; never combine different workload,
dtype, TP, model hash or execution mode.

`compose_selected_profile()` starts from `tp2-bf16-graph16-mtp0` and copies only the winning
category's fields in fixed order: MTP=`speculative_tokens`; Graph/Scheduler=`graph`, capture sizes,
memory utilization, sequence/token budgets, async scheduling and chunked prefill; FHCCL=channel,
buffer and protocol. It calls
`derive_profile()`, writes canonical `analysis/derived-candidate.json`, and emits one ablation profile
after each added category. It never copies dtype, TP or logical devices, so an extreme/TP4 result
cannot leak into the comparable profile.

```python
def geometric_gain(candidate: CandidateResult,
                   baseline: CandidateResult) -> float:
    ratios = [c / b for c, b in zip(candidate.values, baseline.values, strict=True)]
    return math.prod(ratios) ** (1 / len(ratios)) - 1.0
```

- [ ] **Step 4: Run analysis tests**

Run: `uv run pytest tests/python/tuning/test_analyze.py -q`

Expected: PASS for identity mismatch, n<3, failures, high variance, weak gain and ordered selection.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_tuning/analyze.py tests/python/tuning/test_analyze.py
git commit -m "feat: analyze Qwen tuning candidates"
```

### Task 10: Generate machine-readable and Chinese reports

**Files:**
- Create: `python/grserve_tuning/report.py`
- Create: `tests/python/tuning/test_report.py`

- [ ] **Step 1: Write report contract tests**

```python
def test_report_discloses_accuracy_policy(tmp_path: Path) -> None:
    write_report(tmp_path, fixture_analysis())
    summary = json.loads((tmp_path / "summary.json").read_text())
    report = (tmp_path / "REPORT_ZH.md").read_text()
    assert summary["semantic_accuracy_not_evaluated"] is True
    assert summary["token_consistency_not_required"] is True
    assert "未评估输出语义准确性" in report
    assert "单请求 Decode TPS" in report
    assert "聚合 output tok/s" in report


def test_report_keeps_bf16_and_extreme_lanes_separate(tmp_path: Path) -> None:
    write_report(tmp_path, fixture_analysis())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert set(summary["lanes"]) == {"bf16-comparable", "extreme-throughput"}
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/python/tuning/test_report.py -q`

Expected: FAIL because report generation is absent.

- [ ] **Step 3: Implement report rendering**

The JSON schema must include identities, workload formulae, all failed/excluded attempts, raw values,
statistics, feature flags, profile diffs, MTP acceptance fields when available, and the two explicit
accuracy-policy booleans. The Markdown report must lead with the measured outcome, distinguish
per-request decode TPS from aggregate output tok/s, and say `NOT_EVALUATED` instead of `PASS` for
semantic accuracy and token consistency.

```python
POLICY_NOTE = (
    "本轮未评估输出语义准确性，也不要求优化前后 token 一致；"
    "吞吐有效性仅保证真实模型路径、请求完成和约定输出长度。"
)
```

- [ ] **Step 4: Run report tests**

Run: `uv run pytest tests/python/tuning/test_report.py -q`

Expected: `2 passed` plus JSON-schema edge cases.

- [ ] **Step 5: Commit**

```bash
git add python/grserve_tuning/report.py tests/python/tuning/test_report.py
git commit -m "feat: report throughput-first Qwen results"
```

### Task 11: Add the CLI and immutable deployment bundle

**Files:**
- Modify: `pyproject.toml`
- Create: `python/grserve_tuning/cli.py`
- Create: `server/qwen36/Dockerfile.profile`
- Create: `server/qwen36/README.md`
- Create: `tests/python/tuning/test_cli.py`

- [ ] **Step 1: Write CLI tests**

```python
def test_render_profile_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["render", "--profile", "tp2-bf16-eager-mtp0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"]["id"] == "tp2-bf16-eager-mtp0"
    assert payload["identity"]["model_path"] == "/data/models/Qwen3.6-27B"


def test_unknown_profile_fails_without_fallback() -> None:
    with pytest.raises(SystemExit):
        main(["render", "--profile", "missing"])


def test_profile_source_is_mutually_exclusive(tmp_path) -> None:
    profile_file = tmp_path / "profile.json"
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "render", "--profile", "tp2-bf16-eager-mtp0",
            "--profile-file", str(profile_file.resolve()),
        ])


def test_block_preserves_requested_workload_order() -> None:
    args = build_parser().parse_args([
        "block", "--run-root",
        "/data/infra/benchmarks/gr308-qwen36-27b-tuning-20260910T120000Z",
        "--profile", "tp2-bf16-eager-mtp0", "--port", "18000",
        "--phase", "baseline", "--repeat", "r1",
        "--workloads", "primary-b1", "modelzoo-c8", "primary-b16",
    ])
    assert args.workloads == ["primary-b1", "modelzoo-c8", "primary-b16"]


def test_dual_lane_uses_disjoint_device_sets() -> None:
    args = build_parser().parse_args([
        "dual", "--run-root",
        "/data/infra/benchmarks/gr308-qwen36-27b-tuning-20260910T120000Z",
        "--profile", "tp2-bf16-graph16-mtp0", "--ports", "18000", "18001",
        "--workload", "modelzoo-c8", "--repeat", "r1",
    ])
    assert args.ports == [18000, 18001]
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/python/tuning/test_cli.py -q`

Expected: FAIL because the CLI is absent.

- [ ] **Step 3: Implement explicit subcommands**

Add to `pyproject.toml`:

```toml
[project.scripts]
grserve-api = "grserve_api.main:main"
grserve-qwen36-tune = "grserve_tuning.cli:entrypoint"
```

The CLI has `render`, `derive`, `bundle-profile`, `preflight`, `trial`, `block`, `dual`, `analyze`,
and `report` subcommands. Every command that resolves a profile accepts exactly one of `--profile`
(a catalog ID) and `--profile-file` (an absolute, non-symlink JSON file validated by `Profile`).
`derive` accepts a catalog/file base plus repeatable `--set FIELD=JSON_VALUE` arguments, permits only
`DERIVABLE_FIELDS`, requires an ID beginning `derived-`, and atomically writes the validated result.

`trial`, `block`, and `dual` require an absolute evidence root matching
`^/data/infra/benchmarks/gr308-qwen36-27b-tuning-[0-9]{8}T[0-9]{6}Z$`; it rejects symlinks,
existing non-owned directories, baseline prefixes, ports outside 18000-18031, and unknown profiles.
`block` starts one engine-cold container, waits for readiness, runs its ordered workload list against
that same engine, snapshots telemetry between workloads, then performs owned cleanup. `dual` creates
replica A on `0,2` and replica B on `4,6`, requires two different ports, round-robins an even request
count, records both container identities, and cleans up both even after partial failure. Neither
command silently retries or overwrites an existing attempt directory.

Create `Dockerfile.profile`:

```dockerfile
FROM fantasy/vllm:1.1.0.20-py310-ubuntu22.04
ARG PROFILE_SHA256
LABEL org.grserve.qwen36.base-image-id=sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc
LABEL org.grserve.qwen36.profile-sha256=$PROFILE_SHA256
COPY profile.json /opt/grserve-qwen36/profile.json
```

`bundle-profile` creates a new output directory containing exactly this Dockerfile and a canonical
`profile.json`, records both SHA256 values, and refuses an existing or symlinked destination. The
build path first proves the local tag resolves to the frozen image ID and uses `--pull=false`; a
mismatch aborts before `docker build`. The derived image is optional until a winning profile exists;
trials continue to use the pinned base digest so a profile image cannot change runtime behavior
accidentally.

- [ ] **Step 4: Run CLI and full Python tests**

Run: `uv run pytest tests/python/tuning -q && uv run ruff check python/grserve_tuning tests/python/tuning && uv run mypy python/grserve_tuning`

Expected: all tuning tests pass, Ruff reports no errors, mypy reports success.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml python/grserve_tuning/cli.py server/qwen36 tests/python/tuning/test_cli.py
git commit -m "feat: package Qwen tuning workflow"
```

### Task 12: Export, deploy and run the exact server baseline

**Files:**
- Modify only if a deployment defect is found: `python/grserve_tuning/*`
- Create on server: `/data/infra/benchmarks/.grserve-qwen36-work/src/<git-sha>/`
- Create on server: `/data/infra/benchmarks/gr308-qwen36-27b-tuning-<UTC>/`

- [ ] **Step 1: Verify the local source is clean and tested**

Run:

```bash
uv sync --frozen --all-groups
uv run pytest tests/python/tuning -q
uv run ruff check python/grserve_tuning tests/python/tuning
uv run mypy python/grserve_tuning
git status --porcelain --untracked-files=all
```

Expected: all checks pass and status is empty.

- [ ] **Step 2: Export a provenance-bound archive**

Use Git's tracked-file archive so uncommitted files and local secrets cannot enter the server bundle.

Run:

```bash
git archive --format=tar.gz --output=/tmp/grserve-qwen36-source.tgz HEAD
git rev-parse HEAD
shasum -a 256 /tmp/grserve-qwen36-source.tgz
```

Expected: the full commit SHA and archive SHA256 are printed. Record both in the run-root ownership
manifest before the first container starts.

- [ ] **Step 3: Copy to a new server source directory**

Create exactly `/data/infra/benchmarks/.grserve-qwen36-work/src/<full-sha>`, copy the archive, verify
its SHA256 on the server, and extract it there. Refuse if the directory already exists with a
different archive hash. Create `/data/infra/benchmarks/.grserve-qwen36-work/uv-cache`,
`/data/infra/benchmarks/.grserve-qwen36-work/tmp`, and
`/data/infra/benchmarks/.grserve-qwen36-work/runtime-cache`; record their paths in the run manifest.
Run the following in the immutable source directory:

```bash
UV_CACHE_DIR=/data/infra/benchmarks/.grserve-qwen36-work/uv-cache \
TMPDIR=/data/infra/benchmarks/.grserve-qwen36-work/tmp \
  /home/USER/levius/.bootstrap/bin/uv sync --frozen --all-groups
```

Do not deploy or cache under `/home/USER/levius` or `/`; the root filesystem has only about 4.5 GiB
free. Do not build the optional derived image unless preflight still reports at least 4 GB free.

- [ ] **Step 4: Run server preflight**

Run from the immutable source directory:

```bash
/home/USER/levius/.bootstrap/bin/uv run grserve-qwen36-tune preflight \
  --ports 18000 18001 18002 18003
```

Expected: JSON status `PASS`, zero running containers/GPU processes, four distinct physical serials,
model and image digest present, `/data` has at least 200 GB free.

- [ ] **Step 5: Run one quick qualification per baseline mode**

Create a unique evidence root, then run:

```bash
RUN_ROOT="/data/infra/benchmarks/gr308-qwen36-27b-tuning-$(date -u +%Y%m%dT%H%M%SZ)"
/home/USER/levius/.bootstrap/bin/uv run grserve-qwen36-tune trial \
  --run-root "$RUN_ROOT" --profile tp2-bf16-eager-mtp0 \
  --workload primary-b1 --port 18000 --phase qualification --repeat q1
/home/USER/levius/.bootstrap/bin/uv run grserve-qwen36-tune trial \
  --run-root "$RUN_ROOT" --profile tp2-bf16-graph16-mtp0 \
  --workload primary-b1 --port 18000 --phase qualification --repeat q1
```

Expected: each returns 1024 completion tokens, request success 1/1, no restart/OOM/FHCCL fatal,
and owned container cleanup leaves zero running containers and GPU processes.

- [ ] **Step 6: Run three baseline blocks**

For each of `tp2-bf16-eager-mtp0` and `tp2-bf16-graph16-mtp0`, run `primary-b1`,
`primary-b16`, and `modelzoo-c8` in block order `B1,C8,B16` for R1, `B16,B1,C8` for R2,
and `C8,B16,B1` for R3. Use a new engine-cold container per block, not per workload.

Use `block` six times, with the following exact workload suffixes; every invocation also includes
`--run-root "$RUN_ROOT" --port 18000 --phase baseline`:

```text
--profile tp2-bf16-eager-mtp0   --repeat r1 --workloads primary-b1 modelzoo-c8 primary-b16
--profile tp2-bf16-eager-mtp0   --repeat r2 --workloads primary-b16 primary-b1 modelzoo-c8
--profile tp2-bf16-eager-mtp0   --repeat r3 --workloads modelzoo-c8 primary-b16 primary-b1
--profile tp2-bf16-graph16-mtp0 --repeat r1 --workloads primary-b1 modelzoo-c8 primary-b16
--profile tp2-bf16-graph16-mtp0 --repeat r2 --workloads primary-b16 primary-b1 modelzoo-c8
--profile tp2-bf16-graph16-mtp0 --repeat r3 --workloads modelzoo-c8 primary-b16 primary-b1
```

Expected: 18 run summaries, all requests complete, every output length is 1024, and baseline CV is
recorded before tuning.

- [ ] **Step 7: Commit any reproducibility-only fix**

If no defect is found, make no commit. If a defect is found, add a failing local test, implement the
minimal fix, rerun Task 11 checks, commit `fix: harden Qwen tuning deployment`, export a new SHA, and
use a new immutable server source directory. Never edit the deployed source in place.

### Task 13: Run bounded MTP, Graph/Scheduler and FHCCL sweeps

**Files:**
- Server evidence only under the active unique run root
- Modify source only through the defect procedure in Task 12 Step 7

- [ ] **Step 1: Execute the two-day MTP gate**

Run quick `primary-b1` and `primary-b16` trials in this exact order:

```text
tp2-bf16-eager-mtp1
tp2-bf16-eager-mtp2
tp2-bf16-graph16-mtp1
tp2-bf16-graph16-mtp2
```

For each profile, first run one `qualification` repeat for both workloads. If both complete, run
three `screening` repeats of `primary-b16`, always supplying `--run-root "$RUN_ROOT"`, a unique
`--repeat`, and port 18000. The unmodified Eager or Graph profile is interleaved before every MTP
profile so the three gain values are paired with same-period baselines.

For every lane capture ready status, memory, proposal/accepted token counters when exported, output
length and throughput. Stop MTP expansion after two working days or on repeatable initialization
failure; do not substitute MTP3 because the model has one MTP layer and recommends two speculative
tokens.

- [ ] **Step 2: Prune MTP candidates**

Candidates with request failure, incomplete output, restart/OOM, CV>10%, or gain<5% do not receive a
formal run. Semantic mismatch is not a pruning reason. Record `NOT_EXPOSED` rather than inventing
acceptance metrics when the runtime does not export them.

- [ ] **Step 3: Execute Graph/Scheduler single-variable trials**

Compare the exact catalog profiles `tp2-bf16-graph64-mtp0`,
`tp2-bf16-graph16-mtp0-mem80`, `tp2-bf16-graph16-mtp0-seq16`, and
`tp2-bf16-graph16-mtp0-bt16384`, `tp2-bf16-graph16-mtp0-async`, and
`tp2-bf16-graph16-mtp0-chunked` against `tp2-bf16-graph16-mtp0`. Run three interleaved
`primary-b16` screening repeats first because it is the throughput target; only candidates passing
the 5% gate proceed to Batch1 and C8.

- [ ] **Step 4: Execute FHCCL trials**

Compare `tp2-bf16-eager-mtp0`, `tp2-bf16-eager-fhccl30`,
`tp2-bf16-eager-fhccl30-simple`, and `tp2-bf16-eager-fhccl30-ll` under Eager MTP-off so Graph/MTP do
not confound communication. Use three interleaved `primary-b16` screening repeats. Capture
`NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,COLL,P2P,GRAPH,TUNING` for qualification only; formal
runs restore normal logging.

- [ ] **Step 5: Execute the TP4 diagnostic and FP16 extreme lanes**

Run TP4 BF16 only as a diagnostic. Run FP16 profiles in the separate `extreme-throughput` lane.
Neither may replace the TP2 BF16 baseline or be described as an equal-precision win.

- [ ] **Step 6: Select at most one winner per category**

Use `grserve-qwen36-tune analyze` to select no more than one MTP, one Graph/Scheduler and one FHCCL
candidate. The command writes `analysis/derived-candidate.json` plus incremental ablation profiles,
using the fixed category-field mapping from Task 9. Run the ablations via `--profile-file`; retain a
feature only after three paired `primary-b16` repeats still clear the 5% gate. Do not run the Cartesian
product.

- [ ] **Step 7: Measure the C1-C64 saturation curve**

Run `saturation-c1`, `-c2`, `-c4`, `-c8`, `-c16`, `-c32`, and `-c64` once for the unmodified
Graph baseline and once for the combined finalist. Then repeat both curves three times only after the
finalist clears all earlier gates. Stop a curve after OOM, restart, request failure, or monotonic
throughput collapse at two successive higher concurrency points; record unrun higher points as
`PRUNED_AFTER_FAILURE` rather than zero. This curve selects the recommended production concurrency
but does not replace the frozen Batch1/16 or ModelZoo-C8 acceptance results.

### Task 14: Formal A/B, dual replica and hotspot handoff

**Files:**
- Create through CLI: `<run-root>/analysis/summary.json`
- Create through CLI: `<run-root>/report/REPORT_ZH.md`
- Create through CLI: `<run-root>/analysis/hotspot-handoff.json`
- Create: `docs/reports/QWEN36_VLLM_TUNING_ZH.md`

- [ ] **Step 1: Run three interleaved formal blocks**

For every workload, alternate baseline and optimized order:

```text
R1 baseline -> optimized
R2 optimized -> baseline
R3 baseline -> optimized
```

Each block is engine-cold, uses the same physical devices, prompt IDs, output length and telemetry.
Failures remain in denominator and evidence; no rerun overwrites a failed attempt.

Use `block --profile tp2-bf16-graph16-mtp0` for baseline and
`block --profile-file "$RUN_ROOT/analysis/derived-candidate.json"` for optimized. Every invocation
includes `--run-root "$RUN_ROOT" --port 18000 --phase formal`, the workload under test, and the R1/R2/R3
repeat ID. Preserve the alternating order shown above rather than running all baselines first.

- [ ] **Step 2: Run the dual-TP2 scale-out lane**

Start two independently named TP2 containers on ports 18000 and 18001 using logical sets `0,2` and
`4,6`. Validate both labels and model identities, then distribute an even request count round-robin.
Report full-server aggregate throughput separately from single-replica Batch1/16.

Run the optimized settings through the dedicated command:

```bash
/home/USER/levius/.bootstrap/bin/uv run grserve-qwen36-tune dual \
  --run-root "$RUN_ROOT" --profile-file "$RUN_ROOT/analysis/derived-candidate.json" \
  --ports 18000 18001 --workload modelzoo-c8 --phase scale-out --repeat r1
```

Repeat as `r2` and `r3`; `dual` constructs replica B by changing only the validated logical-device
tuple and served port.

- [ ] **Step 3: Generate and validate final artifacts**

Run:

```bash
grserve-qwen36-tune analyze --run-root "$RUN_ROOT"
grserve-qwen36-tune report --run-root "$RUN_ROOT"
```

Expected: summary contains baseline/optimized raw values, gains, CV, failures, lane identities,
`semantic_accuracy_not_evaluated=true`, `token_consistency_not_required=true`, and every result can be
recomputed from request-level JSONL.

- [ ] **Step 4: Produce a profiling-bound kernel handoff**

`hotspot-handoff.json` must contain either:

```json
{"status":"NOT_REQUIRED","reason":"CONFIG_TARGET_MET","top_operation":null,"shapes":[]}
```

or a measured operation identity such as:

```json
{
  "status":"REQUIRES_KERNEL_PLAN",
  "reason":"CONFIG_GAIN_BELOW_TARGET",
  "top_operation":"iFormer.ops.fused_recurrent_gated_delta_rule_fwd",
  "phase":"decode",
  "dtype":"bfloat16",
  "shapes":[{"batch":16,"heads":48,"head_dim":128}],
  "share_of_decode_time_pct":31.4,
  "graph_required":true
}
```

Values in the second form must come from captured profiling evidence; do not copy the example numbers.
If `REQUIRES_KERNEL_PLAN`, stop and create a dedicated FTUCA operator spec/plan using the measured API,
shape corpus and graph constraints.

- [ ] **Step 5: Publish the local report copy**

Copy only the generated Markdown report and its evidence-root SHA/reference into
`docs/reports/QWEN36_VLLM_TUNING_ZH.md`. Do not copy prompts, generated text, model files or large raw
telemetry into Git.

- [ ] **Step 6: Verify server cleanup**

Expected final state:

```text
docker ps: no containers labelled grserve.qwen36.run
ports 18000-18031: unused
fantgpu-smi: no processes and memory returned to idle baseline
all owned container inspect/log evidence retained; no owned container remains after cleanup
```

- [ ] **Step 7: Commit the report**

```bash
git add docs/reports/QWEN36_VLLM_TUNING_ZH.md
git commit -m "docs: report Qwen3.6 GR308 tuning results"
```

## Completion criteria

This plan is complete only when:

- local tuning tests, Ruff and mypy pass;
- server baseline and formal A/B evidence are present and immutable;
- TP2 BF16 Batch1/16 and ModelZoo C8 metrics are reported with unambiguous units;
- baseline and optimized C1/C2/C4/C8/C16/C32/C64 saturation curves are reported or explicitly pruned;
- MTP, Graph/Scheduler and FHCCL each have a qualified, rejected or blocked conclusion;
- extreme FP16 and dual-replica results remain separate from comparable BF16 results;
- semantic accuracy and token consistency are explicitly `NOT_EVALUATED`;
- no owned service remains running;
- the report states whether the 15% throughput gate and stretch targets were reached;
- an FTUCA follow-up is created only when profiling produces a concrete operation, shapes and time share.
