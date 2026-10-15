from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
import anyio

from grserve_api.app import create_app
from grserve_api.config import ApiConfig
import support.engine_process as engine_process_module
from support.engine_process import EngineProcess


ROOT = Path(__file__).resolve().parents[2]
REQUEST = {
    "model": "mini-qwen38-hybrid",
    "prompt_token_ids": [1, 4, 7],
    "max_tokens": 8,
    "stream": True,
    "temperature": 0,
    "top_p": 1,
    "n": 1,
}


def parse_sse(content: bytes) -> list[object]:
    assert content.endswith(b"data: [DONE]\n\n")
    assert content.count(b"data: [DONE]\n\n") == 1
    events: list[object] = []
    for frame in content.split(b"\n\n"):
        if not frame:
            continue
        assert frame.startswith(b"data: ")
        data = frame.removeprefix(b"data: ")
        events.append("[DONE]" if data == b"[DONE]" else json.loads(data))
    return events


async def wait_started(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    while not server.started:
        if task.done():
            task.result()
            raise RuntimeError("uvicorn exited before startup")
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_engine_start_failure_reaps_process_and_owned_socket() -> None:
    engine = EngineProcess(hal_plugin=ROOT / "missing-hal-plugin.dylib")

    with pytest.raises(RuntimeError, match="exited before health readiness"):
        await engine.__aenter__()

    assert engine.process.returncode is not None
    assert not engine.directory_path.exists()


@pytest.mark.asyncio
async def test_engine_exec_failure_removes_private_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRSERVE_ENGINE_BIN", str(ROOT / "missing-engine"))
    engine = EngineProcess(hal_plugin=ROOT / "missing-hal-plugin.dylib")

    with pytest.raises(FileNotFoundError):
        await engine.__aenter__()

    assert not engine.directory_path.exists()


@pytest.mark.asyncio
async def test_engine_large_stderr_cannot_block_exit_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "noisy-engine"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "os.write(2, b'0123456789abcdef0123456789abcdef\\n' * 20000)\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    monkeypatch.setenv("GRSERVE_ENGINE_BIN", str(executable))
    engine = EngineProcess(
        hal_plugin=ROOT / "missing-hal-plugin.dylib", ready_timeout=2.0
    )

    with pytest.raises(RuntimeError, match="exited before health readiness"):
        await engine.__aenter__()

    assert engine.process.returncode == 7
    assert not engine.directory_path.exists()


@pytest.mark.asyncio
async def test_engine_readiness_requires_accepting_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRSERVE_ENGINE_BIN", "/unused")
    responses = iter([False, True])

    class HealthClient:
        calls = 0

        async def health(self) -> SimpleNamespace:
            self.calls += 1
            return SimpleNamespace(accepting_requests=next(responses))

    health_client = HealthClient()
    monkeypatch.setattr(
        engine_process_module, "EngineClient", lambda _: health_client
    )
    engine = EngineProcess(
        hal_plugin=ROOT / "unused", ready_timeout=1
    )
    engine._socket_path = Path("/private/tmp/absent-health.sock")
    engine._process = SimpleNamespace(returncode=None)

    await engine._wait_ready()

    assert health_client.calls == 2


@pytest.mark.asyncio
async def test_engine_exit_during_final_health_probe_is_not_misreported_as_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRSERVE_ENGINE_BIN", "/unused")
    clock = iter([0.0, 0.5, 2.0])
    monkeypatch.setattr(
        engine_process_module,
        "time",
        SimpleNamespace(monotonic=lambda: next(clock)),
    )
    engine = EngineProcess(hal_plugin=ROOT / "unused", ready_timeout=1)
    engine._socket_path = Path("/private/tmp/absent-health.sock")
    engine._process = SimpleNamespace(returncode=None)

    class HealthClient:
        async def health(self) -> None:
            engine.process.returncode = 7
            raise engine_process_module.RpcClientError("engine exited")

    monkeypatch.setattr(engine_process_module, "EngineClient", lambda _: HealthClient())

    with pytest.raises(RuntimeError, match="exited before health readiness"):
        await engine._wait_ready()


@pytest.mark.asyncio
async def test_engine_exit_observed_during_deadline_grace_is_not_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRSERVE_ENGINE_BIN", "/unused")
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(
        engine_process_module,
        "time",
        SimpleNamespace(monotonic=lambda: next(clock)),
    )
    engine = EngineProcess(hal_plugin=ROOT / "unused", ready_timeout=1)
    engine._socket_path = Path("/private/tmp/absent-health.sock")

    class ExitingProcess:
        returncode: int | None = None

        async def wait(self) -> int:
            self.returncode = 7
            return 7

    engine._process = ExitingProcess()

    with pytest.raises(RuntimeError, match="exited before health readiness"):
        await engine._wait_ready()


@pytest.mark.asyncio
async def test_cancelled_teardown_still_reaps_engine_and_is_idempotent() -> None:
    engine = EngineProcess(
        hal_plugin=ROOT / "build/mac-arm64-debug/libgrserve_mock_hal.dylib"
    )
    await engine.__aenter__()
    try:
        with anyio.CancelScope() as scope:
            scope.cancel()
            await engine.__aexit__(None, None, None)

        assert engine.process.returncode is not None
        assert not engine.directory_path.exists()
        await engine.__aexit__(None, None, None)
    finally:
        if engine.process.returncode is None:
            await engine.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_direct_task_cancellation_waits_for_engine_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = EngineProcess(
        hal_plugin=ROOT / "build/mac-arm64-debug/libgrserve_mock_hal.dylib"
    )
    await engine.__aenter__()
    teardown_entered = asyncio.Event()
    teardown_release = asyncio.Event()
    original_teardown_once = engine._teardown_once

    async def delayed_teardown() -> None:
        teardown_entered.set()
        await teardown_release.wait()
        await original_teardown_once()

    monkeypatch.setattr(engine, "_teardown_once", delayed_teardown)
    exit_task = asyncio.create_task(engine.__aexit__(None, None, None))
    await teardown_entered.wait()
    try:
        exit_task.cancel()
        await asyncio.sleep(0)
        assert not exit_task.done()

        exit_task.cancel()
        await asyncio.sleep(0)
        assert not exit_task.done()

        teardown_release.set()
        with pytest.raises(asyncio.CancelledError):
            await exit_task

        assert engine.process.returncode is not None
        assert not engine.socket_path.exists()
        assert not engine.directory_path.exists()
    finally:
        teardown_release.set()
        await asyncio.gather(exit_task, return_exceptions=True)
        if engine.process.returncode is None:
            await engine.__aexit__(None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hal_plugin",
    [
        ROOT / "build/mac-arm64-debug/libgrserve_cpu_hal.dylib",
        ROOT / "build/mac-arm64-debug/libgrserve_mock_hal.dylib",
    ],
    ids=["cpu", "scripted-mock"],
)
async def test_real_engine_and_uvicorn_have_identical_frozen_completion(
    hal_plugin: Path,
) -> None:
    async with EngineProcess(hal_plugin=hal_plugin) as engine:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        host, port = listener.getsockname()
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(ApiConfig(engine_socket=engine.socket_path)),
                log_level="error",
                lifespan="on",
            )
        )
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            await asyncio.wait_for(wait_started(server, server_task), timeout=5)
            async with httpx.AsyncClient(
                base_url=f"http://{host}:{port}", trust_env=False
            ) as client:
                response = await client.post("/v1/completions", json=REQUEST)
        finally:
            server.should_exit = True
            await asyncio.wait_for(server_task, timeout=5)
            listener.close()

    assert response.status_code == 200
    events = parse_sse(response.content)
    assert [event["choices"][0]["token_ids"][0] for event in events[:8]] == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert [event["sequence"] for event in events[:8]] == list(range(1, 9))
    request_ids = {event["id"] for event in events[:-1]}
    assert len(request_ids) == 1
    assert next(iter(request_ids))
    assert events[8]["choices"][0]["finish_reason"] == "length"
    assert events[8]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 8,
        "total_tokens": 11,
    }
    assert events[9] == "[DONE]"
