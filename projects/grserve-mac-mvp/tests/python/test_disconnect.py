from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

import pytest
import anyio
import uvicorn

from grserve_api._generated import engine_pb2
from grserve_api.app import create_app
from grserve_api.config import ApiConfig
from support.fake_engine import FakeConnection, FakeEngine, server_envelope


REQUEST_BODY = (
    b'{"model":"mini-qwen38-hybrid","prompt_token_ids":[1,4,7],'
    b'"max_tokens":1,"stream":true,"temperature":0,"top_p":1,"n":1}'
)


class DisconnectStream:
    def __init__(
        self,
        cancel_behavior: Literal["success", "error", "timeout", "swallow_cancel"],
    ) -> None:
        self.accepted = engine_pb2.CompletionAccepted(request_id="opaque")
        self.cancel_behavior = cancel_behavior
        self.entered = asyncio.Event()
        self.cancel_entered = asyncio.Event()
        self.cancel_exited = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.closed = asyncio.Event()
        self.cancel_calls = 0
        self.close_calls = 0

    def __aiter__(self) -> AsyncIterator[engine_pb2.ServerEnvelope]:
        return self.events()

    async def events(self) -> AsyncIterator[engine_pb2.ServerEnvelope]:
        self.entered.set()
        await asyncio.Future()
        if False:
            yield engine_pb2.ServerEnvelope()

    async def cancel(self, reason: int) -> None:
        try:
            assert reason == engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED
            self.cancel_calls += 1
            self.cancel_entered.set()
            if self.cancel_behavior == "error":
                raise RuntimeError("private cancel failure")
            if self.cancel_behavior == "timeout":
                await asyncio.Future()
            if self.cancel_behavior == "swallow_cancel":
                while not self.cancel_release.is_set():
                    try:
                        await self.cancel_release.wait()
                    except asyncio.CancelledError:
                        continue
        finally:
            self.cancel_exited.set()

    async def aclose(self) -> None:
        self.close_calls += 1
        self.closed.set()


class DisconnectEngine:
    def __init__(self, stream: DisconnectStream) -> None:
        self.stream = stream

    async def health(self) -> engine_pb2.HealthResponse:
        raise AssertionError

    async def list_models(self) -> engine_pb2.ListModelsResponse:
        raise AssertionError

    async def start_completion(
        self, request: engine_pb2.CompletionRequest
    ) -> DisconnectStream:
        return self.stream


async def drive_disconnect(stream: DisconnectStream, cancel_timeout_ms: int) -> None:
    app = create_app(
        ApiConfig(
            engine_socket=Path("/tmp/disconnect.sock"),
            cancel_timeout_ms=cancel_timeout_ms,
        ),
        DisconnectEngine(stream),
    )
    request_sent = False

    async def receive() -> dict[str, object]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": REQUEST_BODY,
                "more_body": False,
            }
        await stream.entered.wait()
        return {"type": "http.disconnect"}

    starts: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.start":
            starts.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "raw_path": b"/v1/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=1)
    assert len(starts) == 1
    assert starts[0]["status"] == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["success", "error", "timeout"])
async def test_http_disconnect_cancels_exactly_once_and_always_closes(
    behavior: Literal["success", "error", "timeout"],
) -> None:
    stream = DisconnectStream(behavior)

    await drive_disconnect(stream, cancel_timeout_ms=5)

    assert stream.cancel_calls == 1
    assert stream.close_calls == 1
    assert stream.closed.is_set()
    assert stream.cancel_exited.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("_repeat", range(100))
async def test_repeated_disconnect_delivery_cannot_duplicate_cancel(
    _repeat: int,
) -> None:
    stream = DisconnectStream("success")

    await drive_disconnect(stream, cancel_timeout_ms=250)
    await asyncio.wait_for(stream.closed.wait(), timeout=1)

    assert stream.cancel_calls == 1
    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_level_cancellation_cannot_skip_close_when_cancel_swallows_cancelled() -> None:
    stream = DisconnectStream("swallow_cancel")

    app_task = asyncio.create_task(drive_disconnect(stream, cancel_timeout_ms=20))
    await stream.cancel_entered.wait()
    try:
        await asyncio.wait_for(stream.closed.wait(), timeout=0.1)
    finally:
        stream.cancel_release.set()
        await app_task

    assert stream.cancel_calls == 1
    assert stream.close_calls == 1
    assert stream.closed.is_set()
    assert stream.cancel_exited.is_set()


class DelayedStartEngine(DisconnectEngine):
    def __init__(self) -> None:
        super().__init__(DisconnectStream("success"))
        self.start_entered = asyncio.Event()
        self.start_cancelled = asyncio.Event()

    async def start_completion(
        self, request: engine_pb2.CompletionRequest
    ) -> DisconnectStream:
        self.start_entered.set()
        try:
            await asyncio.Future()
        finally:
            self.start_cancelled.set()


@pytest.mark.asyncio
async def test_disconnect_before_accepted_cancels_and_awaits_start_task() -> None:
    engine = DelayedStartEngine()
    app = create_app(
        ApiConfig(engine_socket=Path("/tmp/pre-accept.sock")),
        engine,
    )
    request_sent = False

    async def receive() -> dict[str, object]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": REQUEST_BODY, "more_body": False}
        await engine.start_entered.wait()
        return {"type": "http.disconnect"}

    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "raw_path": b"/v1/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
    }

    with anyio.fail_after(0.2):
        await app(scope, receive, send)

    assert engine.start_cancelled.is_set()


async def wait_uvicorn_started(
    server: uvicorn.Server, task: asyncio.Task[None]
) -> None:
    while not server.started:
        if task.done():
            task.result()
            raise RuntimeError("uvicorn exited before startup")
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_real_uvicorn_disconnect_before_delayed_accepted_closes_uds(
) -> None:
    request_received = asyncio.Event()
    peer_closed = asyncio.Event()
    release = asyncio.Event()
    socket_path = Path(f"/private/tmp/grserve-du-{os.getpid()}.sock")

    async def handler(connection: FakeConnection) -> None:
        request = await connection.receive()
        request_received.set()
        read_task = asyncio.create_task(connection.reader.read())
        release_task = asyncio.create_task(release.wait())
        done, pending = await asyncio.wait(
            {read_task, release_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if read_task in done and read_task.result() == b"":
            peer_closed.set()
            return
        await connection.send(
            server_envelope(
                "error",
                code=engine_pb2.ERROR_CODE_INTERNAL,
                message="released",
                request_id=request.completion.request_id,
            )
        )

    async with FakeEngine(socket_path, handler):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        host, port = listener.getsockname()
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(
                    ApiConfig(engine_socket=socket_path, request_timeout_ms=5_000)
                ),
                log_level="error",
                lifespan="on",
            )
        )
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        writer: asyncio.StreamWriter | None = None
        try:
            await wait_uvicorn_started(server, server_task)
            _, writer = await asyncio.open_connection(host, port)
            writer.write(
                b"POST /v1/completions HTTP/1.1\r\n"
                + f"Host: {host}:{port}\r\n".encode()
                + b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(REQUEST_BODY)}\r\n\r\n".encode()
                + REQUEST_BODY
            )
            await writer.drain()
            await request_received.wait()
            writer.close()
            await writer.wait_closed()
            writer = None
            await asyncio.wait_for(peer_closed.wait(), timeout=0.2)
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            release.set()
            server.should_exit = True
            await asyncio.wait_for(server_task, timeout=6)
            listener.close()
