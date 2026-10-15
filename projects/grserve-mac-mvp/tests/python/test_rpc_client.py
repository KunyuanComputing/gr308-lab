from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import struct
from pathlib import Path

import pytest

from grserve_api._generated import engine_pb2
from grserve_api.config import ApiConfig
import grserve_api.rpc_client as rpc_client_module
from grserve_api.rpc_client import (
    EngineClient,
    MAX_FRAME_BYTES,
    ProtocolError,
    RpcError,
    TransportError,
    read_server_envelope,
)
from support.fake_engine import (
    FakeConnection,
    FakeEngine,
    server_envelope,
)

EXPECTED_MAX_BUFFERED_COMPLETION_EVENTS = 8


def config(path: Path, **overrides: int) -> ApiConfig:
    return ApiConfig(engine_socket=path, **overrides)


def completion(request_id: str = "opaque:req/\u96ea") -> engine_pb2.CompletionRequest:
    return engine_pb2.CompletionRequest(
        request_id=request_id,
        model_id="mini-qwen38-hybrid",
        prompt_token_ids=[1, 4, 7],
        max_tokens=2,
        timeout_ms=1_000,
        decode_mode=engine_pb2.DECODE_MODE_GREEDY,
    )


def short_socket(tmp_path: Path) -> Path:
    return Path("/tmp") / f"grserve-py-{tmp_path.name}.sock"


def padded_server_frame(
    envelope: engine_pb2.ServerEnvelope, target_body_size: int
) -> bytes:
    body = envelope.SerializeToString()
    unknown_tag = b"\xfa\x7f"
    padding_size = target_body_size - len(body) - len(unknown_tag) - 3
    encoded_size = bytes(
        [
            (padding_size & 0x7F) | 0x80,
            ((padding_size >> 7) & 0x7F) | 0x80,
            padding_size >> 14,
        ]
    )
    body += unknown_tag + encoded_size + bytes(padding_size)
    assert len(body) == target_body_size
    return struct.pack(">I", len(body)) + body


def test_api_config_is_frozen_slotted_and_validates_socket_path() -> None:
    valid = ApiConfig(Path("/tmp/grserve.sock"))
    assert valid.request_timeout_ms == 30_000
    assert valid.cancel_timeout_ms == 250
    assert not hasattr(valid, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        valid.request_timeout_ms = 1  # type: ignore[misc]

    invalid_paths = [Path("relative.sock"), Path("/tmp/has\x00nul")]
    invalid_paths.append(Path("/tmp/") / ("\u96ea" * 34))
    for path in invalid_paths:
        with pytest.raises(ValueError):
            ApiConfig(path)

    exact = Path("/" + "a" * 99)
    assert len(str(exact).encode("utf-8")) == 100
    assert ApiConfig(exact).engine_socket == exact
    with pytest.raises(ValueError):
        ApiConfig(Path("/" + "a" * 100))
    with pytest.raises(ValueError, match="UTF-8"):
        ApiConfig(Path("/tmp/\udcff"))


@pytest.mark.parametrize("field", ["request_timeout_ms", "cancel_timeout_ms"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
def test_api_config_requires_strict_positive_integer_timeouts(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ApiConfig(Path("/tmp/grserve.sock"), **{field: value})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_health_and_list_models_use_independent_framed_connections(
    tmp_path: Path,
) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        request = await connection.receive()
        if request.HasField("health"):
            await connection.send(
                server_envelope(
                    "health",
                    engine_build_id="build-1",
                    accepting_requests=True,
                    replica_state=engine_pb2.REPLICA_STATE_HEALTHY,
                ),
                splits=range(1, 8),
            )
        else:
            response = server_envelope("list_models")
            response.list_models.models.add(
                id="mini-qwen38-hybrid",
                manifest_sha256="abc",
                max_context_tokens=64,
            )
            await connection.send(response)

    async with FakeEngine(socket_path, handler) as engine:
        client = EngineClient(config(socket_path))
        health, models = await asyncio.gather(client.health(), client.list_models())

    assert health.engine_build_id == "build-1"
    assert models.models[0].id == "mini-qwen38-hybrid"
    assert len(engine.connections) == 2
    for connection in engine.connections:
        request = connection.received[0]
        assert (request.protocol_version.major, request.protocol_version.minor) == (
            1,
            0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (struct.pack(">I", 0), "zero-length"),
        (struct.pack(">I", 4 * 1024 * 1024 + 1), "exceeds 4 MiB"),
        (b"\x00\x00", "EOF in 4-byte header"),
        (struct.pack(">I", 3) + b"\x08", "EOF in frame body"),
        (struct.pack(">I", 1) + b"\x80", "invalid ServerEnvelope protobuf"),
    ],
)
async def test_unary_rejects_malformed_and_partial_frames(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send_bytes(raw)

    async with FakeEngine(socket_path, handler):
        with pytest.raises(ProtocolError, match=message):
            await EngineClient(config(socket_path)).health()


@pytest.mark.asyncio
async def test_exactly_maximum_sized_frame_is_accepted() -> None:
    response = server_envelope("health", engine_build_id="max")
    body = response.SerializeToString()
    unknown_tag = b"\xfa\x7f"
    padding_size = MAX_FRAME_BYTES - len(body) - len(unknown_tag) - 4
    encoded_size = bytes(
        [
            (padding_size & 0x7F) | 0x80,
            ((padding_size >> 7) & 0x7F) | 0x80,
            ((padding_size >> 14) & 0x7F) | 0x80,
            padding_size >> 21,
        ]
    )
    body += unknown_tag + encoded_size + bytes(padding_size)
    assert len(body) == MAX_FRAME_BYTES
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack(">I", len(body)) + body)
    reader.feed_eof()

    decoded = await read_server_envelope(reader)

    assert decoded.health.engine_build_id == "max"


@pytest.mark.asyncio
async def test_connection_failure_is_stable_transport_error(tmp_path: Path) -> None:
    socket_path = short_socket(tmp_path)
    with pytest.raises(TransportError, match="could not connect to Engine"):
        await EngineClient(config(socket_path)).health()


@pytest.mark.asyncio
async def test_unary_validates_version_payload_and_remote_error(tmp_path: Path) -> None:
    responses = [
        server_envelope("health", major=2),
        server_envelope("list_models"),
        server_envelope(
            "error",
            code=engine_pb2.ERROR_CODE_RESOURCE_EXHAUSTED,
            message="busy",
            request_id="req-x",
        ),
    ]
    for index, response in enumerate(responses):
        socket_path = Path("/tmp") / f"grserve-unary-{tmp_path.name}-{index}.sock"

        async def handler(connection: FakeConnection, response=response) -> None:
            await connection.receive()
            await connection.send(response)

        async with FakeEngine(socket_path, handler):
            client = EngineClient(config(socket_path))
            if index == 0:
                with pytest.raises(ProtocolError, match="protocol version"):
                    await client.health()
            elif index == 1:
                with pytest.raises(ProtocolError, match="expected health"):
                    await client.health()
            else:
                with pytest.raises(RpcError) as caught:
                    await client.health()
                assert caught.value.code == engine_pb2.ERROR_CODE_RESOURCE_EXHAUSTED
                assert caught.value.message == "busy"
                assert caught.value.request_id == "req-x"


@pytest.mark.asyncio
async def test_completion_reads_split_frames_and_preserves_opaque_id(
    tmp_path: Path,
) -> None:
    socket_path = short_socket(tmp_path)
    request = completion()

    async def handler(connection: FakeConnection) -> None:
        received = await connection.receive()
        assert received.completion == request
        events = [
            server_envelope("accepted", request_id=request.request_id),
            server_envelope(
                "token",
                request_id=request.request_id,
                transaction_id="txn-1",
                sequence_number=1,
                token_id=9,
            ),
            server_envelope(
                "finished",
                request_id=request.request_id,
                finish_reason=engine_pb2.FINISH_REASON_LENGTH,
                last_sequence_number=1,
                usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            ),
        ]
        for event in events:
            encoded = event.SerializeToString()
            frame_size = len(encoded) + 4
            await connection.send(event, splits=range(1, frame_size))

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(request)
        assert stream.accepted.request_id == request.request_id
        events = [event async for event in stream]
        await stream.aclose()

    assert [event.WhichOneof("payload") for event in events] == ["token", "finished"]
    assert events[0].token.transaction_id == "txn-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "message"),
    [
        (
            server_envelope("token", request_id="req", sequence_number=1),
            "first completion event",
        ),
        (server_envelope("accepted", request_id="wrong"), "request id mismatch"),
        (server_envelope("accepted", request_id="req", minor=1), "protocol version"),
    ],
)
async def test_completion_rejects_invalid_first_event(
    tmp_path: Path, first: engine_pb2.ServerEnvelope, message: str
) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(first)

    async with FakeEngine(socket_path, handler):
        with pytest.raises(ProtocolError, match=message):
            await EngineClient(config(socket_path)).start_completion(completion("req"))


@pytest.mark.asyncio
async def test_preaccept_rpc_error_is_raised_with_all_stable_fields(
    tmp_path: Path,
) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(
            server_envelope(
                "error",
                code=engine_pb2.ERROR_CODE_INVALID_ARGUMENT,
                message="bad request",
                request_id="opaque",
                transaction_id="txn-9",
                retryable=True,
            )
        )

    async with FakeEngine(socket_path, handler):
        with pytest.raises(RpcError) as caught:
            await EngineClient(config(socket_path)).start_completion(
                completion("opaque")
            )
    error = caught.value
    assert (error.code, error.message, error.request_id) == (
        engine_pb2.ERROR_CODE_INVALID_ARGUMENT,
        "bad request",
        "opaque",
    )
    assert error.transaction_id == "txn-9"
    assert error.retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "message"),
    [
        (
            server_envelope("token", request_id="wrong", sequence_number=1),
            "request id mismatch",
        ),
        (
            server_envelope("token", request_id="req", sequence_number=2),
            "token sequence",
        ),
        (
            server_envelope(
                "finished",
                request_id="req",
                finish_reason=engine_pb2.FINISH_REASON_LENGTH,
                last_sequence_number=1,
                usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            ),
            "last sequence",
        ),
        (server_envelope("accepted", request_id="req"), "duplicate Accepted"),
        (server_envelope("health"), "unexpected completion payload"),
    ],
)
async def test_stream_rejects_id_sequence_terminal_and_payload_violations(
    tmp_path: Path, event: engine_pb2.ServerEnvelope, message: str
) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        await connection.send(event)

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        with pytest.raises(ProtocolError, match=message):
            await anext(aiter(stream))
        await stream.aclose()


@pytest.mark.asyncio
async def test_stream_surfaces_postaccept_rpc_error(tmp_path: Path) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        await connection.send(
            server_envelope(
                "error",
                code=engine_pb2.ERROR_CODE_RANK_FAILURE,
                message="rank failed",
                request_id="req",
            )
        )

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        with pytest.raises(RpcError, match="rank failed"):
            await anext(aiter(stream))
        await stream.aclose()


@pytest.mark.asyncio
async def test_cancel_shares_writer_lock_and_concurrent_calls_send_once(
    tmp_path: Path,
) -> None:
    socket_path = short_socket(tmp_path)
    release_ack = asyncio.Event()

    async def handler(connection: FakeConnection) -> None:
        request = await connection.receive()
        await connection.send(
            server_envelope("accepted", request_id=request.completion.request_id)
        )
        cancel = await connection.receive()
        assert cancel.cancel.request_id == request.completion.request_id
        assert cancel.cancel.reason == engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED
        await release_ack.wait()
        await connection.send(
            server_envelope(
                "cancel_ack",
                request_id=request.completion.request_id,
                disposition=engine_pb2.CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION,
            )
        )

    async with FakeEngine(socket_path, handler) as engine:
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        first = asyncio.create_task(
            stream.cancel(engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED)
        )
        second = asyncio.create_task(
            stream.cancel(engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED)
        )
        await asyncio.sleep(0)
        release_ack.set()
        first_ack, second_ack = await asyncio.gather(first, second)
        assert first_ack == second_ack
        assert first_ack is not None
        assert (
            first_ack.disposition
            == engine_pb2.CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION
        )
        await stream.aclose()
    assert len(engine.connections[0].received) == 2


@pytest.mark.asyncio
async def test_cancel_ack_may_arrive_before_cancel_drain_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = short_socket(tmp_path)
    ack_sent = asyncio.Event()
    release_drain = asyncio.Event()
    real_write = rpc_client_module.write_client_envelope

    async def write_then_pause(
        writer: asyncio.StreamWriter, envelope: engine_pb2.ClientEnvelope
    ) -> None:
        if not envelope.HasField("cancel"):
            await real_write(writer, envelope)
            return
        body = envelope.SerializeToString()
        writer.write(struct.pack(">I", len(body)) + body)
        await release_drain.wait()
        await writer.drain()

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        cancel = await connection.receive()
        await connection.send(
            server_envelope(
                "cancel_ack",
                request_id=cancel.cancel.request_id,
                disposition=engine_pb2.CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION,
            )
        )
        ack_sent.set()

    monkeypatch.setattr(rpc_client_module, "write_client_envelope", write_then_pause)
    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        cancel_task = asyncio.create_task(
            stream.cancel(engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED)
        )
        await ack_sent.wait()
        await asyncio.sleep(0)
        release_drain.set()
        ack = await cancel_task
        assert ack is not None
        assert ack.disposition == engine_pb2.CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION
        await stream.aclose()


@pytest.mark.asyncio
async def test_cancel_timeout_does_not_cancel_ack_future(tmp_path: Path) -> None:
    socket_path = short_socket(tmp_path)
    send_ack = asyncio.Event()

    async def handler(connection: FakeConnection) -> None:
        request = await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        await connection.receive()
        await send_ack.wait()
        await connection.send(
            server_envelope(
                "cancel_ack",
                request_id=request.completion.request_id,
                disposition=engine_pb2.CANCEL_DISPOSITION_COMMITTING_DETACHED,
            )
        )

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(
            config(socket_path, cancel_timeout_ms=1)
        ).start_completion(completion("req"))
        with pytest.raises(TimeoutError):
            await stream.cancel(engine_pb2.CANCEL_REASON_DEADLINE)
        send_ack.set()
        ack = await stream.cancel(engine_pb2.CANCEL_REASON_DEADLINE)
        assert ack is not None
        assert ack.disposition == engine_pb2.CANCEL_DISPOSITION_COMMITTING_DETACHED
        await stream.aclose()


@pytest.mark.asyncio
async def test_finished_wins_cancel_ack_race_and_aclose_is_concurrent_idempotent(
    tmp_path: Path,
) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        await connection.receive()
        await connection.send(
            server_envelope(
                "finished",
                request_id="req",
                finish_reason=engine_pb2.FINISH_REASON_LENGTH,
                last_sequence_number=0,
                usage={"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
            )
        )

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        cancel_task = asyncio.create_task(
            stream.cancel(engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED)
        )
        event = await anext(aiter(stream))
        assert event.HasField("finished")
        assert await cancel_task is None
        await asyncio.gather(stream.aclose(), stream.aclose(), stream.aclose())
        assert await stream.cancel(engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED) is None


@pytest.mark.asyncio
async def test_cancel_after_finished_does_not_write_another_frame(
    tmp_path: Path,
) -> None:
    socket_path = short_socket(tmp_path)
    unexpected_frame = asyncio.Event()

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        await connection.send(
            server_envelope(
                "finished",
                request_id="req",
                finish_reason=engine_pb2.FINISH_REASON_LENGTH,
                last_sequence_number=0,
                usage={"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
            )
        )
        try:
            await asyncio.wait_for(connection.receive(), timeout=0.02)
        except (TimeoutError, asyncio.IncompleteReadError):
            return
        unexpected_frame.set()

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        assert (await anext(aiter(stream))).HasField("finished")
        assert await stream.cancel(engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED) is None
        await asyncio.sleep(0.03)
        assert not unexpected_frame.is_set()
        await stream.aclose()


@pytest.mark.asyncio
async def test_oversized_completion_closes_connected_writer(tmp_path: Path) -> None:
    socket_path = short_socket(tmp_path)
    peer_saw_eof = asyncio.Event()

    async def handler(connection: FakeConnection) -> None:
        if await connection.reader.read() == b"":
            peer_saw_eof.set()

    request = completion("req")
    request.prompt_token_ids.extend([1] * (4 * 1024 * 1024))
    async with FakeEngine(socket_path, handler):
        with pytest.raises(ProtocolError, match="exceeds 4 MiB"):
            await EngineClient(config(socket_path)).start_completion(request)
        await asyncio.wait_for(peer_saw_eof.wait(), timeout=1)


@pytest.mark.asyncio
async def test_aclose_waits_for_inflight_cancel_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = short_socket(tmp_path)
    write_entered = asyncio.Event()
    release_write = asyncio.Event()
    real_write = rpc_client_module.write_client_envelope

    async def controlled_write(
        writer: asyncio.StreamWriter, envelope: engine_pb2.ClientEnvelope
    ) -> None:
        if envelope.HasField("cancel"):
            write_entered.set()
            await release_write.wait()
        await real_write(writer, envelope)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        cancel = await connection.receive()
        with contextlib.suppress(ConnectionError):
            await connection.send(
                server_envelope(
                    "cancel_ack",
                    request_id=cancel.cancel.request_id,
                    disposition=engine_pb2.CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION,
                )
            )

    monkeypatch.setattr(rpc_client_module, "write_client_envelope", controlled_write)
    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        cancel_task = asyncio.create_task(
            stream.cancel(engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED)
        )
        await write_entered.wait()
        close_task = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0)
        assert not close_task.done()
        release_write.set()
        await close_task
        assert await cancel_task is None


@pytest.mark.asyncio
async def test_cancelled_aclose_is_reentrant_and_shares_cleanup_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = short_socket(tmp_path)
    wait_closed_entered = asyncio.Event()
    release_wait_closed = asyncio.Event()

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        await asyncio.Event().wait()

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        writer = stream._writer
        real_close = writer.close
        real_wait_closed = writer.wait_closed
        close_calls = 0

        def controlled_close() -> None:
            nonlocal close_calls
            close_calls += 1

        async def controlled_wait_closed() -> None:
            wait_closed_entered.set()
            await release_wait_closed.wait()
            real_close()
            await real_wait_closed()

        monkeypatch.setattr(writer, "close", controlled_close)
        monkeypatch.setattr(writer, "wait_closed", controlled_wait_closed)

        first_close = asyncio.create_task(stream.aclose())
        await wait_closed_entered.wait()
        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close

        retry_close = asyncio.create_task(stream.aclose())
        concurrent_close = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0)
        assert not retry_close.done()
        assert not concurrent_close.done()

        release_wait_closed.set()
        await asyncio.gather(retry_close, concurrent_close)
        assert close_calls == 1
        assert stream._reader_task.done()
        assert stream._reader_task.cancelled()


@pytest.mark.asyncio
async def test_writer_close_failure_still_cancels_and_awaits_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        await asyncio.Event().wait()

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        writer = stream._writer
        real_close = writer.close
        close_calls = 0

        def controlled_close() -> None:
            nonlocal close_calls
            close_calls += 1

        async def failing_wait_closed() -> None:
            raise RuntimeError("wait_closed failed")

        monkeypatch.setattr(writer, "close", controlled_close)
        monkeypatch.setattr(writer, "wait_closed", failing_wait_closed)
        try:
            with pytest.raises(RuntimeError, match="wait_closed failed"):
                await stream.aclose()
            with pytest.raises(RuntimeError, match="wait_closed failed"):
                await stream.aclose()
            assert close_calls == 1
            assert stream._reader_task.done()
            assert stream._reader_task.cancelled()
        finally:
            real_close()
            stream._reader_task.cancel()
            await asyncio.gather(stream._reader_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_terminal_event_closes_writer_without_explicit_aclose(
    tmp_path: Path,
) -> None:
    socket_path = short_socket(tmp_path)
    peer_saw_eof = asyncio.Event()

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        await connection.send(
            server_envelope(
                "finished",
                request_id="req",
                finish_reason=engine_pb2.FINISH_REASON_LENGTH,
                last_sequence_number=0,
                usage={"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
            )
        )
        if await connection.reader.read() == b"":
            peer_saw_eof.set()

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        assert (await anext(aiter(stream))).HasField("finished")
        await asyncio.wait_for(peer_saw_eof.wait(), timeout=1)
        await stream.aclose()


@pytest.mark.asyncio
async def test_invalid_cancel_ack_fails_the_waiting_cancel(tmp_path: Path) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        await connection.receive()
        await connection.send(
            server_envelope(
                "cancel_ack",
                request_id="wrong",
                disposition=engine_pb2.CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION,
            )
        )

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        with pytest.raises(ProtocolError, match="request id mismatch"):
            await stream.cancel(engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED)
        await stream.aclose()


@pytest.mark.asyncio
async def test_rejects_unknown_rpc_error_code(tmp_path: Path) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        response = server_envelope("error", message="future", request_id="req")
        response.error.code = 99
        await connection.send(response)

    async with FakeEngine(socket_path, handler):
        with pytest.raises(ProtocolError, match="unknown RpcError code"):
            await EngineClient(config(socket_path)).start_completion(completion("req"))


@pytest.mark.asyncio
async def test_rejects_unknown_cancel_disposition(tmp_path: Path) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        await connection.receive()
        response = server_envelope("cancel_ack", request_id="req")
        response.cancel_ack.disposition = 99
        await connection.send(response)

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        with pytest.raises(ProtocolError, match="unknown CancelAck disposition"):
            await stream.cancel(engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED)
        await stream.aclose()


@pytest.mark.asyncio
async def test_finished_completion_count_matches_sequence(tmp_path: Path) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        await connection.send(
            server_envelope(
                "token",
                request_id="req",
                transaction_id="txn",
                sequence_number=1,
                token_id=9,
            )
        )
        await connection.send(
            server_envelope(
                "finished",
                request_id="req",
                finish_reason=engine_pb2.FINISH_REASON_LENGTH,
                last_sequence_number=1,
                usage={"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
            )
        )

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        assert (await anext(aiter(stream))).HasField("token")
        with pytest.raises(ProtocolError, match="completion token count"):
            await anext(aiter(stream))
        await stream.aclose()


@pytest.mark.asyncio
async def test_eof_before_terminal_raises_and_closes_without_orphans(
    tmp_path: Path,
) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))

    before = set(asyncio.all_tasks())
    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        with pytest.raises(ProtocolError, match="EOF before terminal"):
            await anext(aiter(stream))
        await stream.aclose()
    await asyncio.sleep(0)
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]


@pytest.mark.asyncio
async def test_event_queue_backpressures_sender_and_aclose_unblocks_pump(
    tmp_path: Path,
) -> None:
    socket_path = short_socket(tmp_path)
    sender_completed = asyncio.Event()

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        await connection.send(server_envelope("accepted", request_id="req"))
        try:
            for sequence in range(1, EXPECTED_MAX_BUFFERED_COMPLETION_EVENTS + 65):
                token = server_envelope(
                    "token",
                    request_id="req",
                    transaction_id=f"txn-{sequence}",
                    sequence_number=sequence,
                    token_id=sequence,
                )
                await connection.send_bytes(padded_server_frame(token, 512 * 1024))
        except ConnectionError:
            return
        sender_completed.set()

    async with FakeEngine(socket_path, handler):
        stream = await EngineClient(config(socket_path)).start_completion(
            completion("req")
        )
        try:
            assert (
                stream._events_queue.maxsize == EXPECTED_MAX_BUFFERED_COMPLETION_EVENTS
            )

            async def wait_until_queue_is_full() -> None:
                while not stream._events_queue.full():
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_until_queue_is_full(), timeout=1)
            assert (
                stream._events_queue.qsize() == EXPECTED_MAX_BUFFERED_COMPLETION_EVENTS
            )
            assert not sender_completed.is_set()
            assert not stream._reader_task.done()

            await stream.aclose()
            assert stream._reader_task.done()
            assert stream._reader_task.cancelled()
            assert stream._writer.is_closing()
            assert (
                stream._events_queue.qsize() <= EXPECTED_MAX_BUFFERED_COMPLETION_EVENTS
            )
        finally:
            await stream.aclose()


@pytest.mark.asyncio
async def test_fake_engine_propagates_handler_assertion(tmp_path: Path) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        raise AssertionError("handler contract failed")

    with pytest.raises(AssertionError, match="handler contract failed"):
        async with FakeEngine(socket_path, handler):
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            request = engine_pb2.ClientEnvelope()
            request.protocol_version.major = 1
            request.health.SetInParent()
            body = request.SerializeToString()
            writer.write(struct.pack(">I", len(body)) + body)
            await writer.drain()
            assert await reader.read() == b""
            writer.close()
            await writer.wait_closed()


@pytest.mark.asyncio
async def test_fake_engine_preserves_with_body_exception_priority(
    tmp_path: Path,
) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        raise AssertionError("secondary handler failure")

    with pytest.raises(ValueError, match="primary body failure"):
        async with FakeEngine(socket_path, handler):
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            request = engine_pb2.ClientEnvelope()
            request.protocol_version.major = 1
            request.health.SetInParent()
            body = request.SerializeToString()
            writer.write(struct.pack(">I", len(body)) + body)
            await writer.drain()
            assert await reader.read() == b""
            writer.close()
            await writer.wait_closed()
            raise ValueError("primary body failure")


@pytest.mark.asyncio
async def test_fake_engine_propagates_handler_initiated_cancellation(
    tmp_path: Path,
) -> None:
    socket_path = short_socket(tmp_path)

    async def handler(connection: FakeConnection) -> None:
        await connection.receive()
        raise asyncio.CancelledError("handler cancelled itself")

    with pytest.raises(asyncio.CancelledError):
        async with FakeEngine(socket_path, handler):
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            request = engine_pb2.ClientEnvelope()
            request.protocol_version.major = 1
            request.health.SetInParent()
            body = request.SerializeToString()
            writer.write(struct.pack(">I", len(body)) + body)
            await writer.drain()
            assert await reader.read() == b""
            writer.close()
            await writer.wait_closed()
