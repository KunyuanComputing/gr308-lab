from __future__ import annotations

import asyncio
import contextlib
import struct
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from types import TracebackType

from grserve_api._generated import engine_pb2

MAX_FRAME_BYTES = 4 * 1024 * 1024


def frame(message: engine_pb2.ServerEnvelope) -> bytes:
    body = message.SerializeToString()
    return struct.pack(">I", len(body)) + body


class FakeConnection:
    """A byte-accurate stand-in for one Engine UDS session."""

    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.received: list[engine_pb2.ClientEnvelope] = []

    async def receive(self) -> engine_pb2.ClientEnvelope:
        header = await self.reader.readexactly(4)
        length = struct.unpack(">I", header)[0]
        if length == 0 or length > MAX_FRAME_BYTES:
            raise ValueError("invalid client frame length")
        body = await self.reader.readexactly(length)
        request = engine_pb2.ClientEnvelope()
        consumed = request.ParseFromString(body)
        if consumed != len(body):
            raise ValueError("invalid client protobuf")
        self.received.append(request)
        return request

    async def send(
        self,
        message: engine_pb2.ServerEnvelope,
        *,
        splits: Iterable[int] = (),
    ) -> None:
        await self.send_bytes(frame(message), splits=splits)

    async def send_bytes(self, data: bytes, *, splits: Iterable[int] = ()) -> None:
        start = 0
        for end in splits:
            self.writer.write(data[start:end])
            await self.writer.drain()
            await asyncio.sleep(0)
            start = end
        self.writer.write(data[start:])
        await self.writer.drain()

    async def close(self) -> None:
        self.writer.close()
        await self.writer.wait_closed()


Handler = Callable[[FakeConnection], Awaitable[None]]


class FakeEngine:
    """A real asyncio UDS server; only the external Engine is replaced."""

    def __init__(self, socket_path: Path, handler: Handler) -> None:
        self.socket_path = socket_path
        self._handler = handler
        self._server: asyncio.AbstractServer | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self.connections: list[FakeConnection] = []

    async def __aenter__(self) -> FakeEngine:
        self._server = await asyncio.start_unix_server(
            self._accept, path=self.socket_path
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self._server is not None
        teardown_errors: list[BaseException] = []
        self._server.close()
        try:
            await self._server.wait_closed()
        except BaseException as error:
            teardown_errors.append(error)
        for connection in self.connections:
            connection.writer.close()
        teardown_cancelled: set[asyncio.Task[None]] = set()
        for task in self._tasks:
            if not task.done() and task.cancelling() == 0:
                teardown_cancelled.add(task)
                task.cancel()
        if self._tasks:
            results = await asyncio.gather(*self._tasks, return_exceptions=True)
            teardown_errors.extend(
                result
                for task, result in zip(self._tasks, results, strict=True)
                if isinstance(result, BaseException)
                and not (
                    isinstance(result, asyncio.CancelledError)
                    and task in teardown_cancelled
                )
            )
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()
        if exc_value is None and teardown_errors:
            raise teardown_errors[0]

    async def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        connection = FakeConnection(reader, writer)
        self.connections.append(connection)
        task = asyncio.create_task(self._run(connection))
        self._tasks.append(task)

    async def _run(self, connection: FakeConnection) -> None:
        try:
            await self._handler(connection)
        finally:
            connection.writer.close()
            with contextlib.suppress(ConnectionError):
                await connection.writer.wait_closed()


def server_envelope(
    payload: str,
    *,
    major: int = 1,
    minor: int = 0,
    **fields: object,
) -> engine_pb2.ServerEnvelope:
    envelope = engine_pb2.ServerEnvelope()
    envelope.protocol_version.major = major
    envelope.protocol_version.minor = minor
    target = getattr(envelope, payload)
    for name, value in fields.items():
        if name == "usage":
            for usage_name, usage_value in value.items():
                setattr(target.usage, usage_name, usage_value)
        else:
            setattr(target, name, value)
    target.SetInParent()
    return envelope
