# mypy: disable-error-code="attr-defined,import-untyped,name-defined"
from __future__ import annotations

import asyncio
import contextlib
import struct
from collections.abc import AsyncIterator
from typing import NoReturn, TypeVar

from google.protobuf.message import DecodeError

from grserve_api._generated import engine_pb2
from grserve_api.config import ApiConfig

MAX_FRAME_BYTES = 4 * 1024 * 1024
PROTOCOL_MAJOR = 1
PROTOCOL_MINOR = 0
# Eight maximum-sized frames cap queued protobuf storage at 32 MiB.
MAX_BUFFERED_COMPLETION_EVENTS = 8


class RpcClientError(Exception):
    """Base class for stable Engine client failures."""


class TransportError(RpcClientError):
    """The local Engine transport could not be used."""


class ProtocolError(RpcClientError):
    """The Engine sent bytes or events outside the frozen protocol."""


class RpcError(RpcClientError):
    """A structured error returned by the Engine."""

    def __init__(self, error: engine_pb2.RpcError) -> None:
        self.code = error.code
        self.message = error.message
        self.request_id = error.request_id
        self.transaction_id = error.transaction_id
        self.retryable = error.retryable
        code_name = engine_pb2.ErrorCode.Name(error.code)
        super().__init__(f"{code_name}: {error.message}")


def _client_envelope(payload: str) -> engine_pb2.ClientEnvelope:
    envelope = engine_pb2.ClientEnvelope()
    envelope.protocol_version.major = PROTOCOL_MAJOR
    envelope.protocol_version.minor = PROTOCOL_MINOR
    getattr(envelope, payload).SetInParent()
    return envelope


def client_envelope_for_health() -> engine_pb2.ClientEnvelope:
    return _client_envelope("health")


def client_envelope_for_models() -> engine_pb2.ClientEnvelope:
    return _client_envelope("list_models")


def client_envelope_for_completion(
    request: engine_pb2.CompletionRequest,
) -> engine_pb2.ClientEnvelope:
    envelope = _client_envelope("completion")
    envelope.completion.CopyFrom(request)
    return envelope


def client_envelope_for_cancel(
    request_id: str, reason: int
) -> engine_pb2.ClientEnvelope:
    envelope = _client_envelope("cancel")
    envelope.cancel.request_id = request_id
    envelope.cancel.reason = reason
    return envelope


async def write_client_envelope(
    writer: asyncio.StreamWriter, envelope: engine_pb2.ClientEnvelope
) -> None:
    body = envelope.SerializeToString()
    if not body:
        raise ProtocolError("cannot write zero-length ClientEnvelope")
    if len(body) > MAX_FRAME_BYTES:
        raise ProtocolError("client frame exceeds 4 MiB limit")
    writer.write(struct.pack(">I", len(body)))
    writer.write(body)
    try:
        await writer.drain()
    except (BrokenPipeError, ConnectionError, OSError) as error:
        raise TransportError("could not write to Engine") from error


async def read_server_envelope(
    reader: asyncio.StreamReader,
) -> engine_pb2.ServerEnvelope:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError as error:
        if not error.partial:
            raise ProtocolError("unexpected EOF before frame header") from error
        raise ProtocolError("EOF in 4-byte header") from error
    length = struct.unpack(">I", header)[0]
    if length == 0:
        raise ProtocolError("zero-length frame")
    if length > MAX_FRAME_BYTES:
        raise ProtocolError("frame exceeds 4 MiB limit")
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError as error:
        raise ProtocolError("EOF in frame body") from error
    envelope = engine_pb2.ServerEnvelope()
    try:
        consumed = envelope.ParseFromString(body)
    except DecodeError as error:
        raise ProtocolError("invalid ServerEnvelope protobuf") from error
    if consumed != len(body):
        raise ProtocolError("invalid ServerEnvelope protobuf")
    return envelope


def _validate_envelope(envelope: engine_pb2.ServerEnvelope) -> str:
    version = envelope.protocol_version
    if (version.major, version.minor) != (PROTOCOL_MAJOR, PROTOCOL_MINOR):
        raise ProtocolError("incompatible RPC protocol version")
    payload = envelope.WhichOneof("payload")
    if payload is None:
        raise ProtocolError("ServerEnvelope payload is missing")
    return payload


def _raise_rpc_error(error: engine_pb2.RpcError) -> NoReturn:
    if error.code not in {
        engine_pb2.ERROR_CODE_INVALID_ARGUMENT,
        engine_pb2.ERROR_CODE_INCOMPATIBLE_VERSION,
        engine_pb2.ERROR_CODE_UNSUPPORTED_CAPABILITY,
        engine_pb2.ERROR_CODE_RESOURCE_EXHAUSTED,
        engine_pb2.ERROR_CODE_DEADLINE_EXCEEDED,
        engine_pb2.ERROR_CODE_RANK_FAILURE,
        engine_pb2.ERROR_CODE_REPLICA_FATAL,
        engine_pb2.ERROR_CODE_INTERNAL,
    }:
        raise ProtocolError("unknown RpcError code")
    if not error.message:
        raise ProtocolError("RpcError message is empty")
    raise RpcError(error)


class EngineClient:
    def __init__(self, config: ApiConfig) -> None:
        self._config = config

    async def health(self) -> engine_pb2.HealthResponse:
        response = await self._unary(client_envelope_for_health())
        payload = _validate_envelope(response)
        if payload == "error":
            _raise_rpc_error(response.error)
        if payload != "health":
            raise ProtocolError(f"expected health payload, received {payload}")
        return response.health

    async def list_models(self) -> engine_pb2.ListModelsResponse:
        response = await self._unary(client_envelope_for_models())
        payload = _validate_envelope(response)
        if payload == "error":
            _raise_rpc_error(response.error)
        if payload != "list_models":
            raise ProtocolError(f"expected list_models payload, received {payload}")
        return response.list_models

    async def _unary(
        self, request: engine_pb2.ClientEnvelope
    ) -> engine_pb2.ServerEnvelope:
        reader: asyncio.StreamReader
        writer: asyncio.StreamWriter
        try:
            async with asyncio.timeout(self._config.request_timeout_ms / 1000):
                reader, writer = await self._connect()
                try:
                    await write_client_envelope(writer, request)
                    return await read_server_envelope(reader)
                finally:
                    writer.close()
                    with contextlib.suppress(ConnectionError, OSError):
                        await writer.wait_closed()
        except TimeoutError as error:
            raise TransportError("Engine RPC request timed out") from error

    async def start_completion(
        self, request: engine_pb2.CompletionRequest
    ) -> CompletionStream:
        writer: asyncio.StreamWriter | None = None
        try:
            async with asyncio.timeout(self._config.request_timeout_ms / 1000):
                reader, writer = await self._connect()
                try:
                    await write_client_envelope(
                        writer, client_envelope_for_completion(request)
                    )
                    return await CompletionStream.create(
                        reader, writer, request.request_id, self._config
                    )
                except BaseException:
                    writer.close()
                    with contextlib.suppress(ConnectionError, OSError):
                        await writer.wait_closed()
                    raise
        except TimeoutError as error:
            if writer is not None:
                writer.close()
                with contextlib.suppress(ConnectionError, OSError):
                    await writer.wait_closed()
            raise TransportError("Engine RPC request timed out") from error

    async def _connect(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        try:
            return await asyncio.open_unix_connection(str(self._config.engine_socket))
        except (ConnectionError, OSError) as error:
            raise TransportError("could not connect to Engine") from error


_QUEUE_END = object()


class CompletionStream:
    accepted: engine_pb2.CompletionAccepted

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        request_id: str,
        config: ApiConfig,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._request_id = request_id
        self._config = config
        self._events_queue: asyncio.Queue[
            engine_pb2.ServerEnvelope | BaseException | object
        ] = asyncio.Queue(maxsize=MAX_BUFFERED_COMPLETION_EVENTS)
        self._reader_done = asyncio.Event()
        loop = asyncio.get_running_loop()
        self._accepted_future: asyncio.Future[engine_pb2.CompletionAccepted] = (
            loop.create_future()
        )
        self._cancel_ack_future: asyncio.Future[engine_pb2.CancelAck | None] = (
            loop.create_future()
        )
        self._cancel_ack_future.add_done_callback(_consume_future_exception)
        self._writer_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._terminal = False
        self._cancel_sent = False
        self._close_task: asyncio.Task[None] | None = None
        self._writer_close_task: asyncio.Task[None] | None = None
        self._reader_task = asyncio.create_task(self._reader_pump())

    @classmethod
    async def create(
        cls,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        request_id: str,
        config: ApiConfig,
    ) -> CompletionStream:
        stream = cls(reader, writer, request_id, config)
        try:
            stream.accepted = await stream._accepted_future
            return stream
        except BaseException:
            await stream.aclose()
            raise

    def __aiter__(self) -> AsyncIterator[engine_pb2.ServerEnvelope]:
        return self._events()

    async def _events(self) -> AsyncIterator[engine_pb2.ServerEnvelope]:
        while True:
            item = await self._next_event()
            if item is _QUEUE_END:
                return
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, engine_pb2.ServerEnvelope)
            yield item

    async def _next_event(
        self,
    ) -> engine_pb2.ServerEnvelope | BaseException | object:
        if self._reader_done.is_set():
            try:
                return self._events_queue.get_nowait()
            except asyncio.QueueEmpty:
                return _QUEUE_END
        event_task = asyncio.create_task(self._events_queue.get())
        done_task = asyncio.create_task(self._reader_done.wait())
        try:
            completed, _ = await asyncio.wait(
                {event_task, done_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if event_task in completed:
                return event_task.result()
            try:
                return self._events_queue.get_nowait()
            except asyncio.QueueEmpty:
                return _QUEUE_END
        finally:
            for task in (event_task, done_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(event_task, done_task, return_exceptions=True)

    async def cancel(self, reason: int) -> engine_pb2.CancelAck | None:
        if reason not in {
            engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED,
            engine_pb2.CANCEL_REASON_DEADLINE,
            engine_pb2.CANCEL_REASON_SERVER_SHUTDOWN,
        }:
            raise ValueError("cancel reason must be specified")
        async with self._writer_lock:
            if self._closed or self._terminal:
                return None
            if not self._cancel_sent:
                self._cancel_sent = True
                await write_client_envelope(
                    self._writer,
                    client_envelope_for_cancel(self.accepted.request_id, reason),
                )
        return await asyncio.wait_for(
            asyncio.shield(self._cancel_ack_future),
            timeout=self._config.cancel_timeout_ms / 1000,
        )

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._close_task is None:
                self._closed = True
                if not self._cancel_ack_future.done():
                    self._cancel_ack_future.set_result(None)
                self._close_task = asyncio.create_task(self._close_resources())
                self._close_task.add_done_callback(_consume_future_exception)
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def _close_resources(self) -> None:
        try:
            await self._writer_close()
        finally:
            if not self._reader_task.done():
                self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)

    async def _writer_close(self) -> None:
        if self._writer_close_task is None:
            self._writer_close_task = asyncio.create_task(self._close_writer_once())
            self._writer_close_task.add_done_callback(_consume_future_exception)
        await asyncio.shield(self._writer_close_task)

    async def _close_writer_once(self) -> None:
        async with self._writer_lock:
            self._writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await self._writer.wait_closed()

    async def _reader_pump(self) -> None:
        accepted = False
        next_sequence = 1
        terminal = False
        try:
            while not terminal:
                try:
                    envelope = await read_server_envelope(self._reader)
                except ProtocolError as error:
                    if str(error) == "unexpected EOF before frame header":
                        raise ProtocolError(
                            "EOF before terminal completion event"
                        ) from error
                    raise
                payload = _validate_envelope(envelope)
                if not accepted:
                    if payload == "error":
                        self._validate_request_id(envelope.error.request_id)
                        _raise_rpc_error(envelope.error)
                    if payload != "accepted":
                        raise ProtocolError("first completion event must be Accepted")
                    self._validate_request_id(envelope.accepted.request_id)
                    accepted = True
                    if not self._accepted_future.done():
                        self._accepted_future.set_result(envelope.accepted)
                    continue
                if payload == "accepted":
                    raise ProtocolError("duplicate Accepted completion event")
                if payload == "token":
                    self._validate_request_id(envelope.token.request_id)
                    if envelope.token.sequence_number != next_sequence:
                        raise ProtocolError(
                            "token sequence must increase strictly by one"
                        )
                    if not envelope.token.transaction_id:
                        raise ProtocolError("token transaction id is empty")
                    next_sequence += 1
                    await self._events_queue.put(envelope)
                    continue
                if payload == "finished":
                    self._validate_request_id(envelope.finished.request_id)
                    if envelope.finished.last_sequence_number != next_sequence - 1:
                        raise ProtocolError(
                            "finished last sequence does not match tokens"
                        )
                    if envelope.finished.finish_reason not in {
                        engine_pb2.FINISH_REASON_LENGTH,
                        engine_pb2.FINISH_REASON_CANCELLED,
                    }:
                        raise ProtocolError("unknown finished reason")
                    if not envelope.finished.HasField("usage"):
                        raise ProtocolError("finished usage is missing")
                    usage = envelope.finished.usage
                    if (
                        usage.completion_tokens
                        != envelope.finished.last_sequence_number
                    ):
                        raise ProtocolError(
                            "finished completion token count does not match sequence"
                        )
                    if (
                        usage.total_tokens
                        != usage.prompt_tokens + usage.completion_tokens
                    ):
                        raise ProtocolError("finished usage totals are inconsistent")
                    terminal = True
                    self._terminal = True
                    await self._events_queue.put(envelope)
                    self._resolve_cancel(None)
                    continue
                if payload == "error":
                    self._validate_request_id(envelope.error.request_id)
                    _raise_rpc_error(envelope.error)
                if payload == "cancel_ack":
                    self._validate_request_id(envelope.cancel_ack.request_id)
                    if not self._cancel_sent:
                        raise ProtocolError("CancelAck arrived before cancel request")
                    if envelope.cancel_ack.disposition not in {
                        engine_pb2.CANCEL_DISPOSITION_ABORTED_BEFORE_DECISION,
                        engine_pb2.CANCEL_DISPOSITION_COMMITTING_DETACHED,
                        engine_pb2.CANCEL_DISPOSITION_ALREADY_TERMINAL,
                        engine_pb2.CANCEL_DISPOSITION_NOT_FOUND,
                    }:
                        raise ProtocolError("unknown CancelAck disposition")
                    self._resolve_cancel(envelope.cancel_ack)
                    terminal = True
                    self._terminal = True
                    continue
                raise ProtocolError(f"unexpected completion payload: {payload}")
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._terminal = True
            if not self._accepted_future.done():
                self._accepted_future.set_exception(error)
            else:
                await self._events_queue.put(error)
            if self._cancel_sent and not self._cancel_ack_future.done():
                self._cancel_ack_future.set_exception(error)
            else:
                self._resolve_cancel(None)
        finally:
            if not self._accepted_future.done():
                self._accepted_future.set_exception(
                    ProtocolError("EOF before first completion event")
                )
            self._resolve_cancel(None)
            self._reader_done.set()
            with contextlib.suppress(Exception):
                await self._writer_close()

    def _validate_request_id(self, received: str) -> None:
        if received != self._request_id:
            raise ProtocolError("completion request id mismatch")

    def _resolve_cancel(self, value: engine_pb2.CancelAck | None) -> None:
        if not self._cancel_ack_future.done():
            self._cancel_ack_future.set_result(value)


_FutureResult = TypeVar("_FutureResult")


def _consume_future_exception(future: asyncio.Future[_FutureResult]) -> None:
    if not future.cancelled():
        future.exception()
