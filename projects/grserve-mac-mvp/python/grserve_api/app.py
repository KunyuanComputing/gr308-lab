# mypy: disable-error-code="attr-defined,import-untyped,name-defined"
from __future__ import annotations

import asyncio
import inspect
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol, cast

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from grserve_api._generated import engine_pb2
from grserve_api.config import ApiConfig
from grserve_api.errors import PublicError, map_rpc_error
from grserve_api.rpc_client import (
    CompletionStream,
    EngineClient,
    ProtocolError,
    RpcError,
    TransportError,
)
from grserve_api.schemas import CompletionRequest
from grserve_api.sse import SseEncoder


class EngineLike(Protocol):
    async def health(self) -> engine_pb2.HealthResponse: ...

    async def list_models(self) -> engine_pb2.ListModelsResponse: ...

    async def start_completion(
        self, request: engine_pb2.CompletionRequest
    ) -> CompletionStream: ...


def _internal_error(request_id: str = "") -> PublicError:
    return PublicError(
        status_code=500,
        code="INTERNAL",
        message="An internal error occurred.",
        request_id=request_id,
        retryable=False,
    )


def _transport_error(request_id: str = "") -> PublicError:
    return PublicError(
        status_code=503,
        code="RANK_FAILURE",
        message="The Engine is temporarily unavailable.",
        request_id=request_id,
        retryable=True,
    )


def _error_response(error: PublicError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content=error.body)


def _consume_task(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        task.exception()


def _track_task(
    task: asyncio.Task[Any], tracked: set[asyncio.Task[Any]]
) -> None:
    tracked.add(task)

    def finished(done: asyncio.Task[Any]) -> None:
        tracked.discard(done)
        _consume_task(done)

    task.add_done_callback(finished)


async def _wait_bounded(task: asyncio.Task[Any], timeout: float) -> bool:
    if task.done():
        return True
    done, _ = await asyncio.wait({task}, timeout=max(0.0, timeout))
    return bool(done)


async def _cancel_and_close(
    stream: CompletionStream,
    *,
    cancel: bool,
    cancel_timeout_ms: int,
    tracked: set[asyncio.Task[Any]],
) -> None:
    loop = asyncio.get_running_loop()
    timeout = cancel_timeout_ms / 1000
    deadline = loop.time() + timeout * 0.8
    cancel_task: asyncio.Task[Any] | None = None
    if cancel:
        cancel_task = asyncio.create_task(
            stream.cancel(engine_pb2.CANCEL_REASON_CLIENT_DISCONNECTED)
        )
        _track_task(cancel_task, tracked)
        await _wait_bounded(cancel_task, timeout * 0.4)
        if not cancel_task.done():
            cancel_task.cancel()

    close_task = asyncio.create_task(stream.aclose())
    _track_task(close_task, tracked)
    await _wait_bounded(close_task, deadline - loop.time())
    if not close_task.done():
        close_task.cancel()
    if cancel_task is not None and not cancel_task.done():
        cancel_task.cancel()
    await asyncio.sleep(0)


async def _run_cleanup_shielded(
    stream: CompletionStream,
    *,
    cancel: bool,
    config: ApiConfig,
    tracked: set[asyncio.Task[Any]],
) -> None:
    with anyio.CancelScope(shield=True):
        cleanup = asyncio.create_task(
            _cancel_and_close(
                stream,
                cancel=cancel,
                cancel_timeout_ms=config.cancel_timeout_ms,
                tracked=tracked,
            )
        )
        _track_task(cleanup, tracked)
        completed = await _wait_bounded(
            cleanup, config.cancel_timeout_ms / 1000
        )
        if not completed:
            cleanup.cancel()
            await asyncio.sleep(0)


async def _watch_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


def _replica_state_name(state: int) -> str:
    names = {
        engine_pb2.REPLICA_STATE_STARTING: "starting",
        engine_pb2.REPLICA_STATE_HEALTHY: "healthy",
        engine_pb2.REPLICA_STATE_FATAL: "fatal",
        engine_pb2.REPLICA_STATE_STOPPING: "stopping",
    }
    return names.get(state, "unknown")


def create_app(config: ApiConfig, engine: EngineClient | None = None) -> FastAPI:
    client: EngineLike = cast(EngineLike, engine or EngineClient(config))
    tracked_tasks: set[asyncio.Task[Any]] = set()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            close = getattr(client, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await cast(Awaitable[object], result)
            with anyio.CancelScope(shield=True):
                pending = list(tracked_tasks)
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.wait(
                        pending, timeout=config.cancel_timeout_ms / 1000
                    )

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _: Request, __: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            PublicError(
                status_code=400,
                code="INVALID_ARGUMENT",
                message="The request is invalid.",
                request_id="",
                retryable=False,
            )
        )

    async def call_unary(call: Callable[[], Awaitable[Any]]) -> Any:
        try:
            return await call()
        except RpcError as error:
            return _error_response(map_rpc_error(error))
        except TransportError:
            return _error_response(_transport_error())
        except ProtocolError:
            return _error_response(_internal_error())

    @app.get("/health", response_model=None)
    async def health() -> dict[str, object] | JSONResponse:
        response = await call_unary(client.health)
        if isinstance(response, JSONResponse):
            return response
        assert isinstance(response, engine_pb2.HealthResponse)
        return {
            "status": "ok" if response.accepting_requests else "unavailable",
            "engine_build_id": response.engine_build_id,
            "accepting_requests": response.accepting_requests,
            "replica_state": _replica_state_name(response.replica_state),
        }

    @app.get("/v1/models", response_model=None)
    async def models() -> dict[str, object] | JSONResponse:
        response = await call_unary(client.list_models)
        if isinstance(response, JSONResponse):
            return response
        assert isinstance(response, engine_pb2.ListModelsResponse)
        return {
            "object": "list",
            "data": [
                {
                    "id": model.id,
                    "object": "model",
                    "owned_by": "grserve",
                    "manifest_sha256": model.manifest_sha256,
                    "max_context_tokens": model.max_context_tokens,
                }
                for model in response.models
            ],
        }

    @app.post("/v1/completions", response_model=None)
    async def completions(
        request: CompletionRequest, http_request: Request
    ) -> StreamingResponse | JSONResponse | Response:
        request_id = secrets.token_urlsafe(18)
        rpc_request = engine_pb2.CompletionRequest(
            request_id=request_id,
            model_id=request.model,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            timeout_ms=config.request_timeout_ms,
            decode_mode=engine_pb2.DECODE_MODE_GREEDY,
        )
        start_task = asyncio.create_task(client.start_completion(rpc_request))
        disconnect_task = asyncio.create_task(_watch_disconnect(http_request))
        _track_task(start_task, tracked_tasks)
        _track_task(disconnect_task, tracked_tasks)
        done, _ = await asyncio.wait(
            {start_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if disconnect_task in done:
            if not start_task.done():
                start_task.cancel()
                with anyio.CancelScope(shield=True):
                    await _wait_bounded(
                        start_task, config.cancel_timeout_ms / 1000
                    )
            elif not start_task.cancelled() and start_task.exception() is None:
                await _run_cleanup_shielded(
                    start_task.result(),
                    cancel=True,
                    config=config,
                    tracked=tracked_tasks,
                )
            return Response(status_code=499)
        disconnect_task.cancel()
        with anyio.CancelScope(shield=True):
            await _wait_bounded(disconnect_task, config.cancel_timeout_ms / 1000)
        try:
            stream = start_task.result()
        except RpcError as error:
            return _error_response(map_rpc_error(error))
        except TransportError:
            return _error_response(_transport_error())
        except ProtocolError:
            return _error_response(_internal_error(request_id))

        encoder = SseEncoder(stream.accepted.request_id, request=request)

        async def body() -> AsyncIterator[bytes]:
            terminal = False
            try:
                try:
                    async for envelope in stream:
                        payload = envelope.WhichOneof("payload")
                        if payload == "token":
                            yield encoder.token(envelope.token)
                        elif payload == "finished":
                            yield encoder.finished(envelope.finished)
                            terminal = True
                            yield encoder.done()
                            return
                        else:
                            raise ProtocolError(
                                f"unexpected completion payload: {payload}"
                            )
                except RpcError as error:
                    yield encoder.error(map_rpc_error(error))
                    terminal = True
                    yield encoder.done()
                except TransportError:
                    yield encoder.error(
                        _transport_error(stream.accepted.request_id)
                    )
                    terminal = True
                    yield encoder.done()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    yield encoder.error(_internal_error(stream.accepted.request_id))
                    terminal = True
                    yield encoder.done()
            finally:
                await _run_cleanup_shielded(
                    stream,
                    cancel=not terminal,
                    config=config,
                    tracked=tracked_tasks,
                )

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app
