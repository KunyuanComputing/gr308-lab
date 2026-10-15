from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from grserve_api._generated import engine_pb2
from grserve_api.app import create_app
from grserve_api.config import ApiConfig
from grserve_api.main import parse_args
from grserve_api.rpc_client import RpcError, TransportError


VALID_REQUEST = {
    "model": "mini-qwen38-hybrid",
    "prompt_token_ids": [1, 4, 7],
    "max_tokens": 1,
    "stream": True,
    "temperature": 0,
    "top_p": 1,
    "n": 1,
}


class FakeStream:
    def __init__(self, request_id: str) -> None:
        self.accepted = engine_pb2.CompletionAccepted(request_id=request_id)
        self.closed = 0

    def __aiter__(self) -> AsyncIterator[engine_pb2.ServerEnvelope]:
        return self.events()

    async def events(self) -> AsyncIterator[engine_pb2.ServerEnvelope]:
        token = engine_pb2.ServerEnvelope()
        token.token.request_id = self.accepted.request_id
        token.token.transaction_id = "private-transaction"
        token.token.sequence_number = 1
        token.token.token_id = 9
        yield token
        finished = engine_pb2.ServerEnvelope()
        finished.finished.request_id = self.accepted.request_id
        finished.finished.finish_reason = engine_pb2.FINISH_REASON_LENGTH
        finished.finished.last_sequence_number = 1
        finished.finished.usage.prompt_tokens = 3
        finished.finished.usage.completion_tokens = 1
        finished.finished.usage.total_tokens = 4
        yield finished

    async def cancel(self, reason: int) -> None:
        raise AssertionError(f"completed streams must not be cancelled: {reason}")

    async def aclose(self) -> None:
        self.closed += 1


class FakeEngine:
    def __init__(self, pre_accept_error: engine_pb2.RpcError | None = None) -> None:
        self.pre_accept_error = pre_accept_error
        self.completion_requests: list[engine_pb2.CompletionRequest] = []
        self.stream: FakeStream | None = None

    async def health(self) -> engine_pb2.HealthResponse:
        return engine_pb2.HealthResponse(
            engine_build_id="build-public",
            accepting_requests=True,
            replica_state=engine_pb2.REPLICA_STATE_HEALTHY,
        )

    async def list_models(self) -> engine_pb2.ListModelsResponse:
        return engine_pb2.ListModelsResponse(
            models=[
                engine_pb2.ModelInfo(
                    id="mini-qwen38-hybrid",
                    manifest_sha256="a" * 64,
                    max_context_tokens=64,
                )
            ]
        )

    async def start_completion(
        self, request: engine_pb2.CompletionRequest
    ) -> FakeStream:
        self.completion_requests.append(request)
        if self.pre_accept_error is not None:
            raise RpcError(self.pre_accept_error)
        self.stream = FakeStream(request.request_id)
        return self.stream


class PostAcceptErrorStream(FakeStream):
    async def events(self) -> AsyncIterator[engine_pb2.ServerEnvelope]:
        raise RpcError(
            engine_pb2.RpcError(
                code=engine_pb2.ERROR_CODE_RANK_FAILURE,
                message="Traceback /tmp/private.sock fault plan",
                request_id=self.accepted.request_id,
            )
        )
        if False:
            yield engine_pb2.ServerEnvelope()


class PostAcceptErrorEngine(FakeEngine):
    async def start_completion(
        self, request: engine_pb2.CompletionRequest
    ) -> FakeStream:
        self.completion_requests.append(request)
        self.stream = PostAcceptErrorStream(request.request_id)
        return self.stream


class ArbitraryPostAcceptErrorStream(FakeStream):
    def __init__(self, request_id: str, error: Exception) -> None:
        super().__init__(request_id)
        self.error = error

    async def events(self) -> AsyncIterator[engine_pb2.ServerEnvelope]:
        raise self.error
        if False:
            yield engine_pb2.ServerEnvelope()


class ArbitraryPostAcceptErrorEngine(FakeEngine):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def start_completion(
        self, request: engine_pb2.CompletionRequest
    ) -> FakeStream:
        self.completion_requests.append(request)
        self.stream = ArbitraryPostAcceptErrorStream(request.request_id, self.error)
        return self.stream


def config() -> ApiConfig:
    return ApiConfig(engine_socket=Path("/tmp/grserve-contract.sock"))


async def request(app: object, method: str, path: str, **kwargs: object) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


@pytest.mark.asyncio
async def test_health_and_models_have_fixed_public_shapes() -> None:
    app = create_app(config(), FakeEngine())

    health = await request(app, "GET", "/health")
    models = await request(app, "GET", "/v1/models")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "engine_build_id": "build-public",
        "accepting_requests": True,
        "replica_state": "healthy",
    }
    assert models.status_code == 200
    assert models.json() == {
        "object": "list",
        "data": [
            {
                "id": "mini-qwen38-hybrid",
                "object": "model",
                "owned_by": "grserve",
                "manifest_sha256": "a" * 64,
                "max_context_tokens": 64,
            }
        ],
    }


@pytest.mark.asyncio
async def test_completion_is_accepted_before_streaming_200_and_has_fixed_headers() -> None:
    engine = FakeEngine()
    response = await request(create_app(config(), engine), "POST", "/v1/completions", json=VALID_REQUEST)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert len(engine.completion_requests) == 1
    rpc_request = engine.completion_requests[0]
    assert rpc_request.request_id
    assert rpc_request.model_id == VALID_REQUEST["model"]
    assert list(rpc_request.prompt_token_ids) == [1, 4, 7]
    assert rpc_request.max_tokens == 1
    assert rpc_request.timeout_ms == 30_000
    assert rpc_request.decode_mode == engine_pb2.DECODE_MODE_GREEDY
    assert engine.stream is not None and engine.stream.closed == 1


@pytest.mark.asyncio
async def test_post_accept_rpc_error_stays_200_and_ends_with_one_done() -> None:
    engine = PostAcceptErrorEngine()

    response = await request(
        create_app(config(), engine), "POST", "/v1/completions", json=VALID_REQUEST
    )

    assert response.status_code == 200
    assert response.content.count(b"data: [DONE]\n\n") == 1
    assert b'"code":"RANK_FAILURE"' in response.content
    assert b"Traceback" not in response.content
    assert engine.stream is not None and engine.stream.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "code"),
    [
        (TransportError("/private/engine.sock"), "RANK_FAILURE"),
        (OSError("Traceback /private/engine.sock backend.dylib"), "INTERNAL"),
    ],
)
async def test_post_accept_transport_and_unknown_errors_are_sanitized_sse(
    error: Exception, code: str
) -> None:
    engine = ArbitraryPostAcceptErrorEngine(error)

    response = await request(
        create_app(config(), engine), "POST", "/v1/completions", json=VALID_REQUEST
    )

    assert response.status_code == 200
    assert response.content.count(b"data: [DONE]\n\n") == 1
    assert f'"code":"{code}"'.encode() in response.content
    assert b"/private" not in response.content
    assert b"Traceback" not in response.content
    assert b".sock" not in response.content
    assert b".dylib" not in response.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({}, "Field required"),
        ({**VALID_REQUEST, "stream": False}, "Input should be True"),
        ({**VALID_REQUEST, "extra": 1}, "Extra inputs are not permitted"),
    ],
)
async def test_validation_errors_are_sanitized_400(payload: object, fragment: str) -> None:
    response = await request(create_app(config(), FakeEngine()), "POST", "/v1/completions", json=payload)

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "grserve_error"
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["request_id"] == ""
    assert body["error"]["retryable"] is False
    assert fragment not in body["error"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "status"),
    [
        (engine_pb2.ERROR_CODE_INVALID_ARGUMENT, 400),
        (engine_pb2.ERROR_CODE_RESOURCE_EXHAUSTED, 429),
        (engine_pb2.ERROR_CODE_INTERNAL, 500),
        (engine_pb2.ERROR_CODE_INCOMPATIBLE_VERSION, 503),
        (engine_pb2.ERROR_CODE_RANK_FAILURE, 503),
        (engine_pb2.ERROR_CODE_REPLICA_FATAL, 503),
        (engine_pb2.ERROR_CODE_DEADLINE_EXCEEDED, 504),
    ],
)
async def test_pre_accept_rpc_errors_are_real_json_statuses(code: int, status: int) -> None:
    secret = "Traceback /tmp/private.sock backend.dylib fault plan"
    engine = FakeEngine(
        engine_pb2.RpcError(
            code=code,
            message=secret,
            request_id="opaque-request",
            retryable=False,
        )
    )

    response = await request(create_app(config(), engine), "POST", "/v1/completions", json=VALID_REQUEST)

    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["request_id"] == "opaque-request"
    assert secret not in response.text


def test_main_cli_has_explicit_validated_configuration() -> None:
    options = parse_args(
        [
            "--engine-socket",
            "/tmp/engine.sock",
            "--host",
            "127.0.0.1",
            "--port",
            "9012",
            "--request-timeout-ms",
            "1234",
            "--cancel-timeout-ms",
            "45",
        ]
    )

    assert options.engine_socket == Path("/tmp/engine.sock")
    assert options.host == "127.0.0.1"
    assert options.port == 9012
    assert options.request_timeout_ms == 1234
    assert options.cancel_timeout_ms == 45
