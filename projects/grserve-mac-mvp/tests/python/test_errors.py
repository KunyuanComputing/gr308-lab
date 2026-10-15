from __future__ import annotations

from types import SimpleNamespace

import pytest

from grserve_api._generated import engine_pb2
from grserve_api.errors import map_rpc_error


@pytest.mark.parametrize(
    ("code", "status", "name", "message", "retryable"),
    [
        (
            engine_pb2.ERROR_CODE_INVALID_ARGUMENT,
            400,
            "INVALID_ARGUMENT",
            "The request is invalid.",
            False,
        ),
        (
            engine_pb2.ERROR_CODE_INCOMPATIBLE_VERSION,
            503,
            "INCOMPATIBLE_VERSION",
            "The Engine is incompatible with this API.",
            False,
        ),
        (
            engine_pb2.ERROR_CODE_UNSUPPORTED_CAPABILITY,
            400,
            "UNSUPPORTED_CAPABILITY",
            "The requested capability is not supported.",
            False,
        ),
        (
            engine_pb2.ERROR_CODE_RESOURCE_EXHAUSTED,
            429,
            "RESOURCE_EXHAUSTED",
            "The Engine is temporarily at capacity.",
            True,
        ),
        (
            engine_pb2.ERROR_CODE_DEADLINE_EXCEEDED,
            504,
            "DEADLINE_EXCEEDED",
            "The request deadline was exceeded.",
            True,
        ),
        (
            engine_pb2.ERROR_CODE_RANK_FAILURE,
            503,
            "RANK_FAILURE",
            "The Engine is temporarily unavailable.",
            True,
        ),
        (
            engine_pb2.ERROR_CODE_REPLICA_FATAL,
            503,
            "REPLICA_FATAL",
            "The model replica is unavailable.",
            False,
        ),
        (
            engine_pb2.ERROR_CODE_INTERNAL,
            500,
            "INTERNAL",
            "An internal error occurred.",
            False,
        ),
    ],
)
def test_all_rpc_errors_have_a_fixed_public_mapping(
    code: int, status: int, name: str, message: str, retryable: bool
) -> None:
    rpc_error = engine_pb2.RpcError(
        code=code,
        message="Traceback /tmp/grserve.sock /opt/private/backend.dylib "
        "plugin.so fault plan rank=1",
        request_id="opaque:\n雪",
        transaction_id="private-txn",
        retryable=not retryable,
    )

    mapped = map_rpc_error(rpc_error)

    assert mapped.status_code == status
    assert mapped.body == {
        "error": {
            "message": message,
            "type": "grserve_error",
            "code": name,
            "request_id": "opaque:\n雪",
            "retryable": retryable,
        }
    }
    serialized = repr(mapped.body)
    for secret in ("Traceback", "/tmp", ".sock", ".dylib", ".so", "fault plan"):
        assert secret not in serialized


@pytest.mark.parametrize("code", [0, -1, 9, 999, "4", None, True])
def test_unknown_or_invalid_error_codes_normalize_to_internal(code: object) -> None:
    mapped = map_rpc_error(
        SimpleNamespace(
            code=code,
            message="socket /private/engine.sock",
            request_id="req-safe",
            transaction_id="txn-private",
            retryable=True,
        )
    )

    assert mapped.status_code == 500
    assert mapped.body == {
        "error": {
            "message": "An internal error occurred.",
            "type": "grserve_error",
            "code": "INTERNAL",
            "request_id": "req-safe",
            "retryable": False,
        }
    }


def test_invalid_request_id_is_not_coerced_or_leaked() -> None:
    mapped = map_rpc_error(
        SimpleNamespace(code=8, message="private", request_id=object())
    )

    assert mapped.body["error"]["request_id"] == ""
