# mypy: disable-error-code="attr-defined,import-untyped"
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from grserve_api._generated import engine_pb2


class RpcErrorLike(Protocol):
    code: object
    request_id: object


@dataclass(frozen=True, slots=True)
class ErrorMapping:
    status_code: int
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class PublicError:
    status_code: int
    code: str
    message: str
    request_id: str
    retryable: bool

    @property
    def body(self) -> dict[str, dict[str, str | bool]]:
        return {
            "error": {
                "message": self.message,
                "type": "grserve_error",
                "code": self.code,
                "request_id": self.request_id,
                "retryable": self.retryable,
            }
        }


_INTERNAL = ErrorMapping(500, "INTERNAL", "An internal error occurred.", False)

_ERROR_MAPPINGS = {
    engine_pb2.ERROR_CODE_INVALID_ARGUMENT: ErrorMapping(
        400, "INVALID_ARGUMENT", "The request is invalid.", False
    ),
    engine_pb2.ERROR_CODE_INCOMPATIBLE_VERSION: ErrorMapping(
        503,
        "INCOMPATIBLE_VERSION",
        "The Engine is incompatible with this API.",
        False,
    ),
    engine_pb2.ERROR_CODE_UNSUPPORTED_CAPABILITY: ErrorMapping(
        400,
        "UNSUPPORTED_CAPABILITY",
        "The requested capability is not supported.",
        False,
    ),
    engine_pb2.ERROR_CODE_RESOURCE_EXHAUSTED: ErrorMapping(
        429,
        "RESOURCE_EXHAUSTED",
        "The Engine is temporarily at capacity.",
        True,
    ),
    engine_pb2.ERROR_CODE_DEADLINE_EXCEEDED: ErrorMapping(
        504,
        "DEADLINE_EXCEEDED",
        "The request deadline was exceeded.",
        True,
    ),
    engine_pb2.ERROR_CODE_RANK_FAILURE: ErrorMapping(
        503, "RANK_FAILURE", "The Engine is temporarily unavailable.", True
    ),
    engine_pb2.ERROR_CODE_REPLICA_FATAL: ErrorMapping(
        503, "REPLICA_FATAL", "The model replica is unavailable.", False
    ),
    engine_pb2.ERROR_CODE_INTERNAL: _INTERNAL,
}


def map_rpc_error(error: RpcErrorLike) -> PublicError:
    code = error.code
    mapping = _ERROR_MAPPINGS.get(code, _INTERNAL) if type(code) is int else _INTERNAL
    request_id = error.request_id if isinstance(error.request_id, str) else ""
    return PublicError(
        status_code=mapping.status_code,
        code=mapping.code,
        message=mapping.message,
        request_id=request_id,
        retryable=mapping.retryable,
    )
