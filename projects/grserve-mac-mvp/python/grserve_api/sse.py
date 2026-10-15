# mypy: disable-error-code="attr-defined,import-untyped,name-defined"
from __future__ import annotations

import json
from typing import Any

from grserve_api._generated import engine_pb2
from grserve_api.errors import PublicError
from grserve_api.schemas import CompletionRequest, VOCAB_SIZE

DONE_FRAME = b"data: [DONE]\n\n"


def _data_frame(payload: dict[str, Any]) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return b"data: " + encoded + b"\n\n"


class SseEncoder:
    def __init__(
        self,
        request_id: str,
        *,
        request: CompletionRequest,
    ) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        self._request_id = request_id
        self._model = request.model
        self._prompt_token_count = len(request.prompt_token_ids)
        self._max_tokens = request.max_tokens
        self._next_sequence = 1
        self._terminal = False
        self._done = False

    def token(self, event: engine_pb2.TokenEvent) -> bytes:
        self._require_open()
        self._require_request_id(event.request_id)
        if self._next_sequence > self._max_tokens:
            raise ValueError("token sequence exceeds max_tokens")
        if event.sequence_number != self._next_sequence:
            raise ValueError("sequence must increase strictly by one")
        if event.token_id >= VOCAB_SIZE:
            raise ValueError("token is outside the frozen vocabulary")
        frame = _data_frame(
            {
                "id": event.request_id,
                "object": "text_completion",
                "model": self._model,
                "sequence": event.sequence_number,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "token_ids": [event.token_id],
                        "finish_reason": None,
                    }
                ],
            }
        )
        self._next_sequence += 1
        return frame

    def finished(self, event: engine_pb2.CompletionFinished) -> bytes:
        self._require_open()
        self._require_request_id(event.request_id)
        if event.finish_reason != engine_pb2.FINISH_REASON_LENGTH:
            raise ValueError("finish reason is invalid")
        if not event.HasField("usage"):
            raise ValueError("usage is missing")
        usage = event.usage
        if usage.prompt_tokens != self._prompt_token_count:
            raise ValueError("prompt usage does not match the request")
        if usage.completion_tokens != self._max_tokens:
            raise ValueError("completion usage does not match max_tokens")
        if (
            usage.total_tokens == 0
            or usage.total_tokens != self._prompt_token_count + self._max_tokens
        ):
            raise ValueError("usage totals are inconsistent")
        if (
            event.last_sequence_number != self._max_tokens
            or event.last_sequence_number != self._next_sequence - 1
        ):
            raise ValueError("sequence and completion usage are inconsistent")
        self._terminal = True
        return _data_frame(
            {
                "id": event.request_id,
                "object": "text_completion",
                "model": self._model,
                "sequence": event.last_sequence_number,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "token_ids": [],
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
            }
        )

    def error(self, error: PublicError) -> bytes:
        self._require_open()
        self._require_request_id(error.request_id)
        self._terminal = True
        return _data_frame(error.body)

    def done(self) -> bytes:
        if self._done:
            raise RuntimeError("DONE was already encoded")
        if not self._terminal:
            raise RuntimeError("DONE requires a terminal event")
        self._done = True
        return DONE_FRAME

    def _require_open(self) -> None:
        if self._terminal or self._done:
            raise RuntimeError("a terminal event was already encoded")

    def _require_request_id(self, received: str) -> None:
        if received != self._request_id:
            raise ValueError("request id mismatch")
