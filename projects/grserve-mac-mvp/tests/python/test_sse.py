from __future__ import annotations

import json

import pytest

from grserve_api._generated import engine_pb2
from grserve_api.errors import map_rpc_error
from grserve_api.schemas import CompletionRequest
from grserve_api.sse import DONE_FRAME, SseEncoder


def sse_encoder(
    request_id: str = "req", *, prompt_token_count: int = 3, max_tokens: int = 1
) -> SseEncoder:
    request = CompletionRequest(
        model="mini-qwen38-hybrid",
        prompt_token_ids=[1] * prompt_token_count,
        max_tokens=max_tokens,
        stream=True,
        temperature=0,
        top_p=1,
        n=1,
    )
    return SseEncoder(
        request_id=request_id,
        request=request,
    )


def parse_one_byte_at_a_time(payload: bytes) -> list[object]:
    buffer = bytearray()
    events: list[object] = []
    for byte in payload:
        buffer.append(byte)
        while b"\n\n" in buffer:
            raw, _, remainder = buffer.partition(b"\n\n")
            buffer = bytearray(remainder)
            assert raw.startswith(b"data: ")
            data = raw.removeprefix(b"data: ")
            events.append("[DONE]" if data == b"[DONE]" else json.loads(data))
    assert buffer == b""
    return events


def test_encodes_token_and_final_frames_as_incrementally_parseable_utf8() -> None:
    request_id = "opaque:\n\x00雪😀"
    encoder = sse_encoder(request_id)
    token = engine_pb2.TokenEvent(
        request_id=request_id,
        transaction_id="txn-private",
        sequence_number=1,
        token_id=9,
    )
    finished = engine_pb2.CompletionFinished(
        request_id=request_id,
        finish_reason=engine_pb2.FINISH_REASON_LENGTH,
        last_sequence_number=1,
        usage=engine_pb2.Usage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )

    wire = encoder.token(token) + encoder.finished(finished) + encoder.done()
    events = parse_one_byte_at_a_time(wire)

    assert events == [
        {
            "id": request_id,
            "object": "text_completion",
            "model": "mini-qwen38-hybrid",
            "sequence": 1,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "token_ids": [9],
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": request_id,
            "object": "text_completion",
            "model": "mini-qwen38-hybrid",
            "sequence": 1,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "token_ids": [],
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        },
        "[DONE]",
    ]
    assert wire.endswith(DONE_FRAME)
    assert wire.count(DONE_FRAME) == 1
    assert b"txn-private" not in wire
    assert b"\n" not in wire.split(b"\n\n", 1)[0]
    assert "雪".encode() in wire


def test_finished_frame_exposes_only_the_frozen_length_reason() -> None:
    encoder = sse_encoder()
    encoder.token(
        engine_pb2.TokenEvent(request_id="req", sequence_number=1, token_id=9)
    )
    finished = engine_pb2.CompletionFinished(
        request_id="req",
        finish_reason=engine_pb2.FINISH_REASON_LENGTH,
        last_sequence_number=1,
        usage=engine_pb2.Usage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )

    event = parse_one_byte_at_a_time(encoder.finished(finished))[0]

    assert isinstance(event, dict)
    assert event["choices"][0]["finish_reason"] == "length"

    cancelled = sse_encoder()
    cancelled.token(
        engine_pb2.TokenEvent(request_id="req", sequence_number=1, token_id=9)
    )
    with pytest.raises(ValueError, match="finish reason"):
        cancelled.finished(
            engine_pb2.CompletionFinished(
                request_id="req",
                finish_reason=engine_pb2.FINISH_REASON_CANCELLED,
                last_sequence_number=1,
                usage=engine_pb2.Usage(
                    prompt_tokens=3, completion_tokens=1, total_tokens=4
                ),
            )
        )


def test_post_accept_error_is_sanitized_and_terminated_by_one_done() -> None:
    encoder = sse_encoder("请求-雪")
    rpc_error = engine_pb2.RpcError(
        code=engine_pb2.ERROR_CODE_RANK_FAILURE,
        message="Traceback /tmp/engine.sock backend.dylib fault plan",
        request_id="请求-雪",
        retryable=False,
    )

    wire = encoder.error(map_rpc_error(rpc_error)) + encoder.done()
    events = parse_one_byte_at_a_time(wire)

    assert events == [
        {
            "error": {
                "message": "The Engine is temporarily unavailable.",
                "type": "grserve_error",
                "code": "RANK_FAILURE",
                "request_id": "请求-雪",
                "retryable": True,
            }
        },
        "[DONE]",
    ]
    assert wire.count(DONE_FRAME) == 1
    for secret in (b"Traceback", b"/tmp", b".sock", b".dylib", b"fault plan"):
        assert secret not in wire


def test_encoder_rejects_id_mismatch_invalid_values_and_duplicate_terminal() -> None:
    encoder = sse_encoder()
    with pytest.raises(ValueError, match="request id"):
        encoder.token(
            engine_pb2.TokenEvent(request_id="other", sequence_number=1, token_id=1)
        )
    with pytest.raises(ValueError, match="sequence"):
        encoder.token(
            engine_pb2.TokenEvent(request_id="req", sequence_number=0, token_id=1)
        )
    with pytest.raises(ValueError, match="token"):
        encoder.token(
            engine_pb2.TokenEvent(request_id="req", sequence_number=1, token_id=16)
        )
    with pytest.raises(ValueError, match="finish reason"):
        encoder.finished(
            engine_pb2.CompletionFinished(
                request_id="req", finish_reason=999, usage=engine_pb2.Usage()
            )
        )

    terminal = map_rpc_error(
        engine_pb2.RpcError(code=engine_pb2.ERROR_CODE_INTERNAL, request_id="req")
    )
    encoder.error(terminal)
    assert encoder.done() == b"data: [DONE]\n\n"
    with pytest.raises(RuntimeError, match="terminal"):
        encoder.token(
            engine_pb2.TokenEvent(request_id="req", sequence_number=1, token_id=1)
        )
    with pytest.raises(RuntimeError, match="DONE"):
        encoder.done()


def test_finished_is_terminal_and_done_is_required_after_it() -> None:
    encoder = sse_encoder()
    encoder.token(
        engine_pb2.TokenEvent(request_id="req", sequence_number=1, token_id=1)
    )
    encoder.finished(
        engine_pb2.CompletionFinished(
            request_id="req",
            finish_reason=engine_pb2.FINISH_REASON_LENGTH,
            last_sequence_number=1,
            usage=engine_pb2.Usage(
                prompt_tokens=3, completion_tokens=1, total_tokens=4
            ),
        )
    )

    with pytest.raises(RuntimeError, match="terminal"):
        encoder.error(
            map_rpc_error(
                engine_pb2.RpcError(
                    code=engine_pb2.ERROR_CODE_INTERNAL, request_id="req"
                )
            )
        )
    assert encoder.done() == DONE_FRAME


def test_encoder_enforces_sequence_and_present_usage_from_real_protobuf() -> None:
    encoder = sse_encoder(max_tokens=2)
    with pytest.raises(ValueError, match="sequence"):
        encoder.token(
            engine_pb2.TokenEvent(request_id="req", sequence_number=2, token_id=1)
        )
    encoder.token(
        engine_pb2.TokenEvent(request_id="req", sequence_number=1, token_id=1)
    )
    with pytest.raises(ValueError, match="sequence"):
        encoder.token(
            engine_pb2.TokenEvent(request_id="req", sequence_number=1, token_id=2)
        )
    with pytest.raises(ValueError, match="sequence"):
        encoder.finished(
            engine_pb2.CompletionFinished(
                request_id="req",
                finish_reason=engine_pb2.FINISH_REASON_LENGTH,
                last_sequence_number=2,
                usage=engine_pb2.Usage(
                    prompt_tokens=3, completion_tokens=2, total_tokens=5
                ),
            )
        )
    with pytest.raises(ValueError, match="usage"):
        encoder.finished(
            engine_pb2.CompletionFinished(
                request_id="req",
                finish_reason=engine_pb2.FINISH_REASON_LENGTH,
                last_sequence_number=2,
            )
        )


@pytest.mark.parametrize(
    ("prompt_tokens", "completion_tokens", "total_tokens", "last_sequence"),
    [
        (0, 0, 0, 0),
        (2, 1, 3, 1),
        (3, 0, 3, 0),
        (3, 2, 5, 2),
        (3, 1, 5, 1),
        (4, 1, 5, 1),
    ],
)
def test_finished_requires_exact_usage_from_the_bound_request(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    last_sequence: int,
) -> None:
    encoder = sse_encoder()
    if last_sequence != 0:
        encoder.token(
            engine_pb2.TokenEvent(request_id="req", sequence_number=1, token_id=1)
        )
    with pytest.raises(ValueError, match="usage|sequence"):
        encoder.finished(
            engine_pb2.CompletionFinished(
                request_id="req",
                finish_reason=engine_pb2.FINISH_REASON_LENGTH,
                last_sequence_number=last_sequence,
                usage=engine_pb2.Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
            )
        )


def test_encoder_rejects_tokens_beyond_bound_max_tokens() -> None:
    encoder = sse_encoder(max_tokens=1)
    encoder.token(
        engine_pb2.TokenEvent(request_id="req", sequence_number=1, token_id=1)
    )
    with pytest.raises(ValueError, match="max_tokens"):
        encoder.token(
            engine_pb2.TokenEvent(request_id="req", sequence_number=2, token_id=2)
        )
