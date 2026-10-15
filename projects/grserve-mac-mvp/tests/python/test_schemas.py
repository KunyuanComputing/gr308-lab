from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from grserve_api.schemas import CompletionRequest


def valid_request() -> dict[str, object]:
    return {
        "model": "mini-qwen38-hybrid",
        "prompt_token_ids": [1, 4, 7],
        "max_tokens": 8,
        "stream": True,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
    }


def test_accepts_only_the_frozen_pretokenized_completion_shape() -> None:
    request = CompletionRequest.model_validate(valid_request())

    assert request.model_dump() == valid_request()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other-model"),
        ("model", 7),
        ("prompt_token_ids", []),
        ("prompt_token_ids", [True]),
        ("prompt_token_ids", [1.0]),
        ("prompt_token_ids", ["1"]),
        ("prompt_token_ids", [-1]),
        ("prompt_token_ids", [16]),
        ("prompt_token_ids", "1,4,7"),
        ("max_tokens", 0),
        ("max_tokens", True),
        ("max_tokens", 1.0),
        ("max_tokens", "8"),
        ("stream", False),
        ("stream", 1),
        ("stream", "true"),
        ("temperature", True),
        ("temperature", "0"),
        ("temperature", 0.1),
        ("temperature", math.nan),
        ("temperature", math.inf),
        ("temperature", -math.inf),
        ("temperature", 10**400),
        ("top_p", True),
        ("top_p", "1"),
        ("top_p", 0.9),
        ("top_p", math.nan),
        ("top_p", math.inf),
        ("top_p", -(10**400)),
        ("n", True),
        ("n", 1.0),
        ("n", "1"),
        ("n", 2),
    ],
)
def test_rejects_invalid_and_coerced_field_values(field: str, value: object) -> None:
    body = valid_request()
    body[field] = value

    with pytest.raises(ValidationError):
        CompletionRequest.model_validate(body)


@pytest.mark.parametrize("missing", list(valid_request()))
def test_every_request_field_is_required(missing: str) -> None:
    body = valid_request()
    del body[missing]

    with pytest.raises(ValidationError):
        CompletionRequest.model_validate(body)


def test_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest.model_validate({**valid_request(), "prompt": "secret"})


@pytest.mark.parametrize(
    ("prompt_size", "max_tokens"),
    [(1, 63), (63, 1), (64, 1), (3, 62)],
)
def test_context_limit_is_prompt_plus_completion(
    prompt_size: int, max_tokens: int
) -> None:
    body = valid_request()
    body["prompt_token_ids"] = [0] * prompt_size
    body["max_tokens"] = max_tokens

    if prompt_size + max_tokens == 64:
        assert CompletionRequest.model_validate(body).max_tokens == max_tokens
    else:
        with pytest.raises(ValidationError):
            CompletionRequest.model_validate(body)
