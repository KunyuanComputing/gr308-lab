from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MODEL_ID = "mini-qwen38-hybrid"
VOCAB_SIZE = 16
MAX_CONTEXT_TOKENS = 64

StrictToken = Annotated[int, Field(strict=True, ge=0, lt=VOCAB_SIZE)]
StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model: Literal["mini-qwen38-hybrid"]
    prompt_token_ids: list[StrictToken]
    max_tokens: StrictPositiveInt
    stream: Literal[True]
    temperature: float
    top_p: float
    n: StrictPositiveInt

    @field_validator("prompt_token_ids", mode="before")
    @classmethod
    def reject_boolean_tokens(cls, value: object) -> object:
        if isinstance(value, list) and any(isinstance(item, bool) for item in value):
            raise ValueError("prompt tokens must be integers")
        return value

    @field_validator("max_tokens", "n", mode="before")
    @classmethod
    def reject_boolean_integers(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("integer fields do not accept booleans")
        return value

    @field_validator("stream", mode="before")
    @classmethod
    def require_boolean_stream(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("stream must be a boolean")
        return value

    @field_validator("temperature", "top_p", mode="before")
    @classmethod
    def require_finite_number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("sampling fields must be numbers")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("sampling fields must be finite")
        return value

    @model_validator(mode="after")
    def validate_frozen_contract(self) -> CompletionRequest:
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if self.temperature != 0:
            raise ValueError("temperature must be zero")
        if self.top_p != 1:
            raise ValueError("top_p must be one")
        if self.n != 1:
            raise ValueError("n must be one")
        if len(self.prompt_token_ids) + self.max_tokens > MAX_CONTEXT_TOKENS:
            raise ValueError("prompt and completion exceed the context limit")
        return self
