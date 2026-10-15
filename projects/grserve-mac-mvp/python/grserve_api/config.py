from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ApiConfig:
    engine_socket: Path
    request_timeout_ms: int = 30_000
    cancel_timeout_ms: int = 250

    def __post_init__(self) -> None:
        if not isinstance(self.engine_socket, Path):
            raise TypeError("engine_socket must be a pathlib.Path")
        socket_text = str(self.engine_socket)
        if not self.engine_socket.is_absolute():
            raise ValueError("engine_socket must be an absolute path")
        if "\x00" in socket_text:
            raise ValueError("engine_socket must not contain NUL")
        try:
            encoded_socket = socket_text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("engine_socket must be UTF-8 encodable") from error
        if len(encoded_socket) > 100:
            raise ValueError("engine_socket must be at most 100 encoded bytes")
        self._validate_timeout("request_timeout_ms", self.request_timeout_ms)
        self._validate_timeout("cancel_timeout_ms", self.cancel_timeout_ms)

    @staticmethod
    def _validate_timeout(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
