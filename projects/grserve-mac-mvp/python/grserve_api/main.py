from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from grserve_api.app import create_app
from grserve_api.config import ApiConfig


@dataclass(frozen=True, slots=True)
class ServerOptions:
    engine_socket: Path
    host: str
    port: int
    request_timeout_ms: int
    cancel_timeout_ms: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 65535:
        raise argparse.ArgumentTypeError("must be at most 65535")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> ServerOptions:
    parser = argparse.ArgumentParser(prog="grserve-api")
    parser.add_argument("--engine-socket", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8000)
    parser.add_argument("--request-timeout-ms", type=_positive_int, default=30_000)
    parser.add_argument("--cancel-timeout-ms", type=_positive_int, default=250)
    args = parser.parse_args(argv)
    config = ApiConfig(
        engine_socket=args.engine_socket,
        request_timeout_ms=args.request_timeout_ms,
        cancel_timeout_ms=args.cancel_timeout_ms,
    )
    return ServerOptions(
        engine_socket=config.engine_socket,
        host=args.host,
        port=args.port,
        request_timeout_ms=config.request_timeout_ms,
        cancel_timeout_ms=config.cancel_timeout_ms,
    )


def main(argv: Sequence[str] | None = None) -> None:
    options = parse_args(argv)
    config = ApiConfig(
        engine_socket=options.engine_socket,
        request_timeout_ms=options.request_timeout_ms,
        cancel_timeout_ms=options.cancel_timeout_ms,
    )
    uvicorn.run(
        create_app(config),
        host=options.host,
        port=options.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
