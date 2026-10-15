from __future__ import annotations

import asyncio
import os
import stat
import tempfile
import time
from pathlib import Path
from types import TracebackType

import anyio

from grserve_api.config import ApiConfig
from grserve_api.rpc_client import EngineClient, RpcClientError


class EngineProcess:
    def __init__(
        self,
        *,
        hal_plugin: Path,
        model_dir: Path | None = None,
        ready_timeout: float = 30.0,
    ) -> None:
        engine_bin = os.environ.get("GRSERVE_ENGINE_BIN")
        if not engine_bin:
            raise RuntimeError("GRSERVE_ENGINE_BIN must name the Engine executable")
        self._engine_bin = Path(engine_bin)
        self._hal_plugin = hal_plugin
        self._root = Path(__file__).resolve().parents[3]
        self._model_dir = model_dir or self._root / "models/mini-qwen38-hybrid"
        self._ready_timeout = ready_timeout
        self._directory: Path | None = None
        self._socket_path: Path | None = None
        self._socket_identity: tuple[int, int, int] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._teardown_task: asyncio.Task[None] | None = None

    @property
    def socket_path(self) -> Path:
        if self._socket_path is None:
            raise RuntimeError("EngineProcess has not been started")
        return self._socket_path

    @property
    def process(self) -> asyncio.subprocess.Process:
        if self._process is None:
            raise RuntimeError("EngineProcess has not been started")
        return self._process

    @property
    def directory_path(self) -> Path:
        if self._directory is None:
            raise RuntimeError("EngineProcess has not been started")
        return self._directory

    async def __aenter__(self) -> EngineProcess:
        temporary_root = Path("/private/tmp") if Path("/private/tmp").is_dir() else None
        self._directory = Path(
            tempfile.mkdtemp(prefix="grserve-api-e2e.", dir=temporary_root)
        )
        os.chmod(self._directory, 0o700)
        self._socket_path = self._directory / "engine.sock"
        try:
            self._process = await asyncio.create_subprocess_exec(
                str(self._engine_bin),
                "--socket",
                str(self._socket_path),
                "--model-dir",
                str(self._model_dir),
                "--hal-plugin",
                str(self._hal_plugin),
                "--ranks=2",
                "--max-concurrent-requests=2",
                "--max-state-slots-per-rank=2",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await self._wait_ready()
            socket_status = self.socket_path.lstat()
            if not stat.S_ISSOCK(socket_status.st_mode):
                raise RuntimeError("Engine ready path is not a Unix socket")
            self._socket_identity = (
                socket_status.st_dev,
                socket_status.st_ino,
                socket_status.st_uid,
            )
            return self
        except BaseException:
            await self._teardown()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._teardown()

    async def _wait_ready(self) -> None:
        deadline = time.monotonic() + self._ready_timeout
        client = EngineClient(
            ApiConfig(engine_socket=self.socket_path, request_timeout_ms=250)
        )
        while time.monotonic() < deadline:
            self._record_socket_identity()
            self._raise_if_exited()
            try:
                health = await client.health()
                if health.accepting_requests:
                    return
            except RpcClientError:
                pass
            await asyncio.sleep(0.01)
        await self._raise_if_exited_after_grace()
        raise TimeoutError("Engine did not become healthy before the ready timeout")

    async def _raise_if_exited_after_grace(self) -> None:
        if self.process.returncode is None:
            try:
                await asyncio.wait_for(self.process.wait(), timeout=0.5)
            except TimeoutError:
                return
        self._raise_if_exited()

    def _raise_if_exited(self) -> None:
        if self.process.returncode is not None:
            raise RuntimeError(
                "Engine exited before health readiness "
                f"with status {self.process.returncode}"
            )

    def _record_socket_identity(self) -> None:
        if self._socket_identity is not None:
            return
        try:
            socket_status = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(socket_status.st_mode):
            self._socket_identity = (
                socket_status.st_dev,
                socket_status.st_ino,
                socket_status.st_uid,
            )

    async def _teardown(self) -> None:
        with anyio.CancelScope(shield=True):
            if self._teardown_task is None:
                self._teardown_task = asyncio.create_task(
                    self._teardown_with_deadline()
                )
            cancellation: asyncio.CancelledError | None = None
            current_task = asyncio.current_task()
            while True:
                try:
                    await asyncio.shield(self._teardown_task)
                    break
                except asyncio.CancelledError as error:
                    if self._teardown_task.cancelled():
                        raise
                    cancellation = cancellation or error
                    if current_task is not None:
                        while current_task.cancelling():
                            current_task.uncancel()

            self._teardown_task.result()
            if cancellation is not None:
                raise cancellation

    async def _teardown_with_deadline(self) -> None:
        with anyio.fail_after(3, shield=True):
            await self._teardown_once()

    async def _teardown_once(self) -> None:
        try:
            if self._process is not None and self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=2)
                except TimeoutError:
                    self._process.kill()
                    await asyncio.wait_for(self._process.wait(), timeout=0.5)
        finally:
            self._remove_owned_paths()

    def _remove_owned_paths(self) -> None:
        if self._socket_path is not None and self._socket_identity is not None:
            try:
                socket_status = self._socket_path.lstat()
            except FileNotFoundError:
                pass
            else:
                identity = (
                    socket_status.st_dev,
                    socket_status.st_ino,
                    socket_status.st_uid,
                )
                if stat.S_ISSOCK(socket_status.st_mode) and identity == self._socket_identity:
                    self._socket_path.unlink()
        if self._directory is not None:
            try:
                self._directory.rmdir()
            except OSError:
                pass
