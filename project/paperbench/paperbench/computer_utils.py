"""Utilities for working with nanoeval computer interfaces."""

from __future__ import annotations

import shlex
import textwrap
import uuid
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from typing import Any, AsyncGenerator

import structlog.stdlib
from dotenv import load_dotenv
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt

from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerConfiguration,
    ComputerInterface,
    ComputerRuntime,
    ExecutionResult,
)
from paperbench.infra.alcatraz import put_file_in_computer
from paperbench.utils import find_dotenv

load_dotenv(find_dotenv())

logger = structlog.stdlib.get_logger(component=__name__)


def _safe_remote_tar_extract_command(tar_path: str, extract_dir: str) -> str:
    """Build a remote command that extracts a tarball without path traversal."""

    script = textwrap.dedent(
        r"""
import inspect
import os
import tarfile
from pathlib import Path


def is_within_directory(directory: Path, target: Path) -> bool:
    return os.path.commonpath([str(directory), str(target)]) == str(directory)


def validate_member(member: tarfile.TarInfo, destination: Path) -> None:
    target = (destination / member.name).resolve()
    if not is_within_directory(destination, target):
        raise tarfile.TarError(f"Tar member would extract outside destination: {member.name}")

    if member.isdev():
        raise tarfile.TarError(f"Refusing to extract special file from tar archive: {member.name}")

    if member.issym() or member.islnk():
        raise tarfile.TarError(f"Refusing to extract link from tar archive: {member.name}")


destination = Path(os.environ["EXTRACT_DIR"]).resolve()
with tarfile.open(os.environ["TAR_PATH"], "r:gz") as archive:
    if "filter" in inspect.signature(archive.extractall).parameters:
        archive.extractall(path=destination, filter="data")
    else:
        members = archive.getmembers()
        for member in members:
            validate_member(member, destination)
        archive.extractall(path=destination, members=members)
"""
    ).strip()
    return (
        f"mkdir -p {shlex.quote(extract_dir)} && \\\n"
        f"TAR_PATH={shlex.quote(tar_path)} EXTRACT_DIR={shlex.quote(extract_dir)} python3 - <<'PY'\n"
        f"{script}\n"
        "PY"
    )


async def put_submission_in_computer(
    computer: ComputerInterface,
    submission_path: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
) -> None:
    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )
    ctx_logger.info(f"Placing submission in computer from: {submission_path}")
    tar_gz_on_computer = "/tmp/logs.tar.gz"
    # Put the tar.gz to the container
    await put_file_in_computer(
        computer=computer,
        blobfile_path=submission_path,
        dest_path=tar_gz_on_computer,
        run_group_id=run_group_id,
        runs_dir=runs_dir,
        run_id=run_id,
    )

    # Extract tar.gz into a unique temp dir to avoid collisions.
    extract_dir = f"/tmp/pb_extract_{uuid.uuid4().hex}"
    cmd = _safe_remote_tar_extract_command(tar_gz_on_computer, extract_dir)
    ctx_logger.info(f"Extracting submission: {cmd}")
    result = await computer.check_shell_command(cmd)

    # Move the submission directory into /submission deterministically
    cmd = f"rm -rf /submission && mv {extract_dir}/submission /submission"
    ctx_logger.info(f"Placing submission to /submission: {cmd}")
    await computer.check_shell_command(cmd)

    # list files in /submission
    result = await computer.check_shell_command("ls -la /submission")
    ctx_logger.info(f"Files in /submission: {result.output.decode('utf-8')}")


@asynccontextmanager
async def start_computer_with_retry(
    computer_runtime: ComputerRuntime,
    computer_config: ComputerConfiguration,
    exception_types: tuple[type[BaseException]] | type[BaseException] | None = None,
    max_attempts: int = 3,
) -> AsyncGenerator[ComputerInterface, None]:
    """Helper method for starting a ComputerInterface given a ComputerRuntime and ComputerConfiguration."""

    def before_sleep(state: RetryCallState) -> None:
        exception = state.outcome.exception() if state.outcome else None
        logger.warning(
            f"Cluster start failed on attempt {state.attempt_number} out of {max_attempts}; "
            f"retrying due to '{exception}'"
        )

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception_type(exception_types if exception_types else ()),
        before_sleep=before_sleep,
        reraise=True,
    ):
        with attempt:
            async with computer_runtime.run(computer_config) as computer:
                yield computer


class ReleasableComputer(ComputerInterface):
    """
    Wrapper around a ComputerInterface that permits explicit early release of the
    delegated computer's resources. Unlike a typical computer object, calls become
    invalid after release(), enabling the caller to hand the machine back before
    outer teardown completes.
    """

    def __init__(self, delegate: ComputerInterface, exit_stack: AsyncExitStack) -> None:
        self._delegate: ComputerInterface | None = delegate
        self._exit_stack = exit_stack
        self._released = False

    def _ensure_active(self) -> ComputerInterface:
        if self._released or self._delegate is None:
            raise RuntimeError("Releasable computer has already been released")
        return self._delegate

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        delegate = self._delegate
        self._delegate = None

        cleanup_stack = self._exit_stack.pop_all()
        try:
            await cleanup_stack.aclose()
        finally:
            if delegate is not None:
                with suppress(Exception):
                    await delegate.stop()

    async def disable_internet(self) -> None:
        await self._ensure_active().disable_internet()

    async def upload(self, file: bytes, destination: str) -> None:
        await self._ensure_active().upload(file, destination)

    async def download(self, file: str) -> bytes:
        return await self._ensure_active().download(file)

    async def send_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        return await self._ensure_active().send_shell_command(cmd, idempotent=idempotent)

    async def fetch_container_names(self) -> list[str]:
        return await self._ensure_active().fetch_container_names()

    async def stop(self) -> None:
        await self.release()

    def __getattr__(self, name: str) -> Any:
        delegate = self._ensure_active()
        return getattr(delegate, name)


async def maybe_release_computer(computer: ComputerInterface) -> None:
    """Call `release()` on a computer if releasable; warn otherwise."""

    if not isinstance(computer, ReleasableComputer):
        logger.warning(
            "Skipping early release for non-releasable computer; will rely on outer context.",
            computer_type=type(computer).__name__,
        )
        return

    await computer.release()
