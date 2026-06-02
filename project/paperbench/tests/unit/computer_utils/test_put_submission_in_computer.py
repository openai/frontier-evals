from __future__ import annotations

import pytest

from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    ExecutionResult,
    JupyterExecutionResult,
)
from paperbench import computer_utils


class RecordingComputer(ComputerInterface):
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def disable_internet(self) -> None:  # pragma: no cover - not used
        raise NotImplementedError

    async def upload(self, file: bytes, destination: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def download(self, file: str) -> bytes:  # pragma: no cover
        raise NotImplementedError

    async def send_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        self.commands.append(cmd)
        return ExecutionResult(output=b"ok", exit_code=0)

    async def fetch_container_names(self) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    async def stop(self) -> None:  # pragma: no cover
        raise NotImplementedError

    async def execute(self, code: str, timeout: int = 120) -> JupyterExecutionResult:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_put_submission_in_computer_uses_safe_tar_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_files: list[tuple[str, str]] = []

    async def fake_put_file_in_computer(
        *,
        computer: ComputerInterface,
        blobfile_path: str,
        dest_path: str,
        run_group_id: str,
        runs_dir: str,
        run_id: str,
    ) -> None:
        copied_files.append((blobfile_path, dest_path))

    monkeypatch.setattr(computer_utils, "put_file_in_computer", fake_put_file_in_computer)

    computer = RecordingComputer()
    await computer_utils.put_submission_in_computer(
        computer=computer,
        submission_path="/tmp/submission.tar.gz",
        run_group_id="group",
        runs_dir="/tmp/runs",
        run_id="run",
    )

    assert copied_files == [("/tmp/submission.tar.gz", "/tmp/logs.tar.gz")]

    extract_command = computer.commands[0]
    assert "tar -xzf" not in extract_command
    assert "tarfile.open" in extract_command
    assert 'filter="data"' in extract_command
    assert "validate_member" in extract_command

    assert computer.commands[1].startswith("rm -rf /submission && mv /tmp/pb_extract_")
    assert computer.commands[1].endswith("/submission /submission")
    assert computer.commands[2] == "ls -la /submission"
