import time
from pathlib import Path
from typing import Optional

import blobfile as bf
import requests
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    ExecutionResult,
)
from paperbench.constants import LOGS_DIR
from structlog.stdlib import BoundLogger


async def populate_exclude_list(
    computer: ComputerInterface,
    dir_path_on_computer: Path,
    max_size: str,
    exclude_list_path: Optional[Path] = None,
) -> ExecutionResult:
    """
    Populates `exclude_list_path` with the list of files in `dir_path_on_computer` that
    are larger than `max_size`.
    """
    exclude_list_path = exclude_list_path or Path("/tmp") / "exclude.txt"
    cmds = [
        f"MAX_SIZE={max_size}",
        f"EXCLUDE_LIST={exclude_list_path}",
        f"find {dir_path_on_computer} -type f -not -name 'agent.log' -not -name 'inspect.log' -size +$MAX_SIZE -printf '%P\\n' > $EXCLUDE_LIST",
        "cat $EXCLUDE_LIST",
    ]
    excluded = await computer.check_shell_command(" && ".join(cmds))
    return excluded


async def upload_sources(
    computer: ComputerInterface,
    sources: list[str],
    run_dir: Path | str,
    logger: BoundLogger,
    filename: str | None = None,
):
    """
    Tars all source directories and files into a single tarball and uploads it
    """
    if filename is None:
        filename = time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())
    file_path = bf.join(run_dir, f"{filename}.tar.gz")

    container_tmp_dir = Path("/tmp") / f"{filename}"
    container_tar_path = Path("/tmp") / f"{filename}.tar.gz"

    logger.info(f"Creating tar for {sources} and uploading to {file_path}")
    await computer.check_shell_command(f"mkdir -p {container_tmp_dir}")

    for source in sources:
        # Create the source directory if it doesn't exist. This is a non-destructive operation;
        # if the directory already exists, this is equivalent to a no-op.
        await computer.check_shell_command(f"mkdir -p {source}")
        await computer.check_shell_command(f"cp -rp {source} {container_tmp_dir}")

    excluded = await populate_exclude_list(computer, container_tmp_dir, "10M")

    for fpath in excluded.output.decode("utf-8").strip().splitlines():
        logger.info(f"Excluding file from submission zip (> 10MB): {fpath}")

    cmds = [
        f"ARCHIVE_PATH={container_tar_path}",
        "EXCLUDE_LIST=/tmp/exclude.txt",
        f"tar -czf $ARCHIVE_PATH -X $EXCLUDE_LIST -C {container_tmp_dir.parent} '{filename}'",
    ]

    await computer.check_shell_command(" && ".join(cmds))

    tar_stream = await computer.download(container_tar_path.as_posix())
    with open(file_path, "wb") as f:
        f.write(tar_stream)

    # cleanup tmp dirs
    await computer.check_shell_command(f"rm -rf {container_tmp_dir}")
    await computer.check_shell_command(f"rm -rf {container_tar_path}")


async def count_aisi_basic_agent_messages(
    computer: ComputerInterface, agent_log_path: str = "/home/logs/agent.log"  # TODO use .env
) -> int:
    """
    Counts the number of occurences of "╭─ Assistant" in the agent log.
    """
    result = await computer.send_shell_command(f"grep -c '╭─ Assistant' {agent_log_path}")
    if result.exit_code != 0 or not result.output:
        return -1
    count = int(result.output.decode("utf-8").strip())
    return count


async def compute_aisi_basic_agent_runtime(
    computer: ComputerInterface,
    inspect_log_path: str = f"{LOGS_DIR}/inspect.log",
) -> tuple[float | None, float | None, float | None]:
    """
    Parses the inspect.log file to extract the total runtime, productive runtime, and retry time.
    """
    cmd = f"grep 'total runtime: ' {inspect_log_path} | tail -n1 | awk '{{print $8 $12 $16}}'"
    result = await computer.send_shell_command(cmd)
    if result.exit_code != 0 or not result.output:
        return None, None, None
    try:
        runtime_str, productive_str, retry_str = result.output.decode("utf-8").strip().split(",")
        return float(runtime_str), float(productive_str), float(retry_str)
    except (ValueError, IndexError):
        return None, None, None


async def tar_and_extract_from_computer(
    computer: ComputerInterface,
    dir_path_on_computer: Path,
    tar_path_on_computer: Path,
    tar_path_on_target: str,
    logger: BoundLogger,
    max_file_size: Optional[str] = None,
):
    """
    1) Tars the dir at dir_path_on_computer to tar_path_on_computer
    2) Uploads to tar_path_on_target
    """
    # extract the tar of the submission
    exclude_list_path = Path("/tmp") / "exclude.txt"
    if max_file_size is not None:
        await populate_exclude_list(
            computer, dir_path_on_computer, max_file_size, exclude_list_path
        )
    else:
        await computer.check_shell_command(f"touch {exclude_list_path}")

    cmd = f"tar -czf {tar_path_on_computer} -X {exclude_list_path} {dir_path_on_computer}"
    result = await computer.send_shell_command(cmd)
    assert result.exit_code == 0, f"Tar submission failed: {result}"

    file = await computer.download(str(tar_path_on_computer))
    with open(tar_path_on_target, "wb") as f:
        f.write(file)
