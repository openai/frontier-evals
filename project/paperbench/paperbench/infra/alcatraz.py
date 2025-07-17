import io
import os
import tarfile
import time
from pathlib import Path
from typing import AsyncGenerator

import blobfile as bf
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    ExecutionResult,
)
from structlog.stdlib import BoundLogger

from paperbench.constants import LOGS_DIR


async def extract_file_from_computer(
    computer: ComputerInterface,
    path_on_computer: Path,
    extract_to: str,
    logger: BoundLogger,
) -> None:
    """
    Extracts a file from the computer.

    Args:
        computer: the computer to upload the file from
        path_on_computer: the path to the file (on the computer) to upload
        extract_to: the path to upload the file to
    """

    result = await computer.send_shell_command(f"ls -l {path_on_computer}")
    if result.exit_code != 0:
        logger.warning(f"File {path_on_computer} does not exist on the computer.")
        return

    files_tar = await computer.download(path_on_computer.as_posix())
    bf.write_bytes(str(extract_to), files_tar)
    logger.info(f"Extracted {path_on_computer} to {extract_to} with exit code {result.exit_code}")


async def put_file_in_computer(
    computer: ComputerInterface,
    blobfile_path: str,
    dest_path: str | Path,
    logger: BoundLogger,
) -> None:
    """
    Puts a file on a computer

    Args:
        computer: the computer to download the file to
        blobfile_path: the path to the file, compatible with blobfile
        dest_path: the path in the cluster to download the file to
    """
    result = await computer.send_shell_command(f"mkdir -p {Path(dest_path).parent}")
    if result.exit_code != 0:
        logger.warning(f"Failed to create directory {Path(dest_path).parent} on the cluster.")
        return

    # Place the file in the computer
    logger.info(f"Placing file in computer: {blobfile_path}")
    submission_tar = bf.read_bytes(str(blobfile_path))
    await computer.upload(submission_tar, str(dest_path))


async def populate_exclude_list(
    computer: ComputerInterface,
    dir_path_on_computer: Path,
    max_size: str,
    exclude_list_path: Path | None = None,
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

    return await computer.check_shell_command(" && ".join(cmds))


async def upload_sources(
    computer: ComputerInterface,
    sources: list[str],
    run_dir: Path | str,
    logger: BoundLogger,
    timestamp: str | None = None,
) -> None:
    """
    Tars all source directories and files into a single tarball and uploads it
    """

    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())

    fpath = bf.join(run_dir, "submissions", timestamp, "submission.tar.gz")
    container_tmp_dir = Path("/") / "tmp" / "submissions" / timestamp
    container_tar_path = Path("/") / "tmp" / "submissions" / f"{timestamp}.tar.gz"

    if not fpath.startswith("az://"):
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating tar for {sources} and uploading to {fpath}")
    await computer.check_shell_command(f"mkdir -p {container_tmp_dir}")

    for source in sources:
        # Create the source directory if it doesn't exist. This is a non-destructive operation;
        # if the directory already exists, this is equivalent to a no-op.
        await computer.check_shell_command(f"mkdir -p {source}")
        await computer.check_shell_command(f"cp -rp {source} {container_tmp_dir}")

    excluded = await populate_exclude_list(computer, container_tmp_dir, "10M")

    for path in excluded.output.decode("utf-8").strip().splitlines():
        logger.info(f"Excluding file from submission zip (> 10MB): {path}")

    cmds = [
        f"ARCHIVE_PATH={container_tar_path}",
        "EXCLUDE_LIST=/tmp/exclude.txt",
        f"tar -czf $ARCHIVE_PATH -X $EXCLUDE_LIST -C {container_tmp_dir.parent} '{timestamp}'",
    ]

    await computer.check_shell_command(" && ".join(cmds))

    await extract_file_from_computer(
        computer=computer,
        path_on_computer=container_tar_path,
        extract_to=fpath,
        logger=logger,
    )

    # cleanup tmp dirs
    await computer.check_shell_command(f"rm -rf {container_tmp_dir}")
    await computer.check_shell_command(f"rm -rf {container_tar_path}")


async def count_aisi_basic_agent_messages(
    computer: ComputerInterface,
    agent_log_path: str = "/home/logs/agent.log",  # TODO use .env
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
    max_file_size: str | None = None,
) -> None:
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
    await computer.check_shell_command(cmd)

    await extract_file_from_computer(
        computer=computer,
        path_on_computer=tar_path_on_computer,
        extract_to=tar_path_on_target,
        logger=logger,
    )


async def file_exists_on_computer(
    computer: ComputerInterface,
    file_path: Path,
) -> bool:
    result = await computer.send_shell_command(f"ls {file_path}")
    return result.exit_code == 0


async def file_is_symlink_on_computer(
    computer: ComputerInterface,
    file_path: Path,
) -> bool:
    result = await computer.send_shell_command(f"ls -l {file_path}")
    return result.exit_code == 0 and "->" in result.output.decode("utf-8")


async def read_text_on_computer(computer: ComputerInterface, file_path: Path) -> str:
    """
    Try to read a file, with robustness to different encodings.
    (Without this, we sometimes get `'utf-8' codec can't decode byte 0xa4 in position 64: invalid start byte`)
    """
    result = await computer.check_shell_command(f"cat {file_path}")
    try:
        return result.output.decode("utf-8")
    except UnicodeDecodeError:
        return result.output.decode("latin1")


async def get_mtime_on_computer(computer: ComputerInterface, file_path: Path) -> float:
    """
    Get the last modified time of a file on the computer.
    i.e. equivalent of doing file_path.stat().st_mtime locally.
    """
    result = await computer.check_shell_command(f"stat -c %Y {file_path}")
    return float(result.output.decode("utf-8").strip())


async def walk_dir_on_computer(
    computer: ComputerInterface, dir_path: Path
) -> AsyncGenerator[tuple[str, list[str], list[str]], None]:
    """
    Asynchronously walks a directory on the remote computer, yielding a tuple of
    (current_directory, list_of_directory_names, list_of_file_names) similarly to os.walk.

    This function uses the remote computer's `find` command with a maximum depth of 1.
    It then parses the output, distinguishing directories (identified with type 'd')
    from other file types. Note that symbolic links (type 'l') and other types are treated as files,
    consistent with the default behavior of os.walk when not following symlinks.
    """

    async def _walk(current_path: str) -> AsyncGenerator[tuple[str, list[str], list[str]], None]:
        # Use find to list items in the current directory with their type.
        # -mindepth 1 and -maxdepth 1 constrain the search to the current directory only.
        # The printf format prints the type (e.g., d for directory, f for file, l for symlink)
        # and the basename of the file, separated by a pipe.
        cmd = f"find '{current_path}' -mindepth 1 -maxdepth 1 -printf '%y|%f\\n'"
        result = await computer.send_shell_command(cmd)
        if result.exit_code != 0:
            # If the command fails, simply return and do not yield further.
            return

        dirs = []
        files = []
        output = result.output.decode("utf-8").strip()
        if output:
            for line in output.splitlines():
                # Split each line into its type and filename parts.
                try:
                    typ, name = line.split("|", 1)
                except ValueError:
                    # If the line is not well formatted, skip it.
                    continue
                # Only treat entries of type 'd' as directories.
                if typ == "d":
                    dirs.append(name)
                else:
                    files.append(name)
        # Yield the tuple consistent with os.walk: (root, dirs, files)
        yield (current_path, dirs, files)
        # Recurse into each discovered directory.
        for d in dirs:
            next_path = os.path.join(current_path, d)
            async for entry in _walk(next_path):
                yield entry

    # Start the walk from the provided directory path.
    async for entry in _walk(str(dir_path)):
        yield entry


async def copy_dir_to_computer(
    computer: ComputerInterface, local_dir: Path, remote_dir: str
) -> None:
    """
    Copies the contents of a local directory to the remote computer.

    Args:
        computer: An instance of ComputerInterface.
        local_dir: The local directory to copy.
        remote_dir: An absolute path on the remote computer where the directory should be copied.
    """
    await computer.check_shell_command(f"mkdir -p {remote_dir}")

    # Create a gzipped tar archive of the local directory.
    # Using arcname="." causes the archive to include files relative to the directory's root.
    tar_bytes_io = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes_io, mode="w:gz") as tar:
        tar.add(str(local_dir), arcname=".")
    tar_bytes = tar_bytes_io.getvalue()

    remote_tmp_tar = os.path.join(remote_dir, "temp_upload.tar.gz")
    await computer.upload(tar_bytes, remote_tmp_tar)

    await computer.check_shell_command(f"tar -xzf {remote_tmp_tar} -C {remote_dir}")

    await computer.check_shell_command(f"rm {remote_tmp_tar}")  # clean up
