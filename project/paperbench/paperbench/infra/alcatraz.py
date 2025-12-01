import io
import os
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import AsyncGenerator, TypedDict

import blobfile as bf
import structlog.stdlib

from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    ExecutionResult,
)
from paperbench.utils import build_canonical_sub_path

logger = structlog.stdlib.get_logger(component=__name__)


async def extract_file_from_computer(
    computer: ComputerInterface,
    path_on_computer: Path,
    extract_to: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
) -> None:
    """
    Extracts a file from the computer.

    Args:
        computer: the computer to upload the file from
        path_on_computer: the path to the file (on the computer) to upload
        extract_to: the path to upload the file to
    """

    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )

    result = await computer.send_shell_command(f"ls -l {path_on_computer}")
    if result.exit_code != 0:
        ctx_logger.warning(f"File {path_on_computer} does not exist on the computer.")
        return

    files_tar = await computer.download(path_on_computer.as_posix())
    bf.write_bytes(str(extract_to), files_tar)
    ctx_logger.info(
        f"Extracted {path_on_computer} to {extract_to} with exit code {result.exit_code}"
    )


async def put_file_in_computer(
    computer: ComputerInterface,
    blobfile_path: str,
    dest_path: str | Path,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
) -> None:
    """
    Puts a file on a computer

    Args:
        computer: the computer to download the file to
        blobfile_path: the path to the file, compatible with blobfile
        dest_path: the path in the cluster to download the file to
    """
    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )

    result = await computer.send_shell_command(f"mkdir -p {Path(dest_path).parent}")
    if result.exit_code != 0:
        ctx_logger.warning(f"Failed to create directory {Path(dest_path).parent} on the cluster.")
        return

    # Place the file in the computer
    ctx_logger.info(f"Placing file in computer: {blobfile_path}")
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
    run_group_id: str,
    runs_dir: str,
    run_id: str,
    timestamp: str | None = None,
) -> None:
    """
    Tars all source directories and files into a single tarball and uploads it
    """

    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())

    fpath = build_canonical_sub_path(run_dir, timestamp)
    container_tmp_dir = Path("/") / "tmp" / "submissions" / timestamp
    container_tar_path = Path("/") / "tmp" / "submissions" / f"{timestamp}.tar.gz"

    if not fpath.startswith("az://"):
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)

    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )
    ctx_logger.info(f"Creating tar for {sources} and uploading to {fpath}")
    await computer.check_shell_command(f"mkdir -p {container_tmp_dir}")

    for source in sources:
        # Create the source directory if it doesn't exist. This is a non-destructive operation;
        # if the directory already exists, this is equivalent to a no-op.
        await computer.check_shell_command(f"mkdir -p {source}")
        await computer.check_shell_command(f"cp -rp {source} {container_tmp_dir}")

    excluded = await populate_exclude_list(computer, container_tmp_dir, "10M")

    for path in excluded.output.decode("utf-8").strip().splitlines():
        ctx_logger.info(f"Excluding file from submission zip (> 10MB): {path}")

    cmds = [
        f"ARCHIVE_PATH={container_tar_path}",
        "EXCLUDE_LIST=/tmp/exclude.txt",
        f"tar -czf $ARCHIVE_PATH -X $EXCLUDE_LIST -C {container_tmp_dir} .",
    ]

    await computer.check_shell_command(" && ".join(cmds))

    await extract_file_from_computer(
        computer=computer,
        path_on_computer=container_tar_path,
        extract_to=fpath,
        run_group_id=run_group_id,
        runs_dir=runs_dir,
        run_id=run_id,
    )

    # cleanup tmp dirs
    await computer.check_shell_command(f"rm -rf {container_tmp_dir}")
    await computer.check_shell_command(f"rm -rf {container_tar_path}")


async def tar_and_extract_from_computer(
    computer: ComputerInterface,
    dir_path_on_computer: Path,
    tar_path_on_computer: Path,
    tar_path_on_target: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
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
        run_group_id=run_group_id,
        runs_dir=runs_dir,
        run_id=run_id,
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


async def read_text_on_computer(
    computer: ComputerInterface, file_path: Path, max_bytes: int | None = None
) -> str:
    """
    Try to read a file, with robustness to different encodings.
    (Without this, we sometimes get `'utf-8' codec can't decode byte 0xa4 in position 64: invalid start byte`)
    """
    file_limiting_string = f" -c {max_bytes}" if max_bytes else ""
    result = await computer.check_shell_command(f"head {file_limiting_string} {file_path}")
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


class WalkDirEntry(TypedDict):
    dirs: list[str]  # sub-directories
    files: list[str]  # file names
    mtimes: list[float]  # modification times (epoch seconds)


async def walk_dir_with_mtimes_on_computer(
    computer: ComputerInterface,
    dir_path: Path,
    warn_threshold: int = 1_000_000,  # One million entries
) -> AsyncGenerator[tuple[str, list[str], list[str], list[float]], None]:
    """
    Asynchronously walk a directory tree on the remote computer with a single find command.
    Yields (current_directory, subdirectory_names, file_names, file_mtimes) in top-down,
    depth-first order, similar to os.walk.
    """

    root_path = os.path.normpath(str(dir_path))

    cmd_count = f"find '{root_path}' -mindepth 0 -print | wc -l"
    result_count = await computer.check_shell_command(cmd_count)
    try:
        entry_count = int(result_count.output.decode("utf-8").strip())
    except ValueError:
        entry_count = None
    if entry_count is not None and entry_count > warn_threshold:
        logger.warning(
            f"WARNING: Directory '{root_path}' contains {entry_count:,} entries. "
            "This may require a large amount of memory to process."
        )

    cmd_list = f"find '{root_path}' -mindepth 0 -printf '%y|%Y|%T@|%p\\n'"

    result_list = await computer.check_shell_command(cmd_list)
    lines = result_list.output.decode("utf-8").splitlines()

    # parent_path → {"dirs": [...], "files": [...], "mtimes": [...]}}
    walk_dir_adjacency: dict[str, WalkDirEntry] = defaultdict(
        lambda: {"dirs": [], "files": [], "mtimes": []}
    )
    for line in lines:
        try:
            # d,f,l; d,f,?; modified time; full path
            base_type, target_type, mtime_str, full_path = line.split("|", 3)
        except ValueError:
            continue  # skip malformed lines

        full_path = os.path.normpath(full_path)

        if full_path == root_path:
            # root directory itself; ensure it's in adjacency
            if full_path not in walk_dir_adjacency:
                walk_dir_adjacency[full_path] = {"dirs": [], "files": [], "mtimes": []}
            continue

        parent = os.path.dirname(full_path)
        name = os.path.basename(full_path)

        is_real_dir = base_type == "d"
        is_symlink_dir = base_type == "l" and target_type == "d"

        if is_real_dir or is_symlink_dir:
            walk_dir_adjacency[parent]["dirs"].append(name)
            if is_real_dir and full_path not in walk_dir_adjacency:
                walk_dir_adjacency[full_path] = {"dirs": [], "files": [], "mtimes": []}
        else:
            walk_dir_adjacency[parent]["files"].append(name)
            try:
                walk_dir_adjacency[parent]["mtimes"].append(float(mtime_str))
            except ValueError:
                walk_dir_adjacency[parent]["mtimes"].append(float("nan"))

    for walkdir_entry in walk_dir_adjacency.values():
        walkdir_entry["dirs"].sort()
        pairs = sorted(zip(walkdir_entry["files"], walkdir_entry["mtimes"]), key=lambda x: x[0])
        walkdir_entry["files"] = [p[0] for p in pairs]
        walkdir_entry["mtimes"] = [p[1] for p in pairs]

    # actual os.walk simulation
    async def _walk(
        current_path: str,
    ) -> AsyncGenerator[tuple[str, list[str], list[str], list[float]], None]:
        try:
            subdirs = walk_dir_adjacency[current_path]["dirs"]
            files = walk_dir_adjacency[current_path]["files"]
            mtimes = walk_dir_adjacency[current_path]["mtimes"]

            yield (current_path, subdirs, files, mtimes)

            for d in subdirs:
                next_path = os.path.join(current_path, d)
                if next_path in walk_dir_adjacency:  # skip symlink dirs we've already seen
                    async for sub_entry in _walk(next_path):
                        yield sub_entry
        except RecursionError:
            logger.warning(
                "RecursionError: Directory tree is too deep to walk in its entirety. Stopping traversal."
            )
            return

    if root_path in walk_dir_adjacency:
        async for entry in _walk(root_path):
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
