import os
import re
from pathlib import Path
from typing import AsyncGenerator, Generator

import structlog.stdlib
from drain3 import TemplateMiner
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface

from paperbench.infra.alcatraz import (
    file_exists_on_computer,
    file_is_symlink_on_computer,
    get_mtime_on_computer,
    read_text_on_computer,
    walk_dir_with_mtimes_on_computer,
)

logger = structlog.stdlib.get_logger(component=__name__)

SIZE_LIMIT_BYTES = 200_000  # read at most 200 kB per file


def safe_read_file(file_path: Path, max_bytes: int | None = None) -> str:
    """
    Try to read a file (up to max_bytes bytes), with robustness to different encodings.
    (Without this, we sometimes get `'utf-8' codec can't decode byte 0xa4 in position 64: invalid start byte`)
    """
    try:
        # Try utf-8 first
        with file_path.open(encoding="utf-8") as f:
            return f.read(max_bytes)
    except UnicodeDecodeError:
        # Try latin1 if utf-8 fails
        with file_path.open(encoding="latin1") as f:
            return f.read(max_bytes)


async def file_exists(file_path: Path, computer: ComputerInterface | None) -> bool:
    """Generic function to check if a file exists; on the computer or locally."""
    return await file_exists_on_computer(computer, file_path) if computer else file_path.exists()


async def read_file_content(file_path: Path, computer: ComputerInterface | None) -> str:
    """Generic function to read the content of a file; on the computer or locally."""
    return (
        await read_text_on_computer(computer, file_path, SIZE_LIMIT_BYTES)
        if computer
        else safe_read_file(file_path, SIZE_LIMIT_BYTES)
    )


async def read_file_mtime(file_path: Path, computer: ComputerInterface | None) -> float:
    """Generic function to read the modification time of a file; on the computer or locally."""
    return (
        await get_mtime_on_computer(computer, file_path) if computer else file_path.stat().st_mtime
    )


async def is_symlink(file_path: Path, computer: ComputerInterface | None) -> bool:
    """Generic function to check if a file is a symlink; on the computer or locally."""
    return (
        await file_is_symlink_on_computer(computer, file_path)
        if computer
        else file_path.is_symlink()
    )


def walk_dir_with_mtimes_locally(
    dir_path: Path,
) -> Generator[tuple[str, list[str], list[str], list[float]], None, None]:
    """like os.walk, but also yields the mtimes of the files"""
    for root, dirs, files in os.walk(dir_path):
        mtimes = []
        for f in files:
            full_path = os.path.join(root, f)
            try:
                st = os.stat(full_path)
                mtimes.append(st.st_mtime)
            except OSError:
                mtimes.append(0.0)
        yield root, dirs, files, mtimes


async def walk_dir_with_mtimes(
    dir_path: Path, computer: ComputerInterface | None
) -> AsyncGenerator[tuple[str, list[str], list[str], list[float]], None]:
    """
    Generic function for running equivalent of os.walk + mtimes; on the computer or locally."""
    if computer:
        async for entry in walk_dir_with_mtimes_on_computer(computer, dir_path):
            yield entry
    else:
        for entry in walk_dir_with_mtimes_locally(dir_path):
            yield entry


def sanitize_line(line: str) -> str:
    """
    Convert ephemeral bits (timestamps, progress bars, numeric tokens, IPs, etc.)
    into placeholders so that repeated patterns can be more easily detected.
    """

    # Mask ISO8601 Timestamps (e.g. 2025-01-28T18:47:06.1465140Z)
    line = re.sub(r"\d{4}-\d{2}-\d{2}T[0-9:.]+Z", "<TIMESTAMP>", line)

    # Mask typical date/time strings (e.g. 2025-01-28 18:47:06 or 18:47:06)
    line = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<DATE>", line)
    line = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "<TIME>", line)

    # TQDM or other progress bars: remove generic progress bar lines by matching percentage and bar or repeated progress symbols
    if (
        re.search(r"\d+%?\|[█=]+", line)
        or re.search(r"[KMG]?B/s", line)
        or re.search(r"\d+%\s*\|", line)
        or re.search(r"[▏▎▍▌▋▊▉]{2,}", line)
    ):
        line = "<PROGRESS_BAR>"

    # IP addresses  (1-3 digits).(1-3).(1-3).(1-3)
    line = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<IP>", line)

    # Mask long hex strings (common in commit hashes, container IDs, etc.)
    line = re.sub(r"\b[0-9a-fA-F]{8,}\b", "<HEX>", line)

    return line


def reduce_log(input_string: str) -> str:
    """
    Reduce a multi-line log string to a filtered version with repeated lines collapsed.
    """
    template_miner = TemplateMiner()
    output_lines = []

    previous_cluster_id = None
    repeat_count = 1

    for raw_line in input_string.splitlines():
        original_line = raw_line
        sanitized = sanitize_line(original_line)

        result = template_miner.add_log_message(sanitized)
        cluster_id = result["cluster_id"]

        if previous_cluster_id is None:
            # First line
            output_lines.append(original_line)
            previous_cluster_id = cluster_id
            continue

        if cluster_id == previous_cluster_id:
            repeat_count += 1
        else:
            if repeat_count > 1:
                output_lines.append(f"  (repeated {repeat_count} times)")
            output_lines.append(original_line)
            repeat_count = 1
            previous_cluster_id = cluster_id

    if previous_cluster_id is not None and repeat_count > 1:
        output_lines.append(f"  (repeated {repeat_count} times)")

    return "\n".join(output_lines)


# just to test the process_log fn
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Process log files and collapse repeated lines.")
    parser.add_argument(
        "-i",
        "--input-file",
        type=Path,
        help="Path to the input log file to be cleaned",
    )

    args = parser.parse_args()

    input_text = args.input_file.read_text()

    print(reduce_log(input_text))


def format_file(file_path: Path, file_content: str) -> str:
    return f"""<FILE:{file_path}>
{file_content if file_content.strip() else "(FILE IS EMPTY)"}
</FILE:{file_path}>"""
