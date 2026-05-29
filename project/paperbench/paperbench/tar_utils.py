from __future__ import annotations

import inspect
import os
import tarfile
from pathlib import Path


def _is_within_directory(directory: Path, target: Path) -> bool:
    return os.path.commonpath([directory, target]) == str(directory)


def _validate_member(member: tarfile.TarInfo, destination: Path) -> None:
    target = (destination / member.name).resolve()
    if not _is_within_directory(destination, target):
        raise tarfile.TarError(f"Tar member would extract outside destination: {member.name}")

    if member.isdev():
        raise tarfile.TarError(f"Refusing to extract special file from tar archive: {member.name}")

    if member.issym() or member.islnk():
        raise tarfile.TarError(f"Refusing to extract link from tar archive: {member.name}")


def safe_extractall(tar: tarfile.TarFile, path: str | Path) -> None:
    """Extract a tar archive without allowing path traversal or special files."""

    destination = Path(path).resolve()

    if "filter" in inspect.signature(tar.extractall).parameters:
        tar.extractall(path=destination, filter="data")
        return

    members = tar.getmembers()
    for member in members:
        _validate_member(member, destination)

    tar.extractall(path=destination, members=members)
