import io
import tarfile

import pytest

from paperbench.tar_utils import safe_extractall


def _tar_with_file(name: str, contents: bytes = b"contents") -> io.BytesIO:
    fileobj = io.BytesIO()
    with tarfile.open(fileobj=fileobj, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(contents)
        tar.addfile(info, io.BytesIO(contents))
    fileobj.seek(0)
    return fileobj


def _tar_with_symlink(name: str, linkname: str) -> io.BytesIO:
    fileobj = io.BytesIO()
    with tarfile.open(fileobj=fileobj, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.type = tarfile.SYMTYPE
        info.linkname = linkname
        tar.addfile(info)
    fileobj.seek(0)
    return fileobj


def test_safe_extractall_extracts_regular_files(tmp_path):
    with tarfile.open(fileobj=_tar_with_file("submission/result.txt"), mode="r") as tar:
        safe_extractall(tar, tmp_path)

    assert (tmp_path / "submission" / "result.txt").read_bytes() == b"contents"


def test_safe_extractall_rejects_path_traversal(tmp_path):
    outside_file = tmp_path.parent / "escaped.txt"

    with tarfile.open(fileobj=_tar_with_file("../escaped.txt"), mode="r") as tar:
        with pytest.raises(tarfile.TarError):
            safe_extractall(tar, tmp_path)

    assert not outside_file.exists()


def test_safe_extractall_keeps_absolute_paths_inside_destination(tmp_path):
    outside_file = tmp_path.parent / "absolute.txt"

    with tarfile.open(fileobj=_tar_with_file(str(outside_file)), mode="r") as tar:
        safe_extractall(tar, tmp_path)

    assert not outside_file.exists()
    assert (tmp_path / str(outside_file).lstrip("/")).read_bytes() == b"contents"


def test_safe_extractall_rejects_symlinks_outside_destination(tmp_path):
    outside_file = tmp_path.parent / "linked.txt"

    with tarfile.open(fileobj=_tar_with_symlink("submission/link", "../../linked.txt"), mode="r") as tar:
        with pytest.raises(tarfile.TarError):
            safe_extractall(tar, tmp_path)

    assert not outside_file.exists()
