import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import pytest
from alcatraz.clusters.local import LocalConfig
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface

from paperbench.infra.alcatraz import copy_dir_to_computer, walk_dir_with_mtimes_on_computer
from paperbench.judge.utils import walk_dir_with_mtimes, walk_dir_with_mtimes_locally


@pytest.fixture(scope="function")
async def computer() -> AsyncGenerator[ComputerInterface, None]:
    """
    Fixture that provides an instance of ComputerInterface for testing.
    This version uses an async context manager to yield the computer instance.
    """
    alcatraz_env = LocalConfig(
        image="pb-env",
        pull_from_registry=False,
    )
    async with alcatraz_env.build() as cluster:
        comp = AlcatrazComputerInterface(cluster_value=cluster)
        yield comp


def normalize_relative_walk(
    walk_results: list[tuple[str, list[str], list[str]]], base: str
) -> list[tuple[str, list[str], list[str]]]:
    """
    Normalize the output of os.walk or walk_dir by converting paths to be relative
    to the provided base, sorting subdirectories and files, and sorting by the relative path.
    """
    normalized = []
    for root, dirs, files in walk_results:
        rel_root = os.path.relpath(root, base)
        normalized.append((rel_root, sorted(dirs), sorted(files)))
    return sorted(normalized, key=lambda item: item[0])


# --- Test Cases --- #
@pytest.mark.asyncio
async def test_walk_dir_equivalence_with_copy(computer: ComputerInterface) -> None:
    """
    Test that the asynchronous remote walk (using computer) returns an equivalent
    directory structure as the local os.walk. Creates a basic directory tree,
    copies it to the remote computer, then compares normalized walk outputs.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_base = Path(tmpdirname)
        # Set up the directory structure:
        (local_base / "dir1").mkdir()
        (local_base / "dir2").mkdir()
        (local_base / "dir1" / "subdir1").mkdir(parents=True)
        # Create some files
        (local_base / "file1.txt").write_text("Content of file 1")
        (local_base / "dir1" / "file2.txt").write_text("Content of file 2")
        (local_base / "dir1" / "subdir1" / "file3.txt").write_text("Content of file 3")

        remote_base = "/tmp/test_dir"
        await copy_dir_to_computer(computer, local_base, remote_base)

        local_walk = []
        for root, dirs, files, _ in walk_dir_with_mtimes_locally(local_base):
            local_walk.append((root, dirs, files))
        normalized_local = normalize_relative_walk(local_walk, str(local_base))

        remote_walk = []
        async for root, dirs, files, _ in walk_dir_with_mtimes(Path(remote_base), computer):
            remote_walk.append((root, dirs, files))
        normalized_remote = normalize_relative_walk(remote_walk, remote_base)

        assert normalized_local == normalized_remote, (
            f"Mismatch between local and remote walk outputs:\nLocal: {normalized_local}\nRemote: {normalized_remote}"
        )


@pytest.mark.asyncio
async def test_empty_directory(computer: ComputerInterface) -> None:
    """
    Test that an empty directory is correctly handled.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_base = Path(tmpdirname) / "empty_dir"
        local_base.mkdir()
        remote_base = "/tmp/test_empty_dir"
        await copy_dir_to_computer(computer, local_base, remote_base)

        local_walk = []
        for root, dirs, files, _ in walk_dir_with_mtimes_locally(local_base):
            local_walk.append((root, dirs, files))
        normalized_local = normalize_relative_walk(local_walk, str(local_base))

        remote_walk = []
        async for root, dirs, files, _ in walk_dir_with_mtimes(Path(remote_base), computer):
            remote_walk.append((root, dirs, files))
        normalized_remote = normalize_relative_walk(remote_walk, remote_base)

        assert normalized_local == normalized_remote, (
            f"Empty directory mismatch:\nLocal: {normalized_local}\nRemote: {normalized_remote}"
        )


@pytest.mark.asyncio
async def test_hidden_files(computer: ComputerInterface) -> None:
    """
    Test that hidden files and directories (those starting with a dot) are correctly included.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_base = Path(tmpdirname) / "hidden_test"
        local_base.mkdir()
        # Create hidden files and directories
        (local_base / ".hidden_file.txt").write_text("hidden content")
        (local_base / ".hidden_dir").mkdir()
        (local_base / ".hidden_dir" / "file_in_hidden.txt").write_text("inside hidden dir")

        remote_base = "/tmp/test_hidden_files"
        await copy_dir_to_computer(computer, local_base, remote_base)

        local_walk = []
        for root, dirs, files, _ in walk_dir_with_mtimes_locally(local_base):
            local_walk.append((root, dirs, files))
        normalized_local = normalize_relative_walk(local_walk, str(local_base))

        remote_walk = []
        async for root, dirs, files, _ in walk_dir_with_mtimes(Path(remote_base), computer):
            remote_walk.append((root, dirs, files))
        normalized_remote = normalize_relative_walk(remote_walk, remote_base)

        assert normalized_local == normalized_remote, (
            f"Hidden files mismatch:\nLocal: {normalized_local}\nRemote: {normalized_remote}"
        )


@pytest.mark.asyncio
async def test_symlink(computer: ComputerInterface) -> None:
    """
    Test that symbolic links are preserved and reported similarly to os.walk.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_base = Path(tmpdirname) / "symlink_test"
        local_base.mkdir()
        # Create a target file and a symbolic link pointing to it.
        target_file = local_base / "target.txt"
        target_file.write_text("target file content")
        symlink_path = local_base / "link_to_target.txt"
        os.symlink(target_file, symlink_path)

        remote_base = "/tmp/test_symlink"
        await copy_dir_to_computer(computer, local_base, remote_base)

        local_walk = []
        for root, dirs, files, _ in walk_dir_with_mtimes_locally(local_base):
            local_walk.append((root, dirs, files))
        normalized_local = normalize_relative_walk(local_walk, str(local_base))

        remote_walk = []
        async for root, dirs, files, _ in walk_dir_with_mtimes(Path(remote_base), computer):
            remote_walk.append((root, dirs, files))
        normalized_remote = normalize_relative_walk(remote_walk, remote_base)

        assert normalized_local == normalized_remote, (
            f"Symlink structure mismatch:\nLocal: {normalized_local}\nRemote: {normalized_remote}"
        )


@pytest.mark.asyncio
async def test_deep_nested_structure(computer: ComputerInterface) -> None:
    """
    Test a deeply nested directory structure.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_base = Path(tmpdirname) / "nested_test"
        local_base.mkdir()
        # Create a 5-level deep nested directory structure with one file per level.
        current = local_base
        for i in range(5):
            new_dir = current / f"nested_{i}"
            new_dir.mkdir()
            (new_dir / f"file_{i}.txt").write_text(f"content {i}")
            current = new_dir

        remote_base = "/tmp/test_deep_nested_structure"
        await copy_dir_to_computer(computer, local_base, remote_base)

        local_walk = []
        for root, dirs, files, _ in walk_dir_with_mtimes_locally(local_base):
            local_walk.append((root, dirs, files))
        normalized_local = normalize_relative_walk(local_walk, str(local_base))

        remote_walk = []
        async for root, dirs, files, _ in walk_dir_with_mtimes(Path(remote_base), computer):
            remote_walk.append((root, dirs, files))
        normalized_remote = normalize_relative_walk(remote_walk, remote_base)

        assert normalized_local == normalized_remote, (
            f"Deep nested structure mismatch:\nLocal: {normalized_local}\nRemote: {normalized_remote}"
        )


@pytest.mark.asyncio
async def test_special_characters_in_names(computer: ComputerInterface) -> None:
    """
    Test that directories and files with spaces and special characters are correctly handled.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_base = Path(tmpdirname) / "special_chars_test"
        local_base.mkdir()
        # Create directories with spaces and special characters
        (local_base / "dir with spaces").mkdir()
        (local_base / "dir-with-dash!").mkdir()
        # Create files with spaces and punctuation
        (local_base / "file with spaces.txt").write_text("space file")
        (local_base / "file#special@.txt").write_text("special chars content")
        # Create a nested file in a subdirectory with spaces in its name
        (local_base / "dir with spaces" / "nested file.txt").write_text("nested content")

        remote_base = "/tmp/test_special_chars"
        await copy_dir_to_computer(computer, local_base, remote_base)

        local_walk = []
        for root, dirs, files, _ in walk_dir_with_mtimes_locally(local_base):
            local_walk.append((root, dirs, files))
        normalized_local = normalize_relative_walk(local_walk, str(local_base))

        remote_walk = []
        async for root, dirs, files, _ in walk_dir_with_mtimes(Path(remote_base), computer):
            remote_walk.append((root, dirs, files))
        normalized_remote = normalize_relative_walk(remote_walk, remote_base)

        assert normalized_local == normalized_remote, (
            f"Special characters test mismatch:\nLocal: {normalized_local}\nRemote: {normalized_remote}"
        )


@pytest.mark.asyncio
async def test_unicode_filenames(computer: ComputerInterface) -> None:
    """
    Test that files and directories with Unicode names are correctly handled.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_base = Path(tmpdirname) / "unicode_test"
        local_base.mkdir()
        # Create directories and files with Unicode characters
        (local_base / "目录").mkdir()  # Chinese for "directory"
        (local_base / "файл.txt").write_text("file in Cyrillic")
        (local_base / "目录" / "文件.txt").write_text("file in Chinese inside directory")

        remote_base = "/tmp/test_unicode"
        await copy_dir_to_computer(computer, local_base, remote_base)

        local_walk = []
        for root, dirs, files, _ in walk_dir_with_mtimes_locally(local_base):
            local_walk.append((root, dirs, files))
        normalized_local = normalize_relative_walk(local_walk, str(local_base))

        remote_walk = []
        async for root, dirs, files, _ in walk_dir_with_mtimes(Path(remote_base), computer):
            remote_walk.append((root, dirs, files))
        normalized_remote = normalize_relative_walk(remote_walk, remote_base)

        assert normalized_local == normalized_remote, (
            f"Unicode filenames mismatch:\nLocal: {normalized_local}\nRemote: {normalized_remote}"
        )


@pytest.mark.asyncio
async def test_broken_symlink(computer: ComputerInterface) -> None:
    """
    Test that a broken symbolic link (one whose target is missing) is correctly reported.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_base = Path(tmpdirname) / "broken_symlink_test"
        local_base.mkdir()
        # Create a file and a symlink pointing to it.
        target_file = local_base / "target.txt"
        target_file.write_text("target content")
        symlink_path = local_base / "broken_link.txt"
        os.symlink(target_file, symlink_path)
        # Remove the target file to break the symlink.
        target_file.unlink()

        remote_base = "/tmp/test_broken_symlink"
        await copy_dir_to_computer(computer, local_base, remote_base)

        local_walk = []
        for root, dirs, files, _ in walk_dir_with_mtimes_locally(local_base):
            local_walk.append((root, dirs, files))
        normalized_local = normalize_relative_walk(local_walk, str(local_base))

        remote_walk = []
        async for root, dirs, files, _ in walk_dir_with_mtimes(Path(remote_base), computer):
            remote_walk.append((root, dirs, files))
        normalized_remote = normalize_relative_walk(remote_walk, remote_base)

        assert normalized_local == normalized_remote, (
            f"Broken symlink test mismatch:\nLocal: {normalized_local}\nRemote: {normalized_remote}"
        )


@pytest.mark.asyncio
async def test_mixed_content(computer: ComputerInterface) -> None:
    """
    Test a directory that contains a mixture of normal files, hidden files,
    directories (with special characters and Unicode), as well as valid and broken symlinks.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_base = Path(tmpdirname) / "mixed_content_test"
        local_base.mkdir()
        # Normal directory and files
        (local_base / "normal_dir").mkdir()
        (local_base / "normal_dir" / "file.txt").write_text("normal file")
        # Hidden file and directory
        (local_base / ".hidden_file.txt").write_text("hidden file")
        (local_base / ".hidden_dir").mkdir()
        (local_base / ".hidden_dir" / "hidden.txt").write_text("hidden dir file")
        # Special characters
        (local_base / "dir with spaces").mkdir()
        (local_base / "dir with spaces" / "spaced file.txt").write_text("spaced file")
        # Unicode directory and file
        (local_base / "目录").mkdir()
        (local_base / "目录" / "文件.txt").write_text("unicode file")
        # Valid symlink in normal_dir
        target_normal = local_base / "normal_dir" / "file.txt"
        symlink_normal = local_base / "normal_dir" / "link_to_file.txt"
        os.symlink(target_normal, symlink_normal)
        # Broken symlink in normal_dir
        broken_target = local_base / "normal_dir" / "non_existent.txt"
        broken_symlink = local_base / "normal_dir" / "broken_link.txt"
        os.symlink(broken_target, broken_symlink)

        remote_base = "/tmp/test_mixed_content"
        await copy_dir_to_computer(computer, local_base, remote_base)

        local_walk = []
        for root, dirs, files, _ in walk_dir_with_mtimes_locally(local_base):
            local_walk.append((root, dirs, files))
        normalized_local = normalize_relative_walk(local_walk, str(local_base))

        remote_walk = []
        async for root, dirs, files, _ in walk_dir_with_mtimes(Path(remote_base), computer):
            remote_walk.append((root, dirs, files))
        normalized_remote = normalize_relative_walk(remote_walk, remote_base)

        assert normalized_local == normalized_remote, (
            f"Mixed content test mismatch:\nLocal: {normalized_local}\nRemote: {normalized_remote}"
        )


@pytest.mark.asyncio
async def test_warn_if_exceeds_threshold(
    computer: ComputerInterface,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Test that if the remote directory has more entries than `warn_threshold`,
    we see a warning message in stdout.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_base = Path(tmpdirname)
        # Create 6 small files
        for i in range(6):
            (local_base / f"file_{i}.txt").write_text("content")

        remote_base = "/tmp/test_warn_if_exceeds_threshold"
        await copy_dir_to_computer(computer, local_base, remote_base)

        # Use a small warn_threshold=5 so that 6 files triggers the warning
        remote_walk = []
        async for root, dirs, files, _ in walk_dir_with_mtimes_on_computer(
            computer=computer, dir_path=Path(remote_base), warn_threshold=5
        ):
            remote_walk.append((root, dirs, files))

        captured = capsys.readouterr()

        is_warning_present = (
            any("WARNING: Directory" in record.getMessage() for record in caplog.records)
            or "WARNING: Directory" in captured.out
        )

        assert is_warning_present, "Expected a warning when exceeding threshold!"


@pytest.mark.asyncio
async def test_symlinked_directory(computer: ComputerInterface) -> None:
    """
    Test that a symlink pointing to a directory is reported just like os.walk:
    it appears in the 'dirs' list and (by default) is not recursed into.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_base = Path(tmpdirname) / "symlinked_dir_test"
        local_base.mkdir()
        # Create a real directory with one file inside
        target_dir = local_base / "real_dir"
        target_dir.mkdir()
        (target_dir / "inner.txt").write_text("inside real dir")

        # Create a symlink pointing to that directory
        symlink_dir = local_base / "link_to_real"
        relative_target = os.path.relpath(target_dir, symlink_dir.parent)
        os.symlink(relative_target, symlink_dir, target_is_directory=True)

        remote_base = "/tmp/test_symlinked_directory"
        await copy_dir_to_computer(computer, local_base, remote_base)

        # Collect local walk
        local_walk = []
        for root, dirs, files, _ in walk_dir_with_mtimes_locally(local_base):
            local_walk.append((root, dirs, files))
        normalized_local = normalize_relative_walk(local_walk, str(local_base))

        # Collect remote walk
        remote_walk = []
        async for root, dirs, files, _ in walk_dir_with_mtimes_on_computer(
            computer, Path(remote_base)
        ):
            remote_walk.append((root, dirs, files))
        normalized_remote = normalize_relative_walk(remote_walk, remote_base)

        # Check that both see "link_to_real" in dirs, not in files,
        # and that they only recurse into real_dir, not into the symlink.
        assert normalized_local == normalized_remote, (
            f"Symlinked-directory mismatch:\nLocal: {normalized_local}\nRemote: {normalized_remote}"
        )
