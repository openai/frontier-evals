from datetime import timedelta

import pytest

from paperbench.nano.utils import get_file_at_duration


def test_get_file_at_duration_returns_expected_result() -> None:
    # Given
    files = [
        "/logs/run/2024-12-07T10-00-00-GMT/output.tar.gz",
        "/logs/run/2024-12-07T10-59-59-GMT/output.tar.gz",
        "/logs/run/2024-12-07T11-30-00-GMT/output.tar.gz",
    ]

    # When
    result = get_file_at_duration(files, 1)

    # Then
    assert result["path"] == "/logs/run/2024-12-07T10-59-59-GMT/output.tar.gz"
    assert result["duration"] == timedelta(hours=0, minutes=59, seconds=59)


def test_get_file_at_duration_raises_when_timestamp_missing() -> None:
    # Given
    files = ["/logs/run/latest/output.tar.gz"]

    # Then
    with pytest.raises(ValueError):
        get_file_at_duration(files, 1)


def test_get_file_at_duration_raises_with_short_path() -> None:
    # Given
    files = ["output.tar.gz"]

    # Then
    with pytest.raises(ValueError):
        get_file_at_duration(files, 1)
