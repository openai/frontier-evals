import math

import pytest

from paperbench.utils import safe_mean


def test_safe_mean_with_values() -> None:
    # Given
    values = [1.0, 3.0]
    expected = 2.0

    # When
    actual = safe_mean(values)

    # Then
    assert math.isclose(actual, expected)


def test_safe_mean_empty_default() -> None:
    # Given
    values: list[float] = []
    expected = 5.0

    # When
    actual = safe_mean(values, default=expected)

    # Then
    assert actual == expected


def test_safe_mean_with_nan() -> None:
    # Given
    values = [float("nan"), 1.0]

    # When
    actual = safe_mean(values)

    # Then
    assert math.isnan(actual)


def test_safe_mean_invalid_values() -> None:
    # Given
    values = [1.0, "bad"]

    # When/Then
    with pytest.raises(AssertionError):
        safe_mean(values)  # type: ignore[arg-type]
