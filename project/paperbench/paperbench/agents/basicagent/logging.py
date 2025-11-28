from __future__ import annotations

from collections import UserList
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")


class LoggableMessages(UserList[T]):
    """
    A list that automatically logs itself after every in-place change
    """

    def __init__(
        self,
        init: Iterable[T] | None = None,
        *,
        log_path: str,
        logger: Callable[[list[T], str], None],
    ):
        # collections.UserList expects an iterable, default to empty tuple when ``None``
        super().__init__(() if init is None else init)
        self._log_path = log_path
        self._logger = logger
        self._log()  # initial dump

    def _log(self) -> None:
        self._logger(self.data, self._log_path)

    def append(self, item: T) -> None:
        super().append(item)
        self._log()

    def extend(self, other: Iterable[T]) -> None:
        super().extend(other)
        self._log()

    def insert(self, i: int, item: T) -> None:
        super().insert(i, item)
        self._log()

    def pop(self, i: int = -1) -> T:
        val: T = super().pop(i)
        self._log()
        return val

    def remove(self, item: T) -> None:
        super().remove(item)
        self._log()

    def clear(self) -> None:
        super().clear()
        self._log()

    def __setitem__(self, i: Any, v: Any) -> None:
        super().__setitem__(i, v)
        self._log()

    def __delitem__(self, i: Any) -> None:
        super().__delitem__(i)
        self._log()

    def __iadd__(self, other: Iterable[T]) -> LoggableMessages[T]:
        super().__iadd__(other)
        self._log()
        return self
