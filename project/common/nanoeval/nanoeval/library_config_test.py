from contextlib import contextmanager
from typing import Any, Iterator

from nanoeval.library_config import LibraryConfig, get_library_config, set_library_config
from nanoeval.recorder import set_default_recorder


class ContextRecorder:
    def __init__(self) -> None:
        self.context: tuple[str, str | None] | None = None

    @contextmanager
    def as_default_recorder(
        self, sample_id: str, group_id: str | None = None
    ) -> Iterator[None]:
        self.context = (sample_id, group_id)
        try:
            yield
        finally:
            self.context = None


def test_default_library_config_applies_recorder_sample_context() -> None:
    original_config = get_library_config()
    recorder: Any = ContextRecorder()
    set_library_config(LibraryConfig())

    try:
        with set_default_recorder(recorder, sample_id="sample", group_id="group"):
            assert recorder.context == ("sample", "group")
    finally:
        set_library_config(original_config)

    assert recorder.context is None
