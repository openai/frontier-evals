from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

import nanoeval._executor_worker as executor_worker


def test_global_concurrency_check_runs_inside_exclusive_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = MagicMock()
    count_result = MagicMock()
    count_result.fetchone.return_value = (1,)
    max_result = MagicMock()
    max_result.fetchone.return_value = ("1",)
    conn.execute.side_effect = [MagicMock(), count_result, max_result, MagicMock()]

    @contextmanager
    def fake_conn():
        yield conn

    monkeypatch.setattr(executor_worker.db, "conn", fake_conn)

    assert executor_worker._maybe_pull_task_from_queue() is None

    statements = [call.args[0].strip() for call in conn.execute.call_args_list]
    assert statements[0] == "BEGIN EXCLUSIVE;"
    assert statements[-1] == "ROLLBACK;"
