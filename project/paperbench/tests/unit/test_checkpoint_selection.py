from datetime import timedelta

import pytest

from paperbench.nano.task import PBTask


@pytest.mark.asyncio
async def test_zero_target_duration_selects_earliest_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoints = [
        "/runs/task/submissions/2026-01-01T02-00-00-GMT/submission.tar.gz",
        "/runs/task/submissions/2026-01-01T00-00-00-GMT/submission.tar.gz",
        "/runs/task/submissions/2026-01-01T01-00-00-GMT/submission.tar.gz",
    ]
    task = PBTask.model_construct(run_dir="/runs/task", target_duration_hr=0)
    monkeypatch.setattr("paperbench.nano.task.bf.glob", lambda _pattern: checkpoints)

    checkpoint = await task._select_checkpoint()

    assert checkpoint == (checkpoints[1], timedelta(0))
