from types import SimpleNamespace
from unittest.mock import MagicMock

import nanoeval.solvers.computer_tasks.solver as solver_module
from nanoeval.solvers.computer_tasks.solver import _log_results
from nanoeval.solvers.computer_tasks.steps import FinalResult
from nanoeval.solvers.computer_tasks.task import Grade


def test_log_results_uses_final_result_correctness(monkeypatch) -> None:
    recorder = MagicMock()
    monkeypatch.setattr(solver_module, "get_recorder", lambda: recorder)
    task = SimpleNamespace(question_id="task.0", attempt_id=3)
    result = FinalResult(grade=Grade(score=0.5, grader_log="partial credit"))

    assert result.correct is False

    _log_results(task, result)

    recorder.record_sampling.assert_called_once_with(
        prompt="",
        sampled="partial credit",
        sample_id="task.0",
        group_id="3",
    )
    recorder.record_match.assert_called_once_with(
        correct=False,
        group_id="3",
    )
