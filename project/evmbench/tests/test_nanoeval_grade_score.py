from nanoeval.solvers.computer_tasks.steps import FinalResult

from evmbench.nano.grade.base import EVMbenchGrade, EVMbenchResult


def _grade(score: float, max_score: float) -> EVMbenchGrade:
    return EVMbenchGrade(
        score=score,
        grader_log="",
        evmbench_result=EVMbenchResult(
            audit_id="sample-audit",
            score=score,
            max_score=max_score,
        ),
    )


def test_evmbench_grade_normalizes_score_for_nanoeval_correctness() -> None:
    perfect = _grade(4, 4)
    partial = _grade(2, 4)

    assert perfect.score == 1.0
    assert perfect.evmbench_result.score == 4
    assert FinalResult(grade=perfect).correct is True

    assert partial.score == 0.5
    assert partial.evmbench_result.score == 2
    assert FinalResult(grade=partial).correct is False


def test_evmbench_grade_handles_zero_max_score() -> None:
    grade = _grade(0, 0)

    assert grade.score == 0.0
    assert grade.is_continuous is True
