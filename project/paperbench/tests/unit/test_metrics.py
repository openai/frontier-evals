from typing import Any, cast

from paperbench.metrics import (
    PaperEvaluation,
    RunGroupRecords,
    _attempt_id_from_recorder_group_id,
    _build_evaluation_runs,
)


def _paper(run_id: str, paper_id: str) -> PaperEvaluation:
    return PaperEvaluation(
        paper_run_id=run_id,
        paper_id=paper_id,
        graded_task_node=cast(Any, object()),
    )


def test_evaluation_runs_preserve_actual_run_group_boundaries() -> None:
    newest = RunGroupRecords(
        timestamp=200.0,
        paper_evaluations={"paper-a": (_paper("b-a", "paper-a"), 200.0)},
    )
    older = RunGroupRecords(
        timestamp=100.0,
        paper_evaluations={
            "paper-a": (_paper("a-a", "paper-a"), 100.0),
            "paper-b": (_paper("a-b", "paper-b"), 101.0),
        },
    )

    runs = _build_evaluation_runs({"run_B": newest, "run_A": older}, seeds_to_keep=None)

    assert [run.seed for run in runs] == ["run_B", "run_A"]
    assert set(runs[0].paper_evaluations) == {"paper-a"}
    assert set(runs[1].paper_evaluations) == {"paper-a", "paper-b"}


def test_seed_limit_applies_to_whole_run_groups() -> None:
    newest = RunGroupRecords(
        timestamp=200.0,
        paper_evaluations={"paper-a": (_paper("b-a", "paper-a"), 200.0)},
    )
    older = RunGroupRecords(
        timestamp=100.0,
        paper_evaluations={"paper-b": (_paper("a-b", "paper-b"), 100.0)},
    )

    runs = _build_evaluation_runs({"run_B": newest, "run_A": older}, seeds_to_keep=1)

    assert len(runs) == 1
    assert runs[0].seed == "run_B"
    assert set(runs[0].paper_evaluations) == {"paper-a"}


def test_recorder_group_id_preserves_attempt_and_ignores_retry() -> None:
    assert _attempt_id_from_recorder_group_id("0.0") == 0
    assert _attempt_id_from_recorder_group_id("2.3") == 2
    assert _attempt_id_from_recorder_group_id(None) is None
    assert _attempt_id_from_recorder_group_id("legacy") is None


def test_attempts_within_one_run_group_remain_distinct() -> None:
    attempt_zero = RunGroupRecords(
        timestamp=100.0,
        paper_evaluations={
            "paper-a": (_paper("a0-a", "paper-a"), 100.0),
            "paper-b": (_paper("a0-b", "paper-b"), 101.0),
        },
    )
    attempt_one = RunGroupRecords(
        timestamp=200.0,
        paper_evaluations={
            "paper-a": (_paper("a1-a", "paper-a"), 200.0),
            "paper-b": (_paper("a1-b", "paper-b"), 201.0),
        },
    )

    runs = _build_evaluation_runs(
        {("run_A", 0): attempt_zero, ("run_A", 1): attempt_one},
        seeds_to_keep=None,
    )

    assert [run.seed for run in runs] == [("run_A", 1), ("run_A", 0)]
    assert all(set(run.paper_evaluations) == {"paper-a", "paper-b"} for run in runs)
