from typing import Any, cast

from paperbench.metrics import (
    PaperEvaluation,
    RunGroupRecords,
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
