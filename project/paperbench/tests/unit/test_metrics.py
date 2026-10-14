import json
from pathlib import Path
from typing import Any, cast

from paperbench.metrics import (
    PaperEvaluation,
    RunGroupRecords,
    _attempt_id_from_recorder_group_id,
    _build_evaluation_runs,
    compute_agg_stats,
    parse_run_data,
)


def _paper(run_id: str, paper_id: str) -> PaperEvaluation:
    return PaperEvaluation(
        paper_run_id=run_id,
        paper_id=paper_id,
        graded_task_node=cast(Any, object()),
    )


def _graded_task_tree(score: float) -> dict[str, Any]:
    return {
        "id": "root",
        "requirements": "test",
        "weight": 1,
        "sub_tasks": [],
        "task_category": "Code Development",
        "score": score,
        "valid_score": True,
        "explanation": "test",
        "judge_metadata": None,
    }


def _write_result(
    path: Path,
    *,
    group_id: str,
    timestamp: str,
    paper_id: str,
    run_id: str,
    score: float,
) -> None:
    entry = {
        "record_type": "extra",
        "group_id": group_id,
        "timestamp": timestamp,
        "data": {
            "run_group_id": "2026_run-group_agent",
            "run_id": run_id,
            "pb_result": {
                "paperbench_result": {
                    "paper_id": paper_id,
                    "judge_output": {"graded_task_tree": _graded_task_tree(score)},
                }
            },
        },
    }
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


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


def test_parse_run_data_preserves_multiple_attempts_in_one_run_group(tmp_path: Path) -> None:
    records = tmp_path / "results.jsonl"
    _write_result(
        records,
        group_id="0.0",
        timestamp="2026-08-17T00:00:00+00:00",
        paper_id="paper-a",
        run_id="attempt0-paper-a",
        score=0.2,
    )
    _write_result(
        records,
        group_id="0.0",
        timestamp="2026-08-17T00:00:01+00:00",
        paper_id="paper-b",
        run_id="attempt0-paper-b",
        score=0.4,
    )
    _write_result(
        records,
        group_id="1.0",
        timestamp="2026-08-17T00:01:00+00:00",
        paper_id="paper-a",
        run_id="attempt1-paper-a",
        score=0.8,
    )
    _write_result(
        records,
        group_id="1.0",
        timestamp="2026-08-17T00:01:01+00:00",
        paper_id="paper-b",
        run_id="attempt1-paper-b",
        score=1.0,
    )

    runs = parse_run_data(tmp_path)["agent"]

    assert [run.seed for run in runs] == [
        ("2026_run-group_agent", 1),
        ("2026_run-group_agent", 0),
    ]
    assert all(set(run.paper_evaluations) == {"paper-a", "paper-b"} for run in runs)
    assert compute_agg_stats(runs, expected_papers=2).n_runs == 2
