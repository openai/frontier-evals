import json
from types import SimpleNamespace

import pytest

import paperbench.metrics as metrics


def _entry(paper_id: str, run_id: str, timestamp: str) -> dict:
    return {
        "record_type": "extra",
        "timestamp": timestamp,
        "data": {
            "run_group_id": "group_agent",
            "run_id": run_id,
            "pb_result": {
                "grader_success": True,
                "grader_output": {"graded_task_tree": {"score": 1.0}},
                "paper_id": paper_id,
            },
        },
    }


def test_parse_run_data_handles_papers_with_uneven_seed_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        metrics.GradedTaskNode,
        "from_dict",
        lambda data: SimpleNamespace(score=data["score"]),
    )

    entries = [
        _entry("paper-short", "short-1", "2026-01-01T00:00:00Z"),
        _entry("paper-long", "long-1", "2026-01-03T00:00:00Z"),
        _entry("paper-long", "long-2", "2026-01-02T00:00:00Z"),
        _entry("paper-long", "long-3", "2026-01-01T00:00:00Z"),
    ]
    data_file = tmp_path / "runs.jsonl"
    data_file.write_text("\n".join(json.dumps(entry) for entry in entries))

    parsed = metrics.parse_run_data(tmp_path)

    runs = parsed["agent"]
    assert len(runs) == 3
    assert set(runs[0].paper_evaluations) == {"paper-short", "paper-long"}
    assert set(runs[1].paper_evaluations) == {"paper-long"}
    assert set(runs[2].paper_evaluations) == {"paper-long"}
