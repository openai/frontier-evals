import json

from paperbench.metrics import parse_run_data


def _graded_task_tree() -> dict:
    return {
        "id": "task",
        "requirements": "test requirement",
        "weight": 1,
        "sub_tasks": [],
        "task_category": "Code Development",
        "score": 1.0,
        "valid_score": True,
        "explanation": "ok",
    }


def _entry(paper_id: str, run_id: str, timestamp: str) -> dict:
    return {
        "record_type": "extra",
        "timestamp": timestamp,
        "data": {
            "run_group_id": "group_agent",
            "run_id": run_id,
            "pb_result": {
                "paperbench_result": {
                    "paper_id": paper_id,
                    "judge_output": {"graded_task_tree": _graded_task_tree()},
                }
            },
        },
    }


def test_parse_run_data_handles_papers_with_uneven_seed_counts(tmp_path):
    entries = [
        _entry("paper-a", "a-1", "2026-01-03T00:00:00Z"),
        _entry("paper-a", "a-2", "2026-01-02T00:00:00Z"),
        _entry("paper-a", "a-3", "2026-01-01T00:00:00Z"),
        _entry("paper-b", "b-1", "2026-01-03T00:00:00Z"),
    ]
    run_file = tmp_path / "runs.jsonl"
    run_file.write_text("".join(f"{json.dumps(entry)}\n" for entry in entries))

    runs = parse_run_data(tmp_path)

    assert list(runs) == ["agent"]
    assert len(runs["agent"]) == 3
    assert set(runs["agent"][0].paper_evaluations) == {"paper-a", "paper-b"}
    assert set(runs["agent"][1].paper_evaluations) == {"paper-a"}
    assert set(runs["agent"][2].paper_evaluations) == {"paper-a"}
