import json

import pytest

from paperbench.metrics import parse_run_data


@pytest.mark.parametrize(
    ("run_group_id", "expected_shortname"),
    [
        (
            "2026-01-01T00-00-00-UTC_run-group_direct_submission_solver",
            "direct_submission_solver",
        ),
        (
            "2026-01-01T00-00-00-UTC_run-group_direct_submission_solver_run-group_variant",
            "direct_submission_solver_run-group_variant",
        ),
    ],
)
def test_parse_run_data_preserves_solver_shortname_with_underscores(
    tmp_path, run_group_id: str, expected_shortname: str
):
    entry = {
        "record_type": "extra",
        "timestamp": "2026-01-01T00:00:00Z",
        "data": {
            "run_group_id": run_group_id,
            "run_id": "paper-a_run-1",
            "pb_result": {
                "paperbench_result": {
                    "paper_id": "paper-a",
                    "judge_output": {
                        "graded_task_tree": {
                            "id": "task",
                            "requirements": "test requirement",
                            "weight": 1,
                            "sub_tasks": [],
                            "task_category": "Code Development",
                            "score": 1.0,
                            "valid_score": True,
                            "explanation": "ok",
                        }
                    },
                }
            },
        },
    }
    run_file = tmp_path / "runs.jsonl"
    run_file.write_text(f"{json.dumps(entry)}\n")

    runs = parse_run_data(tmp_path)

    assert list(runs) == [expected_shortname]
    assert runs[expected_shortname][0].paper_evaluations["paper-a"].paper_run_id == (
        "paper-a_run-1"
    )
