import pandas as pd

import nanoeval.metrics.agents as agents


def test_binary_agent_outcomes_share_answer_groups(monkeypatch) -> None:
    captured: dict[str, pd.DataFrame] = {}

    def capture_metrics(
        samples_df: pd.DataFrame, answer_group_correctness_df: pd.DataFrame
    ) -> dict[str, float]:
        captured["samples"] = samples_df.copy()
        captured["correctness"] = answer_group_correctness_df.copy()
        return {"accuracy": 2 / 3}

    monkeypatch.setattr(agents, "compute_default_metrics", capture_metrics)

    samples = pd.DataFrame(
        {
            "instance": ["q", "q", "q"],
            "attempt": [0, 1, 2],
            "correct": [True, True, False],
            "system_error": [False, False, False],
            "error": [None, None, None],
        }
    )

    summary = agents._compute_metrics_plus_outcome_aggregations(samples)

    assert captured["samples"]["answer_group_id"].tolist() == [1, 1, 0]
    assert captured["correctness"].to_dict("records") == [
        {"instance": "q", "answer_group_id": 1, "is_correct": True},
        {"instance": "q", "answer_group_id": 0, "is_correct": False},
    ]
    assert summary["aggregations"]["num_correct"] == 2
    assert summary["aggregations"]["num_incorrect"] == 1
