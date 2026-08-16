from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.judge.token_usage import get_total_token_usage
from paperbench.rubric.tasks import TaskNode


def _leaf(node_id: str, token_usage):
    task = TaskNode(
        id=node_id,
        requirements=node_id,
        weight=1,
        task_category="Code Development",
    )
    return GradedTaskNode.from_task(
        task,
        score=1.0,
        valid_score=True,
        explanation="graded",
        judge_metadata={"token_usage": token_usage},
    )


def test_total_token_usage_skips_null_leaf_usage() -> None:
    root = GradedTaskNode(
        id="root",
        requirements="root",
        weight=1,
        sub_tasks=[
            _leaf("without-usage", None),
            _leaf("with-usage", {"gpt-test": {"in": 12, "out": 5}}),
        ],
        score=1.0,
        valid_score=True,
        explanation="aggregate",
    )

    total = get_total_token_usage(root)

    assert total.to_dict() == {"gpt-test": {"in": 12, "out": 5}}


def test_total_token_usage_skips_missing_leaf_usage_key() -> None:
    task = TaskNode(
        id="leaf",
        requirements="leaf",
        weight=1,
        task_category="Code Development",
    )
    leaf = GradedTaskNode.from_task(
        task,
        score=1.0,
        valid_score=True,
        explanation="graded",
        judge_metadata={},
    )

    assert get_total_token_usage(leaf).to_dict() == {}
