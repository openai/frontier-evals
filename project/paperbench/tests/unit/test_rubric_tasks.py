from paperbench.judge.graded_task_node import GradedTaskNode, update_all_grades
from paperbench.rubric.tasks import TaskNode

RESOURCE_CATEGORY = "Dataset and Model Acquisition"


def leaf(node_id: str, finegrained_task_category: str, weight: int = 1) -> TaskNode:
    return TaskNode(
        id=node_id,
        requirements=node_id,
        weight=weight,
        task_category="Code Development",
        finegrained_task_category=finegrained_task_category,
    )


def perfect_score(task: TaskNode) -> float:
    graded_task = GradedTaskNode.from_task(
        task,
        score=1.0,
        valid_score=True,
        explanation="Perfect submission",
    )
    return update_all_grades(graded_task).score


def test_resources_provided_zeroes_fully_excluded_subtree_weight() -> None:
    task = TaskNode(
        id="root",
        requirements="root",
        weight=1,
        sub_tasks=[
            leaf("method", "Method Implementation"),
            TaskNode(
                id="resources",
                requirements="resources",
                weight=1,
                sub_tasks=[
                    leaf("dataset", RESOURCE_CATEGORY),
                    leaf("model", RESOURCE_CATEGORY),
                ],
            ),
        ],
    )

    transformed = task.resources_provided()

    assert transformed.find("resources").weight == 0
    assert transformed.find("dataset").weight == 0
    assert transformed.find("model").weight == 0
    assert perfect_score(transformed) == 1.0


def test_resources_provided_propagates_zero_weight_through_nested_subtrees() -> None:
    task = TaskNode(
        id="root",
        requirements="root",
        weight=1,
        sub_tasks=[
            leaf("method", "Method Implementation"),
            TaskNode(
                id="resources",
                requirements="resources",
                weight=2,
                sub_tasks=[
                    TaskNode(
                        id="datasets",
                        requirements="datasets",
                        weight=1,
                        sub_tasks=[leaf("dataset", RESOURCE_CATEGORY)],
                    ),
                    leaf("model", RESOURCE_CATEGORY),
                ],
            ),
        ],
    )

    transformed = task.resources_provided()

    assert transformed.find("resources").weight == 0
    assert transformed.find("datasets").weight == 0
    assert perfect_score(transformed) == 1.0


def test_resources_provided_preserves_mixed_subtree_weight() -> None:
    task = TaskNode(
        id="root",
        requirements="root",
        weight=1,
        sub_tasks=[
            TaskNode(
                id="mixed",
                requirements="mixed",
                weight=3,
                sub_tasks=[
                    leaf("dataset", RESOURCE_CATEGORY),
                    leaf("method", "Method Implementation"),
                ],
            )
        ],
    )

    transformed = task.resources_provided()

    assert transformed.find("mixed").weight == 3
    assert transformed.find("dataset").weight == 0
    assert transformed.find("method").weight == 1


def test_resources_provided_preserves_unrelated_zero_weight_subtree() -> None:
    task = TaskNode(
        id="root",
        requirements="root",
        weight=1,
        sub_tasks=[
            leaf("method", "Method Implementation"),
            TaskNode(
                id="unrelated",
                requirements="unrelated",
                weight=2,
                sub_tasks=[leaf("zero-weight-method", "Method Implementation", weight=0)],
            ),
        ],
    )

    transformed = task.resources_provided()

    assert transformed.find("unrelated").weight == 2
    assert perfect_score(transformed) == perfect_score(task)
