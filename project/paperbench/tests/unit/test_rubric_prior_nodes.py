from paperbench.rubric.tasks import TaskNode


def _rubric() -> tuple[TaskNode, TaskNode]:
    b = TaskNode(
        id="B",
        requirements="B",
        weight=1,
        task_category="Code Development",
    )
    f = TaskNode(
        id="F",
        requirements="F",
        weight=1,
        task_category="Code Development",
    )
    g = TaskNode(
        id="G",
        requirements="G",
        weight=1,
        task_category="Code Development",
    )
    c = TaskNode(id="C", requirements="C", weight=1, sub_tasks=[f, g])
    root = TaskNode(id="A", requirements="A", weight=1, sub_tasks=[b, c])
    return root, g


def test_get_prior_nodes_respects_zero_limit() -> None:
    root, target = _rubric()

    assert [node.id for node in target.get_prior_nodes(root)] == ["A", "B", "C", "F"]
    assert target.get_prior_nodes(root, max_prior_nodes=0) == []
    assert [node.id for node in target.get_prior_nodes(root, max_prior_nodes=2)] == ["C", "F"]
