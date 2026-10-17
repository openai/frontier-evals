from paperbench.rubric.tasks import TaskNode


def test_prune_to_depth_marks_collapsed_internal_node_as_subtree() -> None:
    leaf = TaskNode(
        id="leaf",
        requirements="leaf requirement",
        weight=1,
        task_category="Code Development",
    )
    root = TaskNode(
        id="root",
        requirements="root requirement",
        weight=1,
        sub_tasks=[leaf],
    )

    pruned = root.prune_to_depth(0)

    assert pruned.is_leaf()
    assert pruned.task_category == "Subtree"
    assert pruned.id == root.id
    assert pruned.requirements == root.requirements
    assert pruned.weight == root.weight
