from paperbench.grade import _code_only_task_tree
from paperbench.rubric.tasks import TaskNode


def test_code_only_fallback_collapses_non_code_rubric_to_valid_leaf() -> None:
    leaf = TaskNode(
        id="analysis",
        requirements="analyse results",
        weight=1,
        task_category="Result Analysis",
    )
    root = TaskNode(
        id="root",
        requirements="complete the paper",
        weight=3,
        sub_tasks=[leaf],
    )

    code_only = _code_only_task_tree(root)

    assert code_only.is_leaf()
    assert code_only.task_category == "Code Development"
    assert code_only.id == root.id
    assert code_only.requirements == root.requirements
    assert code_only.weight == root.weight


def test_code_only_keeps_existing_code_development_subtree() -> None:
    code_leaf = TaskNode(
        id="code",
        requirements="implement method",
        weight=1,
        task_category="Code Development",
    )
    analysis_leaf = TaskNode(
        id="analysis",
        requirements="analyse results",
        weight=1,
        task_category="Result Analysis",
    )
    root = TaskNode(
        id="root",
        requirements="complete the paper",
        weight=2,
        sub_tasks=[code_leaf, analysis_leaf],
    )

    code_only = _code_only_task_tree(root)

    assert [node.id for node in code_only.get_leaf_nodes()] == ["code"]
