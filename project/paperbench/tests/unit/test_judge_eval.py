from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.judge.judge_eval.evaluate import (
    _get_leaf_node_scores,
    _get_leaf_node_scores_stratified,
    calculate_judge_scores,
)


def test_leaf_scores_retains_invalid_leaves():
    """Verify that leaf nodes with valid_score=False are counted rather than silently dropped."""
    expected_leaf_1 = GradedTaskNode(
        id="1.1",
        requirements="Task 1.1",
        weight=1.0,
        sub_tasks=[],
        task_category="Code Development",
        score=1.0,
        valid_score=True,
    )
    expected_leaf_2 = GradedTaskNode(
        id="1.2",
        requirements="Task 1.2",
        weight=1.0,
        sub_tasks=[],
        task_category="Code Execution",
        score=1.0,
        valid_score=True,
    )
    expected_root = GradedTaskNode(
        id="1",
        requirements="Root",
        weight=1.0,
        sub_tasks=[expected_leaf_1, expected_leaf_2],
        task_category=None,
        score=1.0,
        valid_score=True,
    )

    # Graded tree where leaf 1.2 failed grading (valid_score=False, score=0.0)
    graded_leaf_1 = GradedTaskNode(
        id="1.1",
        requirements="Task 1.1",
        weight=1.0,
        sub_tasks=[],
        task_category="Code Development",
        score=1.0,
        valid_score=True,
    )
    graded_leaf_2 = GradedTaskNode(
        id="1.2",
        requirements="Task 1.2",
        weight=1.0,
        sub_tasks=[],
        task_category="Code Execution",
        score=0.0,
        valid_score=False,
    )
    graded_root = GradedTaskNode(
        id="1",
        requirements="Root",
        weight=1.0,
        sub_tasks=[graded_leaf_1, graded_leaf_2],
        task_category=None,
        score=0.5,
        valid_score=False,
    )

    leaf_scores = _get_leaf_node_scores(graded_root, expected_root)
    assert len(leaf_scores.y_pred) == 2
    assert len(leaf_scores.y_true) == 2
    assert leaf_scores.y_pred == [1.0, 0.0]
    assert leaf_scores.y_true == [1.0, 1.0]

    stratified = _get_leaf_node_scores_stratified(graded_root, expected_root)
    assert "Code Development" in stratified
    assert "Code Execution" in stratified
    assert stratified["Code Development"].y_pred == [1.0]
    assert stratified["Code Execution"].y_pred == [0.0]

    result = calculate_judge_scores(graded_root, expected_root)
    assert result.metrics.overall.num_samples == 2
    # 1 match out of 2 samples -> accuracy 0.5
    assert result.metrics.overall.accuracy == 0.5
