import structlog.stdlib
from paperbench.judge.graded_task_node import GradedTaskNode
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = structlog.stdlib.get_logger(component=__name__)


def _get_leaf_node_scores(
    task: GradedTaskNode, expected_result: GradedTaskNode
) -> tuple[list[int], list[int]]:
    """Extract predicted and true scores from leaf nodes of the task tree."""
    if not task.valid_score:
        return [], []

    if task.is_leaf():
        expected_result_node = expected_result.find(task.id)
        return [[task.score], [expected_result_node.score]]

    scores = []
    expected_results = []
    for t in task.sub_tasks:
        s, e = _get_leaf_node_scores(t, expected_result)
        scores.extend(s)
        expected_results.extend(e)
    return scores, expected_results


def _get_leaf_node_scores_stratified(
    task: GradedTaskNode, expected_result: GradedTaskNode
) -> dict[str, tuple[list[int], list[int]]]:
    """Extract predicted and true scores from leaf nodes, broken down by task.task_category."""
    category_scores: dict[str, tuple[list[int], list[int]]] = {}
    if not task.valid_score:
        return category_scores
    if task.is_leaf():
        expected_result_node = expected_result.find(task.id)
        cat = task.task_category
        if cat not in category_scores:
            category_scores[cat] = ([], [])
        category_scores[cat][0].append(task.score)
        category_scores[cat][1].append(expected_result_node.score)
        return category_scores
    for sub_task in task.sub_tasks:
        child_scores = _get_leaf_node_scores_stratified(sub_task, expected_result)
        for cat, (child_pred, child_true) in child_scores.items():
            if cat not in category_scores:
                category_scores[cat] = ([], [])
            category_scores[cat][0].extend(child_pred)
            category_scores[cat][1].extend(child_true)
    return category_scores


def calculate_judge_scores(
    graded_task_tree: GradedTaskNode, expected_result: GradedTaskNode
) -> tuple[dict[str, float], dict[str, tuple[list[int], list[int]]]]:
    """Calculate evaluation metrics for a graded task tree against expected results."""
    y_pred, y_true = _get_leaf_node_scores(graded_task_tree, expected_result)

    results = compute_metrics(y_true, y_pred)
    # Compute metrics broken down by task.task_category
    stratified_scores = _get_leaf_node_scores_stratified(graded_task_tree, expected_result)
    stratified = {}
    for category, (cat_pred, cat_true) in stratified_scores.items():
        metrics = compute_metrics(cat_true, cat_pred)
        stratified[category] = metrics
        for key, value in metrics.items():
            logger.info(f"{category} {key}: {round(value, 4)}")
    results["stratified"] = stratified

    for key, value in results.items():
        if key != "stratified":  # already logged category metrics above
            logger.info(f"Overall {key}: {round(value, 4)}")

    scores = {
        "Overall": (y_pred, y_true),
        **stratified_scores,
    }
    return results, scores


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, average="macro"),
        "recall": recall_score(y_true, y_pred, zero_division=0, average="macro"),
        "f1": f1_score(y_true, y_pred, zero_division=0, average="macro"),
        "num_positives": sum(y_true),
        "num_negatives": len(y_true) - sum(y_true),
        "num_samples": len(y_true),
    }
