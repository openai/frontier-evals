from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import structlog.stdlib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from paperbench.judge.graded_task_node import GradedTaskNode

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass(slots=True)
class MetricSummary:
    """Binary‑classification metrics (macro‑averaged)."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    num_positives: int
    num_negatives: int
    num_samples: int

    def log(self, prefix: str = "") -> None:
        """Structured‑log every field with an optional *prefix*."""
        for field_ in dataclasses.fields(self):
            value = getattr(self, field_.name)
            logger.info("%s%s: %.4f", prefix, field_.name, value)


@dataclass(slots=True)
class LeafScores:
    """Predicted and true scores gathered from leaf nodes."""

    y_pred: list[float] = field(default_factory=list)
    y_true: list[float] = field(default_factory=list)

    def extend(self, other: LeafScores) -> None:
        """Extend this instance with another LeafScores in-place."""
        self.y_pred.extend(other.y_pred)
        self.y_true.extend(other.y_true)


@dataclass(slots=True)
class JudgeMetrics:
    """Overall and per‑category metric summaries."""

    overall: MetricSummary
    stratified: dict[str | None, MetricSummary]


@dataclass(slots=True)
class JudgeResult:
    """Return value of calculate_judge_scores."""

    metrics: JudgeMetrics
    scores: dict[str | None, LeafScores]


def _get_leaf_node_scores(task: GradedTaskNode, expected_result: GradedTaskNode) -> LeafScores:
    """Extract predicted and true scores from leaf nodes of the task tree."""

    if not task.valid_score:
        return LeafScores()

    if task.is_leaf():
        expected_result_node = expected_result.find(task.id)
        return LeafScores([task.score], [expected_result_node.score])

    leaf_scores = LeafScores()
    for sub_task in task.sub_tasks:
        s = _get_leaf_node_scores(sub_task, expected_result)
        leaf_scores.extend(s)
    return leaf_scores


def _get_leaf_node_scores_stratified(
    task: GradedTaskNode, expected_result: GradedTaskNode
) -> dict[str | None, LeafScores]:
    """Extract predicted and true scores from leaf nodes, broken down by task.task_category."""
    category_scores: dict[str | None, LeafScores] = {}
    if not task.valid_score:
        return category_scores
    if task.is_leaf():
        expected_result_node = expected_result.find(task.id)
        cat = task.task_category
        if cat not in category_scores:
            category_scores[cat] = LeafScores()
        category_scores[cat].y_pred.append(task.score)
        category_scores[cat].y_true.append(expected_result_node.score)
        return category_scores
    for sub_task in task.sub_tasks:
        stratified_child_scores = _get_leaf_node_scores_stratified(sub_task, expected_result)
        for cat, child_scores in stratified_child_scores.items():
            if cat not in category_scores:
                category_scores[cat] = LeafScores()
            category_scores[cat].extend(child_scores)
    return category_scores


def calculate_judge_scores(
    graded_task_tree: GradedTaskNode, expected_result: GradedTaskNode
) -> JudgeResult:
    """Calculate evaluation metrics for a graded task tree against expected results."""
    leaf_scores = _get_leaf_node_scores(graded_task_tree, expected_result)

    overall_metrics = compute_metrics(leaf_scores.y_true, leaf_scores.y_pred)
    # Compute metrics broken down by task.task_category
    stratified_scores = _get_leaf_node_scores_stratified(graded_task_tree, expected_result)
    stratified: dict[str | None, MetricSummary] = {}

    for category, cat_scores in stratified_scores.items():
        metrics = compute_metrics(cat_scores.y_true, cat_scores.y_pred)
        stratified[category] = metrics
        metrics.log(prefix=f"{category} ")

    results = JudgeMetrics(overall=overall_metrics, stratified=stratified)
    overall_metrics.log(prefix="Overall ")

    scores = {
        "Overall": leaf_scores,
        **stratified_scores,
    }
    return JudgeResult(metrics=results, scores=scores)


def compute_metrics(y_true: list[float], y_pred: list[float]) -> MetricSummary:
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    num_positives = int(sum(y_true))
    num_samples = len(y_true)
    num_negatives = num_samples - num_positives

    return MetricSummary(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        num_positives=num_positives,
        num_negatives=num_negatives,
        num_samples=num_samples,
    )
