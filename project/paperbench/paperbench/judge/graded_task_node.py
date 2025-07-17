from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from paperbench.rubric.tasks import TaskNode
from typing_extensions import override


@dataclass(frozen=True)
class GradedTaskNode(TaskNode):
    """
    Same as a `TaskNode`, but each node also has a `score` and an `explanation`.

    Attributes:
        score: Score between 0 and 1 (and exclusively 0 or 1 for leaf nodes)
        valid_score: Boolean indicating whether the grading is valid, i.e. in case of judge errors
        explanation: Explanation of the grading
        judge_metadata: Additional judge-specific metadata for this node
        sub_tasks: List of sub GradedTaskNodes
    """

    score: float = 0.0
    valid_score: bool = False
    explanation: str = "not yet graded"
    judge_metadata: dict | None = None
    sub_tasks: Sequence["GradedTaskNode"] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "GradedTaskNode":
        try:
            sub_tasks = [cls.from_dict(task) for task in data["sub_tasks"]]
            task = cls(
                id=data["id"],
                requirements=data["requirements"],
                weight=data["weight"],
                sub_tasks=sub_tasks,
                task_category=data["task_category"],
                score=data["score"],
                valid_score=data["valid_score"],
                explanation=data["explanation"],
                judge_metadata=(data["judge_metadata"] if "judge_metadata" in data else None),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in task data: {e}")
        return task

    @override
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "requirements": self.requirements,
            "weight": self.weight,
            "score": self.score,
            "valid_score": self.valid_score,
            "task_category": self.task_category,
            "explanation": self.explanation,
            "judge_metadata": self.judge_metadata,
            "sub_tasks": [task.to_dict() for task in self.sub_tasks],
        }

    def set_score(self, score: float) -> "GradedTaskNode":
        return replace(self, score=score)

    @classmethod
    def from_task(
        cls,
        task: TaskNode,
        score: float,
        valid_score: bool,
        explanation: str,
        judge_metadata: dict | None = None,
    ) -> "GradedTaskNode":
        graded_sub_tasks = [
            cls.from_task(
                sub_task,
                score,
                valid_score,
                explanation=explanation,
                judge_metadata=judge_metadata,
            )
            for sub_task in task.sub_tasks
        ]
        return cls(
            id=task.id,
            requirements=task.requirements,
            weight=task.weight,
            sub_tasks=graded_sub_tasks,
            task_category=task.task_category,
            score=score,
            valid_score=valid_score,
            explanation=explanation,
            judge_metadata=judge_metadata,
        )

    def to_task(self) -> TaskNode:
        sub_tasks = [t.to_task() for t in self.sub_tasks]
        return TaskNode(
            id=self.id,
            requirements=self.requirements,
            weight=self.weight,
            sub_tasks=sub_tasks,
            task_category=self.task_category,
        )


def disqualify_leafs(node: GradedTaskNode) -> GradedTaskNode:
    """
    Sets all leaf scores to 0 and explanations to 'Disqualified'.
    Should be separately followed by `update_all_grades` to propagate the changes.
    """
    if node.is_leaf():
        disqualified_node = node.set_score(0.0)
        disqualified_node = disqualified_node.set_explanation("Submission has been disqualified")
        return disqualified_node

    new_sub_tasks = [disqualify_leafs(child) for child in node.sub_tasks]
    return node.set_sub_tasks(new_sub_tasks)


def disqualify(node: GradedTaskNode) -> GradedTaskNode:
    """
    Sets all leaf scores to 0 and explanations to 'Disqualified'. Updates all scores.
    """
    disqualified_node = disqualify_leafs(node)
    disqualified_node = update_all_grades(disqualified_node)
    return disqualified_node


def update_all_grades(node: GradedTaskNode) -> GradedTaskNode:
    """Recursively updates the scores for all nodes in a graded task tree.
    Leaf nodes retain their existing score, while internal nodes get a score
    computed from their children using score_from_children."""
    if node.is_leaf():
        return node
    new_sub_tasks = [update_all_grades(child) for child in node.sub_tasks]
    computed_score = score_from_children(new_sub_tasks)
    updated_node = node.set_sub_tasks(new_sub_tasks).set_score(computed_score)
    return updated_node


def score_from_children(children: list[GradedTaskNode]) -> float:
    """
    Calculate the weighted score accumulated from a list of graded children.
    """
    total_weight = sum(child.weight for child in children)
    if total_weight == 0:
        return 0.0
    weighted_score = sum(child.score * child.weight for child in children) / total_weight
    return weighted_score
