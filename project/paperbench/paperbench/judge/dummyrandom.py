import random

from paperbench.judge.base import Judge
from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.rubric.tasks import TaskNode
from typing_extensions import override


class DummyJudge(Judge):
    @property
    def judge_type(self) -> str:
        return "dummy"

    @override
    async def grade_leaf(self, task: TaskNode) -> GradedTaskNode:
        return GradedTaskNode.from_task(
            task,
            score=1.0,
            valid_score=True,
            explanation="This is a dummy judge that always gives a score of 1.0.",
            judge_metadata=None,
        )

    @override
    async def grade_subtree(self, task: TaskNode) -> GradedTaskNode:
        # For demonstration, we'll just assign a perfect score:
        return GradedTaskNode.from_task(
            task,
            score=1.0,
            valid_score=True,
            explanation="Dummy approximate subtree grading with a perfect score.",
            judge_metadata=None,
        )


class RandomJudge(Judge):
    @property
    def judge_type(self) -> str:
        return "random"

    @override
    async def grade_leaf(self, task: TaskNode) -> GradedTaskNode:
        return GradedTaskNode.from_task(
            task,
            score=random.randint(0, 1),
            valid_score=True,
            explanation="This is a random judge that gives a random score of 0 or 1.",
            judge_metadata=None,
        )

    @override
    async def grade_subtree(self, task: TaskNode) -> GradedTaskNode:
        return GradedTaskNode.from_task(
            task,
            score=random.randint(0, 1),
            valid_score=True,
            explanation="This is a random judge that gives a random score of 0 or 1.",
            judge_metadata=None,
        )
