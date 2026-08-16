from pathlib import Path

import pytest

from paperbench.judge.base import Judge
from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.rubric.tasks import TaskNode


class ValidityJudge(Judge):
    @property
    def judge_type(self) -> str:
        return "validity-test"

    async def grade_leaf(self, task: TaskNode) -> GradedTaskNode:
        raise NotImplementedError

    async def grade_subtree(self, task: TaskNode) -> GradedTaskNode:
        raise NotImplementedError


def _leaf(node_id: str) -> TaskNode:
    return TaskNode(
        id=node_id,
        requirements=node_id,
        weight=1,
        task_category="Code Development",
    )


@pytest.mark.asyncio
async def test_aggregate_validity_propagates_child_grading_failure(tmp_path: Path) -> None:
    good = _leaf("good")
    bad = _leaf("bad")
    root = TaskNode(
        id="root",
        requirements="root",
        weight=1,
        sub_tasks=[good, bad],
    )
    judge = ValidityJudge(
        paper_path=tmp_path / "paper.pdf",
        rubric=root,
        addendum=None,
        judge_addendum=None,
        submission_dir=tmp_path,
    )

    async def grade_leaf(task: TaskNode) -> GradedTaskNode:
        if task.id == "bad":
            raise RuntimeError("judge failed")
        return GradedTaskNode.from_task(
            task,
            score=1.0,
            valid_score=True,
            explanation="graded",
        )

    graded = await judge.grade(root, grade_leaf)

    assert graded.score == 0.5
    assert graded.valid_score is False
    assert graded.sub_tasks[0].valid_score is True
    assert graded.sub_tasks[1].valid_score is False


@pytest.mark.asyncio
async def test_aggregate_validity_remains_true_when_all_children_are_valid(tmp_path: Path) -> None:
    children = [_leaf("a"), _leaf("b")]
    root = TaskNode(
        id="root",
        requirements="root",
        weight=1,
        sub_tasks=children,
    )
    judge = ValidityJudge(
        paper_path=tmp_path / "paper.pdf",
        rubric=root,
        addendum=None,
        judge_addendum=None,
        submission_dir=tmp_path,
    )

    async def grade_leaf(task: TaskNode) -> GradedTaskNode:
        return GradedTaskNode.from_task(
            task,
            score=1.0,
            valid_score=True,
            explanation="graded",
        )

    graded = await judge.grade(root, grade_leaf)

    assert graded.score == 1.0
    assert graded.valid_score is True
