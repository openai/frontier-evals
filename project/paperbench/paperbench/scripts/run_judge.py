import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import structlog.stdlib
from paperbench.judge.create_judge import create_judge, handle_judge_kwargs
from paperbench.judge.judge import GradedTaskNode, TokenUsage, get_total_token_usage
from paperbench.paper_registry import paper_registry
from paperbench.rubric.tasks import TaskNode
from paperbench.utils import get_timestamp

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass(frozen=True)
class JudgeOutput:
    judge_type: str
    model_name: str
    score: float
    num_leaf_nodes: int
    num_invalid_leaf_nodes: int
    graded_at: str
    graded_task_tree: GradedTaskNode
    token_usage: TokenUsage | None = None

    def to_dict(self) -> dict:
        return {
            "judge_type": self.judge_type,
            "model_name": self.model_name,
            "score": self.score,
            "num_leaf_nodes": self.num_leaf_nodes,
            "num_invalid_leaf_nodes": self.num_invalid_leaf_nodes,
            "graded_at": self.graded_at,
            "graded_task_tree": self.graded_task_tree.to_dict(),
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JudgeOutput":
        token_usage = None

        try:
            if data.get("token_usage"):
                token_usage = TokenUsage.from_dict(data["token_usage"])

            return cls(
                judge_type=data["judge_type"],
                model_name=data["model_name"],
                score=data["score"],
                num_leaf_nodes=data["num_leaf_nodes"],
                num_invalid_leaf_nodes=data["num_invalid_leaf_nodes"],
                graded_at=data["graded_at"],
                graded_task_tree=GradedTaskNode.from_dict(data["graded_task_tree"]),
                token_usage=token_usage,
            )
        except KeyError as e:
            raise ValueError(f"Missing key {e} in judge output data!") from e

    @property
    def success(self) -> bool:
        return self.num_invalid_leaf_nodes < self.num_leaf_nodes


async def judge(
    submission_path: Path,
    paper_id: str,
    judge_type: str,
    model_name: str,
    out_dir: Path,
    max_depth: int,
    code_only: bool,
    reasoning_effort: str | None = None,
    resources_provided: bool = False,
) -> GradedTaskNode:
    """
    Judge a single submission (a directory of files) using the specified judge.
    Returns the graded task tree.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    paper = paper_registry.get_paper(paper_id)
    rubric_path = paper.rubric
    paper_pdf_path = paper.paper_pdf
    paper_md_path = paper.paper_md

    logger.info(f"Judging submission for paper {rubric_path}")
    with open(rubric_path, "r") as f:
        task_tree = TaskNode.from_dict(json.load(f))
    if code_only:
        task_tree = task_tree.code_only() or task_tree.set_task_category(
            "Code Development"
        ).set_sub_tasks([])

    if resources_provided:
        task_tree = task_tree.resources_provided()

    judge_kwargs = handle_judge_kwargs(
        judge_type, code_only, resources_provided, paper, model_name, reasoning_effort
    )
    judge = create_judge(
        judge_type=judge_type,
        judge_kwargs=judge_kwargs,
        paper_path=paper_pdf_path,
        rubric=task_tree,
        addendum=paper.addendum.read_text() if paper.addendum else None,
        judge_addendum=paper.judge_addendum.read_text() if paper.judge_addendum.exists() else None,
        submission_dir=submission_path,
        paper_md=paper_md_path,
        log_path=out_dir,
        max_depth=max_depth,
    )

    return await judge.grade()


async def main(
    submission_path: Path,
    paper_id: str,
    judge_type: str,
    model_name: str,
    max_depth: int,
    out_dir: Path,
    code_only: bool,
    resources_provided: bool = False,
    reasoning_effort: str | None = None,
):
    # Judge the submission
    graded_task_tree = await judge(
        submission_path=submission_path,
        paper_id=paper_id,
        judge_type=judge_type,
        model_name=model_name,
        out_dir=out_dir,
        max_depth=max_depth,
        code_only=code_only,
        resources_provided=resources_provided,
        reasoning_effort=reasoning_effort,
    )

    token_usage = None

    if judge_type == "simple":
        token_usage = get_total_token_usage(graded_task_tree)

    # Save judging outputs
    path_to_judge_output = out_dir / "grader_output.json"
    judge_output = JudgeOutput(
        judge_type=judge_type,
        model_name=model_name,
        score=graded_task_tree.score,
        num_leaf_nodes=len(graded_task_tree.get_leaf_nodes()),
        num_invalid_leaf_nodes=len(
            [node for node in graded_task_tree.get_leaf_nodes() if not node.valid_score]
        ),
        graded_at=get_timestamp(),
        graded_task_tree=graded_task_tree,
        token_usage=token_usage,
    )

    with open(path_to_judge_output, "w") as f:
        json.dump(judge_output.to_dict(), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade a paper submission.")
    parser.add_argument(
        "--submission-path",
        type=Path,
        help="Path to the submission directory.",
        required=True,
    )
    parser.add_argument(
        "--paper-id",
        help="Identifier for the paper.",
        required=True,
    )
    parser.add_argument(
        "-j",
        "--judge",
        choices=["dummy", "random", "simple"],
        default="dummy",
        help="Specify the judge to use (default: dummy).",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Specify the OpenAI model to use (required if judge isn't dummy).",
    )
    parser.add_argument(
        "-d",
        "--max-depth",
        type=int,
        default=999,
        help="Specify the maximum depth to grade.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Path to store the judge's outputs.",
        required=True,
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Only grade 'Code Development' nodes",
    )
    parser.add_argument(
        "--resources-provided",
        action="store_true",
        help="Weight 'Dataset and Model Acquisition' nodes to 0",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        required=False,
        default="high",
        help="Reasoning effort to use for the completion, if using a model/judge that supports it.",
    )

    args = parser.parse_args()

    if args.judge == "simple" and not args.model:
        parser.error("--model is required when using simple judge.")

    asyncio.run(
        main(
            submission_path=args.submission_path,
            paper_id=args.paper_id,
            judge_type=args.judge,
            model_name=args.model,
            max_depth=args.max_depth,
            out_dir=args.out_dir,
            code_only=args.code_only,
            resources_provided=args.resources_provided,
            reasoning_effort=args.reasoning_effort,
        )
    )
