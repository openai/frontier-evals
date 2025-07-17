import argparse
import asyncio
import json
from pathlib import Path

import structlog.stdlib
from preparedness_turn_completer.oai_turn_completer import OpenAITurnCompleter
from preparedness_turn_completer.turn_completer import TurnCompleter

from paperbench.grade import JudgeOutput, run_judge
from paperbench.judge.token_usage import get_total_token_usage
from paperbench.utils import get_timestamp

logger = structlog.stdlib.get_logger(component=__name__)


async def main(
    submission_path: Path,
    paper_id: str,
    judge_type: str,
    max_depth: int,
    out_dir: Path,
    code_only: bool,
    completer_config: TurnCompleter.Config | None = None,
    resources_provided: bool = False,
) -> None:
    # Judge the submission
    graded_task_tree = await run_judge(
        submission_path=submission_path,
        paper_id=paper_id,
        judge_type=judge_type,
        completer_config=completer_config,
        out_dir=out_dir,
        max_depth=max_depth,
        code_only=code_only,
        resources_provided=resources_provided,
    )

    token_usage = None

    if judge_type == "simple":
        token_usage = get_total_token_usage(graded_task_tree)

    # Save judging outputs
    path_to_judge_output = out_dir / "grader_output.json"
    judge_output = JudgeOutput(
        judge_type=judge_type,
        completer_config=completer_config,
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
        choices=["low", "medium", "high", "none"],
        required=False,
        default="high",
        help="Reasoning effort to use for the completion, if using a model/judge that supports it.",
    )

    args = parser.parse_args()

    if args.judge == "simple" and not args.model:
        parser.error("--model is required when using simple judge.")

    completer_config = (
        # TODO: allow user to specify TurnCompleter.Config class
        OpenAITurnCompleter.Config(
            model=args.model,
            reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
        )
        if args.judge == "simple"
        else None
    )
    asyncio.run(
        main(
            submission_path=args.submission_path,
            paper_id=args.paper_id,
            judge_type=args.judge,
            max_depth=args.max_depth,
            out_dir=args.out_dir,
            code_only=args.code_only,
            completer_config=completer_config,
            resources_provided=args.resources_provided,
        )
    )
