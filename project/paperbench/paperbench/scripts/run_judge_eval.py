import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog.stdlib
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)
from preparedness_turn_completer.turn_completer import TurnCompleter

from paperbench.judge.create_judge import create_judge, handle_judge_kwargs
from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.judge.judge_eval.evaluate import (
    JudgeMetrics,
    LeafScores,
    MetricSummary,
    calculate_judge_scores,
    compute_metrics,
)
from paperbench.judge.judge_eval.registry import example_registry
from paperbench.judge.token_usage import TokenUsage, get_total_token_usage
from paperbench.paper_registry import paper_registry

logger = structlog.stdlib.get_logger(component=__name__)


def make_judge_kwargs_serializable(judge_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Convert judge_kwargs to a serializable format.
    """
    serializable_judge_kwargs = judge_kwargs.copy()
    if serializable_judge_kwargs["completer_config"] is not None:
        completer_config: TurnCompleter.Config = serializable_judge_kwargs["completer_config"]
        serializable_judge_kwargs["completer_config"] = completer_config.model_dump(mode="json")
    return serializable_judge_kwargs


@dataclass(slots=True)
class SingleExampleResult:
    example_id: str
    paper_id: str
    judge_type: str
    judge_kwargs: dict[str, Any]
    metrics: JudgeMetrics
    token_usage: TokenUsage | None
    graded_task_tree: GradedTaskNode

    def to_dict(self) -> dict[str, Any]:
        serializable_judge_kwargs = make_judge_kwargs_serializable(self.judge_kwargs)

        return {
            "example_id": self.example_id,
            "paper_id": self.paper_id,
            "judge_type": self.judge_type,
            "judge_kwargs": serializable_judge_kwargs,
            "metrics": asdict(self.metrics),
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
            "graded_task_tree": self.graded_task_tree.to_dict(),
        }


@dataclass(slots=True)
class JudgeEvalResult:
    judge_type: str
    code_only: bool
    judge_kwargs: dict[str, Any]
    results: list[SingleExampleResult]
    aggregate_metrics: JudgeMetrics
    aggregate_token_usage: dict[str, Any] | None
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        serializable_judge_kwargs = make_judge_kwargs_serializable(self.judge_kwargs)

        return {
            "judge_type": self.judge_type,
            "code_only": self.code_only,
            "judge_kwargs": serializable_judge_kwargs,
            "results": [result.to_dict() for result in self.results],
            "aggregate_metrics": asdict(self.aggregate_metrics),
            "aggregate_token_usage": self.aggregate_token_usage,
            "timestamp": self.timestamp,
        }


async def evaluate_single_example(
    example_id: str,
    judge_type: str,
    completer_config: TurnCompleter.Config,
    log_dir: Path,
    code_only: bool = False,
) -> tuple[SingleExampleResult, dict[str | None, LeafScores]]:
    """
    Evaluate a judge on a single example from the evaluation dataset.

    Args:
        example_id: ID of the example run to evaluate
        judge_type: Type of judge to evaluate
        model_name: Name of the model to use (required for non-dummy judges)
        code_only: Whether to grade on PaperBench-CodeOnly
        reasoning_effort: Level of reasoning effort to use for the completion for reasoning models

    Returns:
        Dictionary containing evaluation results and metadata
        Dictionary containing the predicted and true scores for each task category and overall
    """
    # Get example run and paper info
    example = example_registry.get_example_run(example_id)
    paper = paper_registry.get_paper(example.paper_id)

    # Load rubric and expected results
    expected_result: None | GradedTaskNode
    with open(example.expected_result) as f:
        expected_result = GradedTaskNode.from_dict(json.load(f))
    if code_only:
        expected_result = expected_result.code_only() or expected_result.set_task_category(
            "Code Development"
        ).set_sub_tasks([])

    # Initialize appropriate judge
    log_path = log_dir / example_id
    log_path.mkdir(parents=True, exist_ok=True)
    judge_kwargs = handle_judge_kwargs(judge_type, code_only, paper, completer_config)
    judge = create_judge(
        judge_type=judge_type,
        judge_kwargs=judge_kwargs,
        paper_path=paper.paper_pdf,
        rubric=expected_result,
        addendum=paper.addendum.read_text() if paper.addendum else None,
        judge_addendum=paper.judge_addendum.read_text() if paper.judge_addendum.exists() else None,
        submission_dir=example.submission_dir,
        paper_md=paper.paper_md,
        log_path=log_path,
    )

    # Grade submission
    graded_task_tree = await judge.judge()

    # Evaluate judge's grading
    # evaluation_results, scores = calculate_judge_scores(graded_task_tree, expected_result)
    judge_summary = calculate_judge_scores(graded_task_tree, expected_result)
    evaluation_results = judge_summary.metrics
    scores = judge_summary.scores
    token_usage = None
    if judge_type == "simple":
        token_usage = get_total_token_usage(graded_task_tree)

    single_exampale_result = SingleExampleResult(
        example_id=example_id,
        paper_id=example.paper_id,
        judge_type=judge_type,
        judge_kwargs=judge_kwargs,
        metrics=evaluation_results,
        token_usage=token_usage,
        graded_task_tree=graded_task_tree,
    )

    return single_exampale_result, scores


async def main(
    example_ids: list[str],
    judge_type: str,
    output_dir: Path,
    completer_config: TurnCompleter.Config | None = None,
    code_only: bool = False,
) -> None:
    """
    Run judge evaluation on specified example runs.

    Args:
        example_ids: List of example run IDs to evaluate
        judge_type: Type of judge to evaluate
        model_name: Name of model to use (for non-dummy judges)
        output_dir: Directory to write evaluation results to
    """
    model_name = None
    if isinstance(completer_config, OpenAICompletionsTurnCompleter.Config):
        model_name = completer_config.model
    else:
        raise NotImplementedError("Only OpenAICompletionsTurnCompleter is supported for now")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_prefix = f"{judge_type}_{model_name}" if model_name is not None else judge_type
    run_dir = output_dir / f"{run_dir_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[SingleExampleResult] = []
    all_scores = {
        kind: LeafScores()
        for kind in ["Overall", "Code Development", "Code Execution", "Result Analysis"]
    }
    for example_id in example_ids:
        logger.info(f"Evaluating judge on example run: {example_id}")
        try:
            result, scores = await evaluate_single_example(
                example_id=example_id,
                completer_config=completer_config,
                judge_type=judge_type,
                log_dir=run_dir,
                code_only=code_only,
            )
            results.append(result)
            for kind, leaf_scores in scores.items():
                if kind is not None:
                    all_scores[kind].extend(leaf_scores)
        except Exception as e:
            logger.exception(f"Failed to evaluate example {example_id}: {e}", exc_info=True)
            continue

    # Calculate aggregate metrics by combining scores from all examples
    overall_metrics = compute_metrics(all_scores["Overall"].y_true, all_scores["Overall"].y_pred)
    stratified_metrics: dict[str | None, MetricSummary] = {
        category: compute_metrics(all_scores[category].y_true, all_scores[category].y_pred)
        for category in ["Code Development", "Code Execution", "Result Analysis"]
    }
    aggregate_metrics = JudgeMetrics(
        overall=overall_metrics,
        stratified=stratified_metrics,
    )

    agg_token_usage = None
    if judge_type == "simple":
        total_token_usage = TokenUsage()
        model_stats: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"in": [], "out": []})
        for result in results:
            if result.token_usage is None:
                continue
            for model, usage in result.token_usage.usage.items():
                total_token_usage.add_usage(model, usage["in"], usage["out"])
                model_stats[model]["in"].append(usage["in"])
                model_stats[model]["out"].append(usage["out"])
        agg_token_usage = {
            "total": total_token_usage.to_dict(),
            "mean": {
                model: {"in": np.mean(stats["in"]), "out": np.mean(stats["out"])}
                for model, stats in model_stats.items()
            },
        }

    # Save results
    evaluation_results = JudgeEvalResult(
        judge_type=judge_type,
        code_only=code_only,
        judge_kwargs=handle_judge_kwargs(judge_type, code_only, None, completer_config),
        results=results,
        aggregate_metrics=aggregate_metrics,
        aggregate_token_usage=agg_token_usage,
        timestamp=timestamp,
    )

    results_file = run_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(evaluation_results.to_dict(), f, indent=2)
    logger.info(f"Evaluation results written to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate judge performance on example runs.")
    parser.add_argument(
        "--judge",
        "-j",
        required=True,
        choices=["dummy", "random", "simple"],
        help="Type of judge to evaluate.",
    )
    parser.add_argument(
        "--example-ids",
        nargs="+",
        help="IDs of example runs to evaluate. If not provided, evaluates all examples.",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Model to use (required for non-dummy judges).",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("judge_eval_results"),
        help="Directory to write evaluation results to.",
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Only grade 'Code Development' nodes",
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

    # If no example IDs provided, use all examples in registry
    if not args.example_ids:
        args.example_ids = example_registry.list_example_ids()

    completer_config = (
        # TODO: allow user to specify TurnCompleter.Config class
        OpenAICompletionsTurnCompleter.Config(
            model=args.model,
            reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
        )
        if args.judge == "simple"
        else None
    )
    asyncio.run(
        main(
            example_ids=args.example_ids,
            judge_type=args.judge,
            completer_config=completer_config,
            output_dir=args.output_dir,
            code_only=args.code_only,
        )
    )
