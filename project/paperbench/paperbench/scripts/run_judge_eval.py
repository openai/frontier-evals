import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import structlog.stdlib
from paperbench.judge.create_judge import create_judge, handle_judge_kwargs
from paperbench.judge.judge import GradedTaskNode, TokenUsage, get_total_token_usage
from paperbench.judge.judge_eval.evaluate import calculate_judge_scores, compute_metrics
from paperbench.judge.judge_eval.registry import example_registry
from paperbench.paper_registry import paper_registry

logger = structlog.stdlib.get_logger(component=__name__)


async def evaluate_single_example(
    example_id: str,
    judge_type: str,
    log_dir: Path,
    model_name: str | None = None,
    code_only: bool = False,
    reasoning_effort: str | None = None,
) -> tuple[dict, dict]:
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
    with open(example.expected_result) as f:
        expected_result = GradedTaskNode.from_dict(json.load(f))
    if code_only:
        expected_result = expected_result.code_only() or expected_result.set_task_category(
            "Code Development"
        ).set_sub_tasks([])

    # Initialize appropriate judge
    log_path = log_dir / example_id
    log_path.mkdir(parents=True, exist_ok=True)
    judge_kwargs = handle_judge_kwargs(judge_type, code_only, paper, model_name, reasoning_effort)
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
    graded_task_tree = await judge.grade()

    # Evaluate judge's grading
    evaluation_results, scores = calculate_judge_scores(graded_task_tree, expected_result)
    token_usage = None
    if judge_type == "simple":
        token_usage = get_total_token_usage(graded_task_tree).to_dict()

    results = {
        "example_id": example_id,
        "paper_id": example.paper_id,
        "judge_type": judge_type,
        "judge_kwargs": judge_kwargs,
        "model_name": model_name,
        "metrics": evaluation_results,
        "token_usage": token_usage,
        "graded_task_tree": graded_task_tree.to_dict(),
    }

    return results, scores


async def main(
    example_ids: list[str],
    judge_type: str,
    output_dir: Path,
    model_name: str | None = None,
    code_only: bool = False,
    reasoning_effort: Optional[str] = None,
):
    """
    Run judge evaluation on specified example runs.

    Args:
        example_ids: List of example run IDs to evaluate
        judge_type: Type of judge to evaluate
        model_name: Name of model to use (for non-dummy judges)
        output_dir: Directory to write evaluation results to
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_prefix = f"{judge_type}_{model_name}" if model_name is not None else judge_type
    run_dir = output_dir / f"{run_dir_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_scores = {
        kind: ([], [])
        for kind in ["Overall", "Code Development", "Code Execution", "Result Analysis"]
    }
    for example_id in example_ids:
        logger.info(f"Evaluating judge on example run: {example_id}")
        try:
            result, scores = await evaluate_single_example(
                example_id=example_id,
                judge_type=judge_type,
                log_dir=run_dir,
                model_name=model_name,
                code_only=code_only,
                reasoning_effort=reasoning_effort,
            )
            results.append(result)
            for kind, (y_pred, y_true) in scores.items():
                all_scores[kind][0].extend(y_pred)
                all_scores[kind][1].extend(y_true)
        except Exception as e:
            logger.exception(f"Failed to evaluate example {example_id}: {e}", exc_info=True)
            continue

    # Calculate aggregate metrics by combining scores from all examples
    aggregate_metrics = compute_metrics(all_scores["Overall"][1], all_scores["Overall"][0])
    aggregate_metrics["stratified"] = {
        category: compute_metrics(all_scores[category][1], all_scores[category][0])
        for category in ["Code Development", "Code Execution", "Result Analysis"]
    }

    agg_token_usage = None
    if judge_type == "simple":
        total_token_usage = TokenUsage()
        model_stats = defaultdict(lambda: {"in": [], "out": []})
        for result in results:
            for model, usage in result["token_usage"].items():
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
    evaluation_results = {
        "judge_type": judge_type,
        "code_only": code_only,
        "judge_kwargs": handle_judge_kwargs(
            judge_type, code_only, None, model_name, reasoning_effort
        ),
        "model_name": model_name,
        "results": results,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_token_usage": agg_token_usage,
        "timestamp": timestamp,
    }

    results_file = run_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info(f"Evaluation results written to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate judge performance on example runs.")
    parser.add_argument(
        "--judge",
        "-j",
        required=True,
        choices=["dummy", "random", "simple", "agent", "assistant"],
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
        choices=["low", "medium", "high"],
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

    asyncio.run(
        main(
            example_ids=args.example_ids,
            judge_type=args.judge,
            model_name=args.model,
            output_dir=args.output_dir,
            code_only=args.code_only,
            reasoning_effort=args.reasoning_effort,
        )
    )
