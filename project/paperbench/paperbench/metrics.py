import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Hashable

import dateutil.parser
import numpy as np
from paperbench.judge.judge import GradedTaskNode, disqualify
from tqdm import tqdm

EXPECTED_PAPERS = 20


@dataclass
class PaperEvaluation:
    """A single evaluation of a paper"""

    paper_run_id: str
    paper_id: str
    graded_task_node: GradedTaskNode


@dataclass
class EvaluationRun:
    """A single run of the Evaluation, i.e. where all papers have been evaluated"""

    seed: Hashable
    paper_evaluations: dict[str, PaperEvaluation]

    def is_complete(self, expected_papers: int = EXPECTED_PAPERS) -> bool:
        return len(self.paper_evaluations) == expected_papers

    def is_valid(self) -> bool:
        paper_ids = [pe.paper_id for pe in self.paper_evaluations.values()]
        return len(paper_ids) == len(set(paper_ids))


@dataclass
class MetricResult:
    mean: float
    std_err: float
    n_runs: int


@dataclass
class ParsedEntry:
    """Contains parsed JSONL entry data"""

    agent_id: str
    paper_id: str
    paper_run_id: str
    timestamp: float
    graded_task_tree: GradedTaskNode


def compute_ars(
    eval_run: EvaluationRun,
    task_node_transform: Callable[[GradedTaskNode], GradedTaskNode] | None = None,
) -> float:
    """
    Computes the average replication score (ARS) for a single evaluation run
    """
    assert eval_run.is_valid(), "Evaluation run contains duplicate paper evaluations"

    scores = []

    for paper_eval in eval_run.paper_evaluations.values():
        graded_task_node = paper_eval.graded_task_node
        if task_node_transform is not None:
            graded_task_node = task_node_transform(graded_task_node)
        score = graded_task_node.score
        scores.append(score)

    return np.mean(scores).item()


def compute_agg_stats(
    evaluation_runs: list[EvaluationRun],
    compute_ars_kwargs: dict = {},
    expected_papers: int = EXPECTED_PAPERS,
) -> MetricResult:
    """
    Computes aggregate statistics for replication scores across multiple eval runs.
    Returns the mean score, standard error, and number of valid seeds.
    """
    # Filter for complete evaluations (i.e. all papers have been evaluated)
    complete_evaluations = [eval for eval in evaluation_runs if eval.is_complete(expected_papers)]

    scores = [compute_ars(eval_run, **compute_ars_kwargs) for eval_run in complete_evaluations]

    return MetricResult(
        mean=np.mean(scores).item(),
        std_err=(np.std(scores, ddof=1) / np.sqrt(len(scores))).item() if scores else float("nan"),
        n_runs=len(complete_evaluations),
    )


def per_paper_results(eval_runs: list[EvaluationRun], n_runs: int) -> dict[str, dict]:
    """
    Computes the mean and standard error of the replication score for each paper
    over the expected number of runs.
    """
    paper_ids = {
        pe.paper_id for eval_run in eval_runs for pe in eval_run.paper_evaluations.values()
    }

    def _init_result(num_runs) -> dict:
        seeds = {f"run_{i}": None for i in range(1, num_runs + 1)}
        results = {
            "mean": None,
            "std_err": None,
            "n_runs": None,
            **seeds,
        }
        return results

    results = {paper_id: _init_result(n_runs) for paper_id in paper_ids}

    # first, fill in the scores for each seed that's available
    for i, eval_run in enumerate(eval_runs, start=1):
        for paper_eval in eval_run.paper_evaluations.values():
            paper_id = paper_eval.paper_id
            score = paper_eval.graded_task_node.score
            seed = f"run_{i}"
            results[paper_id][seed] = score

    # then compute the mean/stderr over the available seeds
    for paper_id, paper_results in results.items():
        avail_scores = [score for score in paper_results.values() if score is not None]
        results[paper_id]["mean"] = np.mean(avail_scores).item()
        results[paper_id]["n_runs"] = len(avail_scores)
        results[paper_id]["std_err"] = (
            (np.std(avail_scores, ddof=1) / np.sqrt(len(avail_scores))).item()
            if avail_scores
            else float("nan")
        )

    return results


def check_disqualification(
    paper_eval: PaperEvaluation, disquailified_paper_runs: set[str]
) -> PaperEvaluation:
    """
    Checks if a PaperEvaluation is from a disqualified paper run
    and if so sets the graded task node score to 0.0
    """
    if paper_eval.paper_run_id in disquailified_paper_runs:
        disqualified_graded_node = disqualify(paper_eval.graded_task_node)
        paper_eval.graded_task_node = disqualified_graded_node
    return paper_eval


def parse_disqualified_runs(disqualification_data_path: Path) -> set[str]:
    with open(disqualification_data_path, "r") as f:
        return {line.strip() for line in f}


def parse_run_data(
    run_data_path: Path,
    disqualification_data_path: Path | None = None,
    seeds_to_keep: int | None = None,
) -> dict[str, list[EvaluationRun]]:
    """
    Parses run data from JSONL files and organizes it into EvaluationRun objects.
    Keeps only the N most recent seeds for each agent based on timestamps.

    Args:
        run_data_path: Directory path containing evaluation JSONL files
        disqualification_data_path: (Optional) Path to a file where each line is a disqualified
        paper run ID
        seeds_to_keep: (Optional) Number of most recent seeds to keep per agent

    Returns:
        Dictionary mapping agent IDs to lists of EvaluationRun objects
        where each EvaluationRun contains 1 seed of paper evaluations
    """
    agent_runs = {}

    # Helper function to validate and extract required data
    def parse_jsonl_entry(entry: dict) -> ParsedEntry | None:
        if not all(
            [
                entry.get("record_type") == "extra",
                entry.get("data", {}).get("pb_result", {}).get("grader_output"),
                entry.get("data", {}).get("run_group_id"),
                entry.get("timestamp"),
            ]
        ):
            return None

        pb_result = entry["data"]["pb_result"]
        if not pb_result.get("grader_success"):
            return None

        run_group_id = entry["data"]["run_group_id"]
        paper_run_id = entry["data"]["run_id"]
        agent_id = run_group_id.split("_")[-1]
        paper_id = pb_result["paper_id"]
        timestamp = dateutil.parser.parse(entry["timestamp"]).timestamp()
        graded_task_tree = GradedTaskNode.from_dict(pb_result["grader_output"]["graded_task_tree"])

        return ParsedEntry(
            agent_id=agent_id,
            paper_id=paper_id,
            paper_run_id=paper_run_id,
            timestamp=timestamp,
            graded_task_tree=graded_task_tree,
        )

    if disqualification_data_path is not None:
        disqualified_paper_runs = parse_disqualified_runs(disqualification_data_path)
    else:
        disqualified_paper_runs = set()

    for file in tqdm(sorted((run_data_path.glob("*.jsonl"))), desc="Parsing run data"):
        with open(file, "r") as f:
            for line in f:
                entry = json.loads(line)
                parsed_entry = parse_jsonl_entry(entry)
                if not parsed_entry:
                    continue

                # Initialize agent and seed data structures if needed
                if parsed_entry.agent_id not in agent_runs:
                    agent_runs[parsed_entry.agent_id] = {}

                paper_eval = PaperEvaluation(
                    paper_run_id=parsed_entry.paper_run_id,
                    paper_id=parsed_entry.paper_id,
                    graded_task_node=parsed_entry.graded_task_tree,
                )
                paper_eval = check_disqualification(paper_eval, disqualified_paper_runs)

                if parsed_entry.paper_id not in agent_runs[parsed_entry.agent_id]:
                    agent_runs[parsed_entry.agent_id][parsed_entry.paper_id] = []

                agent_runs[parsed_entry.agent_id][parsed_entry.paper_id].append(
                    {"paper_eval": paper_eval, "timestamp": parsed_entry.timestamp}
                )

    # Convert to final format, keeping only the N most recent seeds
    agent_to_eval_runs = {agent: [] for agent in agent_runs.keys()}

    for agent, paper_data in agent_runs.items():

        # keep only N most recent seeds
        filtered_paper_data = {}
        for paper_id, data in paper_data.items():
            sorted_data = sorted(data, key=lambda x: x["timestamp"], reverse=True)
            filtered_paper_data[paper_id] = sorted_data[:seeds_to_keep]

        # then, create the N EvaluationRun objects
        max_num_seeds = max([len(data) for data in filtered_paper_data.values()])
        for seed in range(max_num_seeds):
            eval_run = EvaluationRun(seed=seed, paper_evaluations={})
            for paper_id, data in filtered_paper_data.items():
                # some evaluation runs may not have all papers evaluated
                if seed == len(data):
                    continue
                paper_eval = data[seed]["paper_eval"]
                eval_run.paper_evaluations[paper_eval.paper_id] = paper_eval

            agent_to_eval_runs[agent].append(eval_run)

    return agent_to_eval_runs


if __name__ == "__main__":
    """Example usage of parse_run_data and compute_agg_stats"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-data-path", type=Path, required=True)
    parser.add_argument(
        "--disqualified-runs-path",
        type=Path,
        required=False,
        default=None,
        help="Path to file where each line is a disqualified paper run ID",
    )
    args = parser.parse_args()

    agent_to_eval_runs = parse_run_data(
        args.run_data_path, args.disqualified_runs_path, seeds_to_keep=3
    )

    results = {}
    for agent in agent_to_eval_runs.keys():
        results[agent] = compute_agg_stats(agent_to_eval_runs[agent])

    results = {agent: asdict(metric) for agent, metric in results.items()}

    import pandas as pd

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("mean", ascending=False)

    print(results_df)
