import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from data/judge_eval/rice/0/grading/expected_results.json, ie the manually graded submission
HUMAN_JUDGE_SCORE = 0.18568121693121692


def load_grader_data(base_path: Path, paper: str = "rice") -> list[dict]:
    """
    Load grader output data from json files.

    Args:
        base_path: Base path to search for grader output files
        paper: Name of the paper being analyzed

    Returns:
        List of dictionaries containing depth, score and num_leaf_nodes
    """
    grader_files = list(base_path.glob("**/grader_output.json"))
    data = []

    for file_path in grader_files:
        parts = file_path.parent.name.split("_")
        depth = int(parts[1].replace("depth", ""))

        with open(file_path) as f:
            obj = json.load(f)
            score = obj["score"]
            num_leaf_nodes = obj["num_leaf_nodes"]
            data.append({"depth": depth, "score": score, "num_leaf_nodes": num_leaf_nodes})

    return data


def create_depth_score_plot(data: list[dict], output_path: str) -> None:
    """
    Create and save a bar plot showing scores by depth.

    Args:
        data: List of dictionaries containing depth, score and num_leaf_nodes
        output_path: Path where to save the output plot
    """
    # Create DataFrame and calculate statistics
    df = pd.DataFrame(data)
    stats = (
        df.groupby("depth")
        .agg({"score": ["mean", "std", "count"], "num_leaf_nodes": ["max"]})
        .reset_index()
    )
    stats.columns = ["depth", "mean", "std", "count", "max_leaves"]
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])

    # Create bar plot
    plt.rcParams.update({"font.size": 7})
    plt.figure(figsize=(6.75133 / 1.5, 3.75))
    x = np.arange(len(stats))
    plt.bar(x, stats["mean"], yerr=stats["stderr"], capsize=5)

    # Add horizontal line for human score
    plt.axhline(y=HUMAN_JUDGE_SCORE, color="red", linestyle="--", label="Human Judge")
    plt.legend()

    plt.ylabel("Reproduction Score")
    plt.grid(True, axis="y", linestyle="--", alpha=0.2)
    plt.xticks(x, [f"{d} depth / {m} leaves" for d, m in zip(stats["depth"], stats["max_leaves"])])
    plt.xticks(rotation=45, ha="right")

    # Add value labels on top of bars
    for i, row in stats.iterrows():
        plt.text(
            i,
            row["mean"] + row["stderr"] + 0.001,
            f"{row['mean']:.2f}±{row['stderr']:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=400, pad_inches=0.01)
    plt.show()
    plt.close()


def main():
    """Main function to run the plotting script."""
    base_path = Path("experiments/judge_max_depth/results/")
    output_path = "experiments/judge_max_depth/depth_scores.pdf"

    # Load and process data
    data = load_grader_data(base_path)

    # Create and save plot
    create_depth_score_plot(data, output_path)


if __name__ == "__main__":
    main()
