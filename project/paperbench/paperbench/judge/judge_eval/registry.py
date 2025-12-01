from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog.stdlib

from paperbench.utils import get_paperbench_data_dir

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass(frozen=True)
class ExampleRun:
    """An example run for evaluating judge performance, containing a submission and its expected grading."""

    id: str
    paper_id: str
    submission_dir: Path
    expected_result: Path

    def __post_init__(self) -> None:
        assert isinstance(self.id, str), "Example run id must be a string."
        assert isinstance(self.paper_id, str), "Paper id must be a string."
        assert isinstance(self.submission_dir, Path), "Submission directory must be a Path."
        assert isinstance(self.expected_result, Path), "Expected result must be a Path."
        assert len(self.id) > 0, "Example run id cannot be empty."
        assert len(self.paper_id) > 0, "Paper id cannot be empty."
        assert self.submission_dir.exists(), (
            f"Submission directory {self.submission_dir} does not exist."
        )
        assert self.expected_result.exists(), (
            f"Expected result file {self.expected_result} does not exist."
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ExampleRun:
        try:
            return ExampleRun(
                id=data["id"],
                paper_id=data["paper_id"],
                submission_dir=data["submission_dir"],
                expected_result=data["expected_result"],
            )
        except KeyError as e:
            raise ValueError("Missing key in example run config!") from e


class ExampleRunRegistry:
    """Registry for judge evaluation example runs."""

    def __init__(self, dataset_dir: Path | None = None):
        """
        Initializes the registry with a dataset directory.

        Args:
            dataset_dir: Optional; Path to the directory containing example runs.
                         Defaults to a directory within the module if not provided.
        """
        if dataset_dir is None:
            dataset_dir = self.get_default_dataset_dir()
        self.dataset_dir = dataset_dir

    def get_default_dataset_dir(self) -> Path:
        """Returns the default dataset directory."""
        return get_paperbench_data_dir() / "judge_eval"

    def get_examples_dir(self) -> Path:
        """Retrieves the examples directory within the registry."""
        return self.dataset_dir

    def get_example_run(self, example_id: str) -> ExampleRun:
        """
        Fetch an example run from the registry.

        Args:
            example_id: ID of the example run in format "paper_id/example_name"
        """
        paper_id, example_name = example_id.split("/", 1)
        example_dir = self.get_examples_dir() / paper_id / example_name
        if not example_dir.exists():
            raise ValueError(f"Example run {example_id} not found in registry")

        submission_dir = example_dir / "submission"
        expected_result = example_dir / "grading" / "expected_result.json"

        return ExampleRun.from_dict(
            {
                "id": example_id,
                "paper_id": paper_id,
                "submission_dir": submission_dir,
                "expected_result": expected_result,
            }
        )

    def list_example_ids(self) -> list[str]:
        """
        List all example run IDs available in the registry, sorted alphabetically.
        Returns IDs in format "paper_id/example_name".
        """
        examples_dir = self.get_examples_dir()
        example_ids = []

        # Iterate through paper directories
        for paper_dir in sorted(examples_dir.iterdir()):
            if not paper_dir.is_dir() or paper_dir.name.startswith("."):
                continue

            # Iterate through example directories within each paper
            for example_dir in sorted(paper_dir.iterdir()):
                if not example_dir.is_dir() or example_dir.name.startswith("."):
                    continue

                example_ids.append(f"{paper_dir.name}/{example_dir.name}")

        return example_ids


example_registry = ExampleRunRegistry()
