import json
from pathlib import Path

import pytest

from paperbench.paper_registry import paper_registry
from paperbench.rubric.tasks import TaskNode

# Discover all paper IDs from the registry once for parametrization
PAPER_IDS = paper_registry.list_paper_ids()


@pytest.mark.parametrize("paper_id", PAPER_IDS)
def test_rubric_loads_into_tasknode_without_error(paper_id: str) -> None:
    # Given
    paper = paper_registry.get_paper(paper_id)
    rubric_path: Path = paper.rubric
    assert rubric_path.exists(), f"Rubric file does not exist for paper '{paper_id}': {rubric_path}"

    # When
    with open(rubric_path, "r") as f:
        data = json.load(f)
    task_tree = TaskNode.from_dict(data)

    # Then
    assert isinstance(task_tree, TaskNode), (
        f"Expected a TaskNode when loading rubric for paper '{paper_id}', got {type(task_tree)}"
    )
