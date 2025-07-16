import json
import math
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Callable, Generator, Type

import pytest
from paperbench.judge.judge import DummyJudge, Judge, SimpleJudge
from paperbench.rubric.tasks import TaskNode
from paperbench.utils import in_ci

non_dummy_judges = [SimpleJudge]


def get_ancestor(name: str) -> Path:
    """Returns the path to an ancestor directory with `name`, starting from the current file."""

    dir = Path(__file__).parent

    while dir != dir.parent:
        if dir.name == name:
            return dir
        dir = dir.parent

    raise Exception(f"No `{name}` directory found from {Path(__file__).parent} to root.")


@pytest.fixture
def submission_factory() -> Generator[Callable[[str], Path], None, None]:
    """Creates a submission directory from a given fixture name."""

    tmpdir = TemporaryDirectory()

    def create_gold_submission(submission_name: str) -> Path:
        assert (
            len(submission_name.split(".")) and submission_name.count(".") == 1
        ), "Expected `submission_name` to be of the form `{{paper_id}}.{{uuid}}`."

        paper_id, uuid = submission_name.split(".")
        src = get_ancestor("tests") / "unit" / "fixtures" / "submissions" / paper_id / uuid

        # Handle the edge case of an empty submission. This doesn't exist in the fixtures directory
        # by default, since Git only tracks files, and therefore empty directories are not tracked.
        if paper_id == "empty" and uuid == "gold":
            src.mkdir(parents=True, exist_ok=True)

        dst = Path(tmpdir.name)
        shutil.copytree(src, dst, dirs_exist_ok=True)

        return dst

    yield create_gold_submission

    tmpdir.cleanup()


@pytest.fixture
def rubric_factory() -> Callable[[str], TaskNode]:
    """Creates a rubric from a given fixture name."""

    def create_rubric(rubric_name: str) -> TaskNode:
        path = get_ancestor("tests") / "unit" / "fixtures" / "rubrics" / f"{rubric_name}.json"
        with open(path, "r") as f:
            data = json.load(f)
        return TaskNode.from_dict(data)

    return create_rubric


@pytest.fixture
def empty_pdf() -> Generator[Path, None, None]:
    """Creates an empty PDF file."""

    tmp_file = NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    yield tmp_path

    tmp_path.unlink()


@pytest.fixture
def empty_markdown() -> Generator[Path, None, None]:
    """Creates an empty Markdown file."""

    tmp_file = NamedTemporaryFile(suffix=".md", delete=False)
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    yield tmp_path

    tmp_path.unlink()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "JudgeClass", [DummyJudge, *non_dummy_judges] if not in_ci() else [DummyJudge]
)
@pytest.mark.parametrize("task", ["empty", "hex_flag", "hex_flags", "nested_hex_flags"])
async def test_all_gold_submissions_achieve_a_perfect_score_on_a_trivial_rubric(
    task: str,
    JudgeClass: Type[Judge],
    empty_pdf: Path,
    empty_markdown: Path,
    submission_factory: Callable[[str], Path],
    rubric_factory: Callable[[str], TaskNode],
):
    # Given
    gold_submission = submission_factory(f"{task}.gold")
    rubric = rubric_factory("trivial")
    judge = JudgeClass(
        paper_path=empty_pdf,
        rubric=rubric,
        addendum=None,
        judge_addendum=None,
        submission_dir=gold_submission,
        model="gpt-4o",
        paper_md=empty_markdown,
    )

    # When
    graded_tree = await judge.grade()

    # Then
    assert math.isclose(
        graded_tree.score, 1.0
    ), f"Expected score to be 1.0, but got {graded_tree.score}"


@pytest.mark.asyncio
@pytest.mark.parametrize("JudgeClass", non_dummy_judges)  # Skip DummyJudge
@pytest.mark.skipif(in_ci(), reason="Test does not apply to DummyJudge")
@pytest.mark.parametrize("task", ["empty", "hex_flag", "hex_flags", "nested_hex_flags"])
async def test_all_gold_submissions_achieve_a_null_score_on_an_impossible_rubric(
    task: str,
    JudgeClass: Type[Judge],
    empty_pdf: Path,
    empty_markdown: Path,
    submission_factory: Callable[[str], Path],
    rubric_factory: Callable[[str], TaskNode],
):
    # Given
    gold_submission = submission_factory(f"{task}.gold")
    rubric = rubric_factory("impossible")
    judge = JudgeClass(
        paper_path=empty_pdf,
        rubric=rubric,
        addendum=None,
        judge_addendum=None,
        submission_dir=gold_submission,
        model="gpt-4o",
        paper_md=empty_markdown,
    )

    # When
    graded_tree = await judge.grade()

    # Then
    assert math.isclose(
        graded_tree.score, 0.0
    ), f"Expected score to be 0.0, but got {graded_tree.score}"


@pytest.mark.asyncio
@pytest.mark.parametrize("JudgeClass", non_dummy_judges)  # Skip DummyJudge
@pytest.mark.skipif(in_ci(), reason="Test does not apply to DummyJudge")
@pytest.mark.parametrize("task", ["empty", "hex_flag", "hex_flags", "nested_hex_flags"])
async def test_all_gold_submissions_achieve_a_perfect_score_on_their_corresponding_rubric(
    task: str,
    JudgeClass: Judge,
    empty_pdf: Path,
    empty_markdown: Path,
    submission_factory: Callable[[str], Path],
    rubric_factory: Callable[[str], TaskNode],
):
    # Given
    gold_submission = submission_factory(f"{task}.gold")
    rubric = rubric_factory(task)
    judge = JudgeClass(
        paper_path=empty_pdf,
        rubric=rubric,
        addendum=None,
        judge_addendum=None,
        submission_dir=gold_submission,
        model="gpt-4o",
        paper_md=empty_markdown,
    )

    # When
    graded_tree = await judge.grade()

    # Then
    assert math.isclose(
        graded_tree.score, 1.0
    ), f"Expected score to be 1.0, but got {graded_tree.score}"


@pytest.mark.asyncio
@pytest.mark.parametrize("JudgeClass", non_dummy_judges)  # Skip DummyJudge
@pytest.mark.skipif(in_ci(), reason="Test does not apply to DummyJudge")
@pytest.mark.parametrize("task", ["hex_flag", "hex_flags", "nested_hex_flags"])
async def test_empty_submission_achieves_a_null_score_on_all_non_trvial_rubrics(
    task: str,
    JudgeClass: Judge,
    empty_pdf: Path,
    empty_markdown: Path,
    submission_factory: Callable[[str], Path],
    rubric_factory: Callable[[str], TaskNode],
):
    # Given
    empty_submission = submission_factory("empty.gold")
    rubric = rubric_factory(task)
    judge = JudgeClass(
        paper_path=empty_pdf,
        rubric=rubric,
        addendum=None,
        judge_addendum=None,
        submission_dir=empty_submission,
        model="gpt-4o",
        paper_md=empty_markdown,
    )

    # When
    graded_tree = await judge.grade()

    # Then
    assert math.isclose(
        graded_tree.score, 0.0
    ), f"Expected score to be 0.0, but got {graded_tree.score}"


@pytest.mark.asyncio
@pytest.mark.parametrize("JudgeClass", non_dummy_judges)  # Skip DummyJudge
@pytest.mark.skipif(in_ci(), reason="Test does not apply to DummyJudge")
@pytest.mark.parametrize(
    "n_missing",
    [
        # TODO (dane): These only fail sometimes. Make the judge deterministic!
        pytest.param(1, marks=pytest.mark.skip("Known to fail")),
        pytest.param(4, marks=pytest.mark.skip("Known to fail")),
    ],
)
async def test_submission_with_n_missing_files_to_the_hex_flags_task_achieves_a_partial_score(
    n_missing: int,
    JudgeClass: Judge,
    empty_pdf: Path,
    empty_markdown: Path,
    submission_factory: Callable[[str], Path],
    rubric_factory: Callable[[str], TaskNode],
):
    # Given
    submission = submission_factory("hex_flags.gold")
    files = list(sorted(submission.rglob("*.txt")))

    for file in files[:n_missing]:  # Delete the first `n_missing` files
        file.unlink()

    expected_score = (len(files) - n_missing) / len(files)
    rubric = rubric_factory("hex_flags")
    judge = JudgeClass(
        paper_path=empty_pdf,
        rubric=rubric,
        addendum=None,
        judge_addendum=None,
        submission_dir=submission,
        model="gpt-4o",
        paper_md=empty_markdown,
    )

    # When
    graded_tree = await judge.grade()
    actual_score = graded_tree.score

    # Then
    assert math.isclose(
        actual_score, expected_score
    ), f"Expected score to be {expected_score}, but got {actual_score}"


@pytest.mark.asyncio
@pytest.mark.parametrize("JudgeClass", non_dummy_judges)  # Skip DummyJudge
@pytest.mark.skipif(in_ci(), reason="Test does not apply to DummyJudge")
async def test_nested_context_preserved_in_grading(
    JudgeClass: Type[Judge],
    empty_pdf: Path,
    empty_markdown: Path,
    submission_factory: Callable[[str], Path],
    rubric_factory: Callable[[str], TaskNode],
):
    """
    This test checks whether the nested context is preserved in grading.
    """

    # Given
    node_id = "4b9e1b5e-98a1-4f3c-88fa-fb9e7846c1bb"
    sample_id = "hex_flag_context"
    submission_id = "hex_flag.gold"
    submission = submission_factory(submission_id)
    rubric = rubric_factory(sample_id)
    judge = JudgeClass(
        paper_path=empty_pdf,
        rubric=rubric,
        addendum=None,
        judge_addendum=None,
        submission_dir=submission,
        model="gpt-4o",
        paper_md=empty_markdown,
    )

    # When
    graded_tree = await judge.grade()
    actual_score = graded_tree.find(node_id).score

    # Then
    assert math.isclose(
        actual_score, 1.0
    ), f"Expected score for node {node_id} to be 1.0, but got {actual_score}"
