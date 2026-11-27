import asyncio
from pathlib import Path
from types import SimpleNamespace

from paperbench.judge.simple import SimpleJudge
from paperbench.rubric.tasks import TaskNode


class CharEncoder:
    """Stub encoder: one character is one token."""

    def encode(self, text: str, disallowed_special: tuple[str, ...] | str = ()) -> list[str]:
        return list(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


def _judge() -> SimpleJudge:
    judge = SimpleJudge.__new__(SimpleJudge)
    judge.token_encoder = CharEncoder()  # type: ignore[assignment]
    return judge


def test_truncate_in_token_space_keeps_nothing_for_negative_budget() -> None:
    judge = _judge()
    assert judge._truncate_in_token_space("abcdef", 3) == "abc"
    assert judge._truncate_in_token_space("abcdef", 0) == ""
    assert judge._truncate_in_token_space("abcdef", -2) == ""


def test_prepare_relevant_files_keeps_nothing_for_negative_budget(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x" * 9800)

    class Completer:
        async def async_completion(self, conversation: object) -> SimpleNamespace:
            return SimpleNamespace(output_messages=[SimpleNamespace(content="a.py")])

    judge = _judge()
    judge.completer = Completer()  # type: ignore[assignment]
    judge.submission_dir = tmp_path
    judge.computer = None
    judge.paper_md = ""
    judge.joined_addendum = ""
    judge.log_path = None
    judge.tree_structures = {"Code Execution": "a.py"}
    task = TaskNode(
        id="t",
        requirements="r",
        weight=1,
        sub_tasks=[],
        task_category="Code Execution",
    )

    # 2000 is the buffer _prepare_relevant_files subtracts, so -1500 gives a budget of -3500.
    for avail in (10_000, 2_000, 500, -1_500):
        judge.avail_context_lens = {"Code Execution": avail}
        kept = asyncio.run(judge._prepare_relevant_files(task))
        if avail - 2000 <= 0:
            assert kept == "", f"budget {avail - 2000} kept {len(kept)} characters"
        else:
            assert kept != ""
