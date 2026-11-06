"""Check the API MCQ prompt against its answer-format contract."""

import pytest

from nanoeval.solvers.mcq_api import _answer_format_instruction


@pytest.mark.parametrize("multiple", [False, True])
def test_answer_format_uses_available_choices(multiple: bool) -> None:
    instruction = _answer_format_instruction(["A", "B", "C"], multiple)
    assert "A, B, C" in instruction
    assert "Think step by step before answering." in instruction
    if multiple:
        assert "Answer: $LETTER1, $LETTER2, ..." in instruction
        assert "where each LETTER is one of A, B, C" in instruction
    else:
        assert "Answer: $LETTER'" in instruction
        assert "where LETTER is one of A, B, C" in instruction
        assert "LETTER1" not in instruction
