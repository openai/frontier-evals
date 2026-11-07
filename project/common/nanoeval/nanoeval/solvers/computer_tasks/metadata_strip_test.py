from pydantic import BaseModel

from nanoeval.solvers.computer_tasks.solver import strip_all_metadata


class TestMessage(BaseModel):
    metadata: dict[str, object]


class TestConversation(BaseModel):
    metadata: dict[str, object]
    messages: list[TestMessage]


def test_strip_all_metadata_does_not_mutate_original_conversation() -> None:
    original = TestConversation(
        metadata={"keep": "conversation", "drop": "secret"},
        messages=[
            TestMessage(metadata={"keep": "message", "drop": "secret"}),
        ],
    )

    stripped = strip_all_metadata(original, allowed_metadata_fields=["keep"])

    assert stripped is not original
    assert stripped.messages[0] is not original.messages[0]  # type: ignore[attr-defined]
    assert stripped.metadata == {"keep": "conversation"}  # type: ignore[attr-defined]
    assert stripped.messages[0].metadata == {"keep": "message"}  # type: ignore[attr-defined]

    assert original.metadata == {"keep": "conversation", "drop": "secret"}
    assert original.messages[0].metadata == {"keep": "message", "drop": "secret"}
