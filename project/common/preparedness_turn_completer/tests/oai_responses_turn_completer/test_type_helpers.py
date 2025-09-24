from preparedness_turn_completer.oai_responses_turn_completer.type_helpers import (
    is_assistant_message,
    is_content_part,
    is_content_part_list,
    is_text_part,
    is_text_parts_list,
    is_tool_message,
)


def test_is_text_part() -> None:
    assert not is_text_part("not a dict")


def test_is_text_parts_list() -> None:
    parts = [{"type": "text", "text": "hi"}, {"type": "text", "text": "world"}]
    assert is_text_parts_list(parts)
    assert not is_text_parts_list("string")
    assert not is_text_parts_list([{"type": "text"}])


def test_is_content_part() -> None:
    assert is_content_part({"type": "text", "text": "hi"})
    assert is_content_part(
        {"type": "file", "file": {"file_data": "data", "file_id": "id", "filename": "f"}}
    )
    assert not is_content_part({"type": "unknown", "text": "hi"})


def test_is_content_part_list() -> None:
    parts = [
        {"type": "text", "text": "a"},
        {"type": "file", "file": {"file_data": "d", "file_id": "i", "filename": "f"}},
    ]
    assert is_content_part_list(parts)
    assert not is_content_part_list("string")


def test_is_assistant_message() -> None:
    assert is_assistant_message({"role": "assistant"})
    assert not is_assistant_message({"role": "user"})


def test_is_tool_message() -> None:
    assert is_tool_message({"role": "tool", "tool_call_id": "123"})
    assert not is_tool_message({"role": "tool"})


def test_is_content_part_image_and_audio() -> None:
    """Image and audio content parts are correctly identified."""
    image_part = {"type": "image_url", "image_url": {"url": "u", "detail": "d"}}
    audio_part = {"type": "input_audio", "input_audio": {"audio_url": "u"}}
    assert is_content_part(image_part)
    assert is_content_part(audio_part)
    assert not is_content_part({"type": "image_url"})


def test_is_content_part_list_with_mixed_parts() -> None:
    """Lists containing valid content parts should be identified as content-part lists."""
    parts = [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "u", "detail": "d"}},
        {"type": "input_audio", "input_audio": {"audio_url": "u"}},
    ]
    assert is_content_part_list(parts)
