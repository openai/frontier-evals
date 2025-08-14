import json
from typing import Iterable

import pytest
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_assistant_message_param import ContentArrayOfContentPart
from openai.types.chat.chat_completion_content_part_param import File
from openai.types.chat.chat_completion_message import (
    Annotation as CompletionAnnotation,
)
from openai.types.chat.chat_completion_message import (
    AnnotationURLCitation as CompletionAnnotationURLCitation,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.responses import (
    Response,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)
from openai.types.responses.response_output_text import (
    AnnotationURLCitation as ResponsesAnnotationURLCitation,
)
from preparedness_turn_completer.oai_responses_turn_completer.converters import (
    _assistant_completion_to_response_input_items,
    _chat_completion_content_to_response_input_content,
    _chat_completion_part_to_response_part,
    _chat_completion_text_to_response_input_item,
    convert_conversation_to_response_input,
    convert_response_to_completion_messages,
)
from preparedness_turn_completer.turn_completer import TurnCompleter


def test_chat_completion_text_to_response_input_item_str() -> None:
    assert _chat_completion_text_to_response_input_item("hello") == "hello"


def test_chat_completion_text_to_response_input_item_invalid() -> None:
    """Non-text parts passed to text-to-input should error."""
    input_convo: TurnCompleter.RuntimeConversation = [
        {"type": "image_url", "image_url": {"url": "u", "detail": "low"}}  # type: ignore[list-item]
    ]
    with pytest.raises(ValueError):
        _chat_completion_text_to_response_input_item(input_convo)  # type: ignore[arg-type]


def test_chat_completion_text_to_response_input_item_parts() -> None:
    parts: list[ChatCompletionContentPartTextParam] = [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]
    items = _chat_completion_text_to_response_input_item(parts)
    assert isinstance(items, list)
    assert all(
        item.get("text") in ("a", "b") and item.get("type") == "input_text" for item in items
    )


def test_chat_completion_part_to_response_part_text() -> None:
    part: ChatCompletionContentPartTextParam = {"type": "text", "text": "hi"}
    item = _chat_completion_part_to_response_part(part)
    assert item.get("text") == "hi" and item.get("type") == "input_text"


def test_chat_completion_part_to_response_part_image() -> None:
    part: ChatCompletionContentPartImageParam = {
        "type": "image_url",
        "image_url": {"url": "u", "detail": "low"},
    }
    item = _chat_completion_part_to_response_part(part)
    assert item.get("image_url") == "u" and item.get("detail") == "low"


def test_chat_completion_part_to_response_part_file() -> None:
    part: File = {"type": "file", "file": {"file_data": "data", "file_id": "id", "filename": "f"}}
    item = _chat_completion_part_to_response_part(part)
    assert (
        item.get("file_data") == "data"
        and item.get("file_id") == "id"
        and item.get("filename") == "f"
    )


def test_chat_completion_part_to_response_part_unknown_raises() -> None:
    """Unknown content part types should error."""
    with pytest.raises(ValueError):
        _chat_completion_part_to_response_part({"type": "foo", "foo": "bar"})  # type: ignore[arg-type]


def test_convert_conversation_to_response_input_mixed() -> None:
    conv: TurnCompleter.RuntimeConversation = [
        {"role": "system", "content": "s"},
        {"role": "developer", "content": [{"type": "text", "text": "d"}]},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "tool_call_id": "cid", "content": "o"},
    ]
    items = convert_conversation_to_response_input(conv)
    assert len(items) == 5
    assert any(i.get("role", None) == "system" for i in items)
    assert any(isinstance(i, dict) and i.get("type") == "function_call_output" for i in items)


def test_chat_completion_content_to_response_input_content_mixed() -> None:
    """Mixed text and image content parts convert correctly."""
    parts: Iterable[ChatCompletionContentPartParam] = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "u", "detail": "low"}},
    ]
    input_items = _chat_completion_content_to_response_input_content(parts)
    assert isinstance(input_items, list)
    assert any(
        isinstance(item, dict) and item.get("type") == "input_text" and item.get("text") == "hi"
        for item in input_items
    )
    assert any(
        isinstance(item, dict)
        and item.get("type") == "input_image"
        and item.get("image_url") == "u"
        for item in input_items
    )


def test_chat_completion_content_to_response_input_content_audio_not_supported() -> None:
    """Audio content parts should raise NotImplementedError."""
    parts: Iterable[ChatCompletionContentPartParam] = [
        {"input_audio": {"data": "placeholder", "format": "mp3"}, "type": "input_audio"}
    ]
    with pytest.raises(NotImplementedError):
        _chat_completion_content_to_response_input_content(parts)


def test_convert_response_to_completion_messages_text_and_refusal() -> None:
    text_item = ResponseOutputText(text="hi", annotations=[], type="output_text")
    refusal_item = ResponseOutputRefusal(refusal="no", type="refusal")
    out_msg = ResponseOutputMessage(
        id="dummy",
        role="assistant",
        status="completed",
        type="message",
        content=[text_item, refusal_item],
    )
    resp = Response.model_construct(id="r", output=[out_msg], usage=None)
    msgs = convert_response_to_completion_messages(resp)
    assert msgs[0].content == "hi"
    assert msgs[1].refusal == "no"


def test_convert_response_to_completion_messages_function_call() -> None:
    fn = ResponseFunctionToolCall(
        call_id="cid",
        name="foo",
        arguments=json.dumps({"a": "b"}),
        type="function_call",
    )
    resp = Response.model_construct(id="r2", output=[fn], usage=None)
    msgs = convert_response_to_completion_messages(resp)
    assert msgs is not None
    assert msgs[0].tool_calls is not None
    assert msgs[0].tool_calls[0].id == "cid"


def test_convert_response_to_completion_messages_annotations() -> None:
    """URL citation annotations are mapped to completion annotations."""
    resp_ann = ResponsesAnnotationURLCitation(
        start_index=0, end_index=5, title="Example", url="http://example.com", type="url_citation"
    )
    text_item = ResponseOutputText(
        text="annotated",
        annotations=[resp_ann],
        type="output_text",
    )
    out_msg = ResponseOutputMessage(
        id="m1",
        role="assistant",
        status="completed",
        type="message",
        content=[text_item],
    )
    resp = Response.model_construct(id="r3", output=[out_msg], usage=None)
    msgs = convert_response_to_completion_messages(resp)
    ann_list = msgs[0].annotations
    assert ann_list is not None and len(ann_list) == 1
    ann = ann_list[0]
    assert isinstance(ann, CompletionAnnotation)
    assert ann.type == "url_citation"
    url_cit = ann.url_citation
    assert isinstance(url_cit, CompletionAnnotationURLCitation)
    assert url_cit.start_index == resp_ann.start_index
    assert url_cit.end_index == resp_ann.end_index
    assert url_cit.title == resp_ann.title
    assert url_cit.url == resp_ann.url


def test_convert_response_to_completion_messages_unsupported_items() -> None:
    """Unsupported response items are returned as JSON content messages."""
    resp_item = ResponseFileSearchToolCall.model_construct(
        call_id="file123", id="i1", type="file_search", status="completed", query="find file"
    )
    resp = Response.model_construct(id="r4", output=[resp_item], usage=None)
    msgs = convert_response_to_completion_messages(resp)
    assert len(msgs) == 1
    assert isinstance(msgs[0].content, str)
    assert json.loads(msgs[0].content) == resp_item.model_dump(exclude_none=True)


def test_convert_response_to_completion_messages_unsupported_web_search() -> None:
    """Web-search tool calls are returned as JSON content messages."""
    resp_item = ResponseFunctionWebSearch.model_construct(
        call_id="ws123", id="w1", type="web_search", status="completed", query="openai"
    )
    resp = Response.model_construct(id="r5", output=[resp_item], usage=None)
    msgs = convert_response_to_completion_messages(resp)
    assert len(msgs) == 1
    assert isinstance(msgs[0].content, str)
    assert json.loads(msgs[0].content) == resp_item.model_dump(exclude_none=True)


def test_convert_conversation_to_response_input_with_image_part() -> None:
    """User content with image parts converts to ResponseInputImageParam."""
    conv: TurnCompleter.RuntimeConversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "uimg",
                        "detail": "low",
                    },
                }
            ],
        }
    ]
    items = convert_conversation_to_response_input(conv)
    assert len(items) == 1
    item = items[0]
    assert item.get("role") == "user"
    assert item.get("type") == "message"
    content = item.get("content", [])
    assert isinstance(content, list)
    assert len(content) == 1
    img_item = content[0]
    assert isinstance(img_item, dict)
    assert img_item.get("type") == "input_image"
    assert img_item.get("image_url") == "uimg"


def test_assistant_string_content_only() -> None:
    """String content → single EasyInputMessageParam with correct fields."""
    # given
    msg: ChatCompletionAssistantMessageParam = {"role": "assistant", "content": "hello"}
    # when
    items = _assistant_completion_to_response_input_items(msg)
    # then
    assert len(items) == 1
    itm = items[0]
    assert itm.get("content") == "hello"
    assert itm.get("role") == "assistant"
    assert itm.get("type") == "message"


def test_assistant_string_content_with_refusal() -> None:
    """Top‑level refusal string adds a ResponseOutputMessageParam after the message."""
    # given
    msg: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": "ans",
        "refusal": "not allowed",
    }
    # when
    items = _assistant_completion_to_response_input_items(msg)
    # then
    assert len(items) == 2
    msg_itm, refusal_itm = items
    assert msg_itm.get("content") == "ans"
    assert msg_itm.get("role") == "assistant"
    assert refusal_itm.get("role") == "assistant"
    assert refusal_itm.get("type") == "message"
    content_list = refusal_itm.get("content")
    assert isinstance(content_list, list) and len(content_list) == 1
    ref_param = content_list[0]
    assert ref_param.get("refusal") == "not allowed"
    assert ref_param.get("type") == "refusal"


def test_assistant_string_content_with_tool_calls() -> None:
    """Top‑level tool_calls array adds corresponding ResponseFunctionToolCallParam(s)."""
    # given
    fname = "my_fn"
    call_id = "tcid"
    arguments = json.dumps({"x": 1})
    function: Function = {"name": fname, "arguments": arguments}
    tool_call: ChatCompletionMessageToolCallParam = {
        "id": call_id,
        "function": function,
        "type": "function",
    }
    msg: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": "foo",
        "tool_calls": [tool_call],
    }
    # when
    items = _assistant_completion_to_response_input_items(msg)
    # then
    assert len(items) == 2
    fn_itm = items[1]
    assert fn_itm.get("call_id") == call_id
    assert fn_itm.get("name") == fname
    assert fn_itm.get("arguments") == arguments
    assert fn_itm.get("type") == "function_call"


def test_assistant_array_content_all_text() -> None:
    """Content array of only text yields one EasyInputMessageParam per element."""
    # given
    content_array_list: Iterable[ContentArrayOfContentPart] = [
        {"type": "text", "text": "t1"},
        {"type": "text", "text": "t2"},
    ]
    msg: ChatCompletionAssistantMessageParam = {"role": "assistant", "content": content_array_list}
    # when
    items = _assistant_completion_to_response_input_items(msg)
    # then
    assert len(items) == 2
    texts = [itm.get("content") for itm in items]
    assert texts == ["t1", "t2"]
    for array, itm in zip(content_array_list, items):
        assert itm.get("role") == "assistant"
        assert itm.get("content") == array.get("text")


def test_assistant_array_content_mixed_text_and_refusal() -> None:
    """
    An assistant message containing both text and refusal content array parts.
    """
    # given
    content_array_list: Iterable[ContentArrayOfContentPart] = [
        {"type": "text", "text": "hi"},
        {"type": "refusal", "refusal": "stop"},
        {"type": "text", "text": "bye"},
    ]
    msg: ChatCompletionAssistantMessageParam = {"role": "assistant", "content": content_array_list}
    # when
    items = _assistant_completion_to_response_input_items(msg)
    # then
    assert len(items) == 3
    assert items[0].get("content") == "hi"
    assert items[2].get("content") == "bye"
    ref_item = items[1]
    assert ref_item.get("content")[0].get("refusal") == "stop"  # type: ignore


def test_assistant_string_content_with_refusal_and_tool_calls() -> None:
    """
    An assistant string message,
    accompanied by a top level refusal and tool calls
    """
    # given
    function: Function = {"name": "fn", "arguments": json.dumps({"a": "b"})}
    tool_call: ChatCompletionMessageToolCallParam = {
        "id": "tid",
        "function": function,
        "type": "function",
    }
    msg: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": "c",
        "refusal": "nope",
        "tool_calls": [tool_call],
    }
    # when
    items = _assistant_completion_to_response_input_items(msg)
    # then
    assert len(items) == 3
    assert items[0].get("content") == "c"
    assert items[1].get("content")[0].get("refusal") == "nope"  # type: ignore
    assert items[2].get("call_id") == "tid"


def test_assistant_array_with_top_level_refusal_and_tool_calls() -> None:
    """
    An assistant content array list message,
    accompanied by a top level refusal and tool calls
    """
    # given
    content_array_list: Iterable[ContentArrayOfContentPart] = [{"type": "text", "text": "x"}]
    function: Function = {"name": "f", "arguments": json.dumps({})}
    tc: ChatCompletionMessageToolCallParam = {
        "id": "fx",
        "function": function,
        "type": "function",
    }
    msg: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": content_array_list,
        "refusal": "r",
        "tool_calls": [tc],
    }
    # when
    items = _assistant_completion_to_response_input_items(msg)
    # then
    assert len(items) == 3
    assert items[0].get("content") == "x"
    assert items[1].get("content")[0].get("refusal") == "r"  # type: ignore
    assert items[2].get("call_id") == "fx"


def test_assistant_invalid_content_raises() -> None:
    """Non‑string, non‑array content should assert."""
    # given
    msg = {"role": "assistant", "content": 123}
    # then
    with pytest.raises(AssertionError):
        _assistant_completion_to_response_input_items(msg)  # type: ignore
