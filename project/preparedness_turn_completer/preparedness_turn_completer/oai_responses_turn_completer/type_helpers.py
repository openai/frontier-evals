from typing import Any, Iterable, TypeAlias, TypeGuard

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageCustomToolCallParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import ContentArrayOfContentPart

ChatCompletionContent: TypeAlias = (
    str
    | Iterable[ChatCompletionContentPartTextParam]
    | Iterable[ChatCompletionContentPartParam]
    | Iterable[ContentArrayOfContentPart]
    | None
)


def is_text_part(obj: Any) -> TypeGuard[ChatCompletionContentPartTextParam]:
    if not isinstance(obj, dict):
        return False
    return "text" in obj and isinstance(obj["text"], str) and obj.get("type") == "text"


def is_text_parts_list(obj: Any) -> TypeGuard[Iterable[ChatCompletionContentPartTextParam]]:
    if isinstance(obj, str):
        return False
    try:
        iterator = iter(obj)
    except TypeError:
        return False
    return all(is_text_part(item) for item in iterator)


def is_content_part(obj: Any) -> TypeGuard[ChatCompletionContentPartParam]:
    if not isinstance(obj, dict):
        return False
    return "type" in obj and (
        (obj["type"] == "text" and "text" in obj)
        or (obj["type"] == "file" and "file" in obj)
        or (obj["type"] == "image_url" and "image_url" in obj)
        or (obj["type"] == "input_audio" and "input_audio" in obj)
    )


def is_content_part_list(obj: Any) -> TypeGuard[Iterable[ChatCompletionContentPartParam]]:
    if isinstance(obj, str):
        return False
    try:
        iterator = iter(obj)
    except TypeError:
        return False
    return all(is_content_part(item) for item in iterator)


def is_content_array(obj: Any) -> TypeGuard[ContentArrayOfContentPart]:
    if not isinstance(obj, dict):
        return False
    return "type" in obj and (
        (obj["type"] == "text" and "text" in obj) or (obj["type"] == "refusal" and "refusal" in obj)
    )


def is_content_array_list(obj: Any) -> TypeGuard[Iterable[ContentArrayOfContentPart]]:
    if isinstance(obj, str):
        return False
    try:
        iterator = iter(obj)
    except TypeError:
        return False
    return all(is_content_array(item) for item in iterator)


def is_assistant_message(obj: Any) -> TypeGuard[ChatCompletionAssistantMessageParam]:
    if not isinstance(obj, dict):
        return False
    return obj.get("role") == "assistant"


def is_tool_message(obj: Any) -> TypeGuard[ChatCompletionToolMessageParam]:
    if not isinstance(obj, dict):
        return False
    return obj.get("role") == "tool" and "tool_call_id" in obj


def is_function_tool_call_param(obj: Any) -> TypeGuard[ChatCompletionMessageFunctionToolCallParam]:
    if not isinstance(obj, dict):
        return False
    return (
        obj.get("type") == "function"
        and "id" in obj
        and "function" in obj
        and isinstance(obj["function"], dict)
        and "arguments" in obj["function"]
        and "name" in obj["function"]
    )


def is_custom_tool_call_param(obj: Any) -> TypeGuard[ChatCompletionMessageCustomToolCallParam]:
    if not isinstance(obj, dict):
        return False
    return (
        obj.get("type") == "custom"
        and "id" in obj
        and "custom" in obj
        and isinstance(obj["custom"], dict)
        and "input" in obj["custom"]
        and "name" in obj["custom"]
    )
