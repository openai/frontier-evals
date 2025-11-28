from typing import Any, Iterable, TypeAlias, TypeGuard

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageCustomToolCallParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnionParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    Audio,
    ContentArrayOfContentPart,
)
from openai.types.chat.chat_completion_message_custom_tool_call_param import Custom
from openai.types.chat.chat_completion_message_tool_call_param import Function

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


def is_chat_completion_assistant_message_param(
    obj: Any,
) -> TypeGuard[ChatCompletionAssistantMessageParam]:
    return (
        isinstance(obj, dict)
        and (
            {"role"}
            <= obj.keys()
            <= {"role", "audio", "content", "function_call", "name", "refusal", "tool_calls"}
        )
        and obj["role"] == "assistant"
        and (
            isinstance(obj.get("content"), str)
            or is_content_array_list(obj.get("content"))
            or (
                obj.get("content") is None
                and is_chat_completion_message_tool_call_union_list(obj.get("tool_calls"))
            )
        )
    )


def is_chat_completion_tool_message_param(obj: Any) -> TypeGuard[ChatCompletionToolMessageParam]:
    return (
        isinstance(obj, dict)
        and obj.keys() == {"role", "content", "tool_call_id"}
        and obj["role"] == "tool"
        and isinstance(obj["tool_call_id"], str)
        and (isinstance(obj["content"], str) or is_text_parts_list(obj["content"]))
    )


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


def is_chat_completion_message_tool_call_union_list(
    obj: Any,
) -> TypeGuard[Iterable[ChatCompletionMessageToolCallUnionParam]]:
    try:
        iterator = iter(obj)
    except TypeError:
        return False
    return all(is_chat_completion_tool_call_union_param(item) for item in iterator)


def is_chat_completion_tool_call_union_param(
    obj: Any,
) -> TypeGuard[ChatCompletionMessageToolCallUnionParam]:
    return is_function_tool_call_param(obj) or is_custom_tool_call_param(obj)


def is_chat_completion_dev_message_param(
    obj: Any,
) -> TypeGuard[ChatCompletionDeveloperMessageParam]:
    return (
        isinstance(obj, dict)
        and ({"role", "content"} <= obj.keys() <= {"role", "content", "name"})
        and obj["role"] == "developer"
        and (isinstance(obj["content"], str) or is_text_parts_list(obj["content"]))
    )


def is_chat_completion_sys_message_param(obj: Any) -> TypeGuard[ChatCompletionSystemMessageParam]:
    return (
        isinstance(obj, dict)
        and ({"role", "content"} <= obj.keys() <= {"role", "content", "name"})
        and obj["role"] == "system"
        and (isinstance(obj["content"], str) or is_text_parts_list(obj["content"]))
    )


def is_chat_completion_user_message_param(obj: Any) -> TypeGuard[ChatCompletionUserMessageParam]:
    return (
        isinstance(obj, dict)
        and ({"role", "content"} <= obj.keys() <= {"role", "content", "name"})
        and obj["role"] == "user"
        and (isinstance(obj["content"], str) or is_content_part_list(obj["content"]))
    )


def is_chat_completion_function_message_param(
    obj: Any,
) -> TypeGuard[ChatCompletionFunctionMessageParam]:
    return (
        isinstance(obj, dict)
        and obj.keys() == {"content", "name", "role"}
        and obj["role"] == "function"
        and isinstance(obj["name"], str)
        and isinstance(obj["content"], str)
    )


def is_chat_completion_message_param(obj: Any) -> TypeGuard[ChatCompletionMessageParam]:
    return (
        is_chat_completion_assistant_message_param(obj)
        or is_chat_completion_user_message_param(obj)
        or is_chat_completion_sys_message_param(obj)
        or is_chat_completion_dev_message_param(obj)
        or is_chat_completion_tool_message_param(obj)
        or is_chat_completion_function_message_param(obj)
    )


def chat_completion_message_to_message_param(
    message: ChatCompletionMessage,
) -> ChatCompletionAssistantMessageParam:
    """
    Convert a response-side assistant message (ChatCompletionMessage)
    into a request-side assistant message param (ChatCompletionAssistantMessageParam),
    preserving content and tool/function calls.
    """

    # ChatCompletionMessage role is always "assistant"
    message_param: ChatCompletionAssistantMessageParam = {"role": "assistant"}

    message_param["audio"] = Audio(id=message.audio.id) if message.audio else None

    message_param["content"] = message.content

    # message_param['function'] is deprecated, so we won't handle it
    # message_param['name'] is optional, and is not present on ChatCompletionMessage

    message_param["refusal"] = message.refusal

    if message.tool_calls:
        message_param["tool_calls"] = [
            ChatCompletionMessageFunctionToolCallParam(
                id=tc.id,
                function=Function(arguments=tc.function.arguments, name=tc.function.name),
                type=tc.type,
            )
            if tc.type == "function"
            else ChatCompletionMessageCustomToolCallParam(
                id=tc.id,
                custom=Custom(input=tc.custom.input, name=tc.custom.name),
                type=tc.type,
            )
            for tc in message.tool_calls
        ]

    return message_param
