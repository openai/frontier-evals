import json
import uuid
from typing import Literal

import structlog.stdlib
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message import Annotation as CompletionAnnotation
from openai.types.chat.chat_completion_message import (
    AnnotationURLCitation as CompletionAnnotationURLCitation,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseCodeInterpreterToolCall,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseFunctionWebSearch,
    ResponseInputContentParam,
    ResponseInputFileParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputMessage,
    ResponseOutputMessageParam,
    ResponseOutputRefusal,
    ResponseOutputRefusalParam,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_item import (
    ImageGenerationCall,
    LocalShellCall,
    McpCall,
    McpListTools,
)
from openai.types.responses.response_output_item import McpApprovalRequest
from openai.types.responses.response_output_message import Content
from openai.types.responses.response_output_text import Annotation as ResponseAnnotation
from openai.types.responses.response_output_text import (
    AnnotationURLCitation as ResponsesAnnotationURLCitation,
)
from preparedness_turn_completer.oai_responses_turn_completer.type_helpers import (
    ChatCompletionContent,
    is_assistant_message,
    is_content_part_list,
    is_text_parts_list,
    is_tool_message,
)
from preparedness_turn_completer.turn_completer import TurnCompleter

logger = structlog.stdlib.get_logger(component=__name__)


# ----------------- ChatCompletionMessageParam to ResponseInputParam


def convert_conversation_to_response_input(
    conversation: TurnCompleter.RuntimeConversation,
) -> ResponseInputParam:
    response_input_items: ResponseInputParam = [
        item
        for message in conversation
        for item in _chat_completion_message_to_response_input_items(message)
    ]
    return response_input_items


def _chat_completion_message_to_response_input_items(
    message: ChatCompletionMessageParam,
) -> list[ResponseInputItemParam]:
    role = message["role"]
    content = message["content"]

    if role == "developer" or role == "system":
        return _dev_or_sys_completion_to_response_input_items(role, content)
    elif role == "user":
        return _user_completion_to_response_input_items(content)
    elif role == "assistant":
        return _assistant_completion_to_response_input_items(message)
    elif role == "tool":
        return _tool_completion_to_response_input_items(message)
    elif role == "function":
        raise NotImplementedError(
            "Converting `role='function'` messages is not supported, as it is deprecated."
            " Please use a `role='tool'` message instead."
        )
    else:
        raise ValueError(f"Unknown role: {role}")


def _dev_or_sys_completion_to_response_input_items(
    role: Literal["developer", "system"],
    content: ChatCompletionContent,
) -> list[ResponseInputItemParam]:
    assert isinstance(content, str) or is_text_parts_list(content), (
        f"Expected content to be str or text‐parts, got {content!r}"
    )
    return [
        EasyInputMessageParam(
            content=_chat_completion_text_to_response_input_item(content),
            role=role,
            type="message",
        )
    ]


def _user_completion_to_response_input_items(
    content: ChatCompletionContent,
) -> list[ResponseInputItemParam]:
    assert isinstance(content, str) or is_content_part_list(content), (
        f"Expected content to be str or content‐parts, got {content!r}"
    )
    return [
        EasyInputMessageParam(
            content=_chat_completion_content_to_response_input_content(content),
            role="user",
            type="message",
        )
    ]


def _assistant_completion_to_response_input_items(
    message: ChatCompletionMessageParam,
) -> list[ResponseInputItemParam]:
    content = message["content"]
    role: Literal["assistant"] = "assistant"
    input_items: list[ResponseInputItemParam] = []
    assert isinstance(content, str) or is_content_part_list(content), (
        f"Expected content to be str or text‐parts, got {content!r}"
    )
    assert is_assistant_message(message)
    input_items.append(
        EasyInputMessageParam(
            content=_chat_completion_content_to_response_input_content(content),
            role=role,
            type="message",
        )
    )
    refusal = message.get("refusal", None)
    if isinstance(refusal, str):
        input_items.append(
            ResponseOutputMessageParam(
                content=[ResponseOutputRefusalParam(refusal=refusal, type="refusal")],
                id=uuid.uuid4().hex,
                role=role,
                status="completed",
                type="message",
            )
        )
    if "tool_calls" in message:
        tool_call_input_items = [
            ResponseFunctionToolCallParam(
                call_id=tool_call["id"],
                name=tool_call["function"]["name"],
                type="function_call",
                arguments=tool_call["function"]["arguments"],
            )
            for tool_call in message["tool_calls"]
        ]
        input_items.extend(tool_call_input_items)
    return input_items


def _tool_completion_to_response_input_items(
    message: ChatCompletionMessageParam,
) -> list[ResponseInputItemParam]:
    assert is_tool_message(message)
    content = message["content"]
    output_str: str
    if isinstance(content, str):
        output_str = content
    else:
        output_str = "".join(part["text"] for part in content)

    return [
        FunctionCallOutput(
            call_id=message["tool_call_id"],
            output=output_str,
            type="function_call_output",
            id=uuid.uuid4().hex,
            status="completed",
        )
    ]


def _chat_completion_text_to_response_input_item(
    content: ChatCompletionContent,
) -> str | ResponseInputMessageContentListParam:
    if isinstance(content, str):
        return content
    elif is_text_parts_list(content):
        return [ResponseInputTextParam(text=part["text"], type="input_text") for part in content]
    else:
        raise ValueError(f"Expected content to be str or text‐parts, got {content!r}")


def _chat_completion_content_to_response_input_content(
    content: ChatCompletionContent,
) -> str | ResponseInputMessageContentListParam:
    if isinstance(content, str):
        return content
    elif is_content_part_list(content):
        if any(part["type"] == "input_audio" for part in content):
            raise NotImplementedError("Audio content is not supported in ResponsesAPI")
        return [_chat_completion_part_to_response_part(part) for part in content]
    else:
        raise ValueError(f"Expected content to be str or content‐parts, got {content!r}")


def _chat_completion_part_to_response_part(
    part: ChatCompletionContentPartParam,
) -> ResponseInputContentParam:
    if part["type"] == "input_audio":
        raise NotImplementedError("Audio content is not supported in ResponsesAPI")
    elif part["type"] == "text":
        return ResponseInputTextParam(text=part["text"], type="input_text")
    elif part["type"] == "image_url":
        return ResponseInputImageParam(
            image_url=part["image_url"]["url"],
            type="input_image",
            detail=part["image_url"]["detail"],
        )
    elif part["type"] == "file":
        return ResponseInputFileParam(
            file_data=part["file"]["file_data"],
            file_id=part["file"]["file_id"],
            filename=part["file"]["filename"],
            type="input_file",
        )
    else:
        raise ValueError(f"Unknown content part type: {part['type']}")


# ----------------- Response to ChatCompletionMessages


def convert_response_to_completion_messages(response: Response) -> list[ChatCompletionMessage]:
    completion_messages: list[ChatCompletionMessage] = []

    for response_item in response.output:
        if isinstance(response_item, ResponseOutputMessage):
            completion_messages.extend(
                _response_output_message_to_chat_completion_messages(response_item)
            )
        elif isinstance(response_item, ResponseFunctionToolCall):
            completion_messages.append(
                _response_function_tool_call_to_chat_completion_message(response_item)
            )
        elif isinstance(response_item, ResponseFunctionWebSearch):
            # We might want to just return the JSON representation of this data (as we were before)
            # instead of returning a message marked with "<| Web Search tool call: 'query' |>"
            # In general, we may want to extend the TurnCompleter.Completion class to include a
            # general tool outputs field to handle such cases.
            completion_messages.append(
                _response_function_web_search_to_chat_completion_message(response_item)
            )
        elif isinstance(response_item, ResponseReasoningItem):
            # Similar to above, we may want to extend the TurnCompleter.Completion class to include a field for reasoning strings in the case of ResponseReasoningItems.
            completion_messages.append(
                _response_reasoning_item_to_chat_completion_message(response_item)
            )
        elif isinstance(
            response_item,
            (
                ResponseFileSearchToolCall,
                ResponseComputerToolCall,
                ImageGenerationCall,
                ResponseCodeInterpreterToolCall,
                LocalShellCall,
                McpCall,
                McpListTools,
                McpApprovalRequest,
            ),
        ):
            completion_messages.append(
                _unsupported_response_to_chat_completion_message(response_item)
            )
        else:
            raise ValueError(f"Unknown response item type: {type(response_item)}")

    return completion_messages


def _response_output_message_to_chat_completion_messages(
    response_item: ResponseOutputMessage,
) -> list[ChatCompletionMessage]:
    return [
        _response_output_message_content_to_chat_completion_message(content)
        for content in response_item.content
    ]


def _response_output_message_content_to_chat_completion_message(
    content: Content,
) -> ChatCompletionMessage:
    if isinstance(content, ResponseOutputText):
        return ChatCompletionMessage(
            content=content.text,
            role="assistant",
            annotations=_response_annotations_to_completion_annotations(content.annotations)
            if content.annotations
            else None,
        )
    elif isinstance(content, ResponseOutputRefusal):
        return ChatCompletionMessage(refusal=content.refusal, role="assistant")
    else:
        raise ValueError(f"Unknown content type: {type(content)}")


def _response_annotations_to_completion_annotations(
    response_annotations: list[ResponseAnnotation],
) -> list[CompletionAnnotation]:
    completion_annotations = []
    for annotation in response_annotations:
        if not isinstance(annotation, ResponsesAnnotationURLCitation):
            logger.warning(
                f"Only Responses AnnotationURLCitation can be ported to Completions API, got {annotation!r}. Skipping"
            )
            continue
        completion_annotations.append(
            CompletionAnnotation(
                type="url_citation",
                url_citation=CompletionAnnotationURLCitation(
                    end_index=annotation.end_index,
                    start_index=annotation.start_index,
                    title=annotation.title,
                    url=annotation.url,
                ),
            )
        )
    return completion_annotations


def _response_function_tool_call_to_chat_completion_message(
    response_item: ResponseFunctionToolCall,
) -> ChatCompletionMessage:
    return ChatCompletionMessage(
        role="assistant",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id=response_item.call_id,
                type="function",
                function=Function(name=response_item.name, arguments=response_item.arguments),
            ),
        ],
    )


def _response_function_web_search_to_chat_completion_message(
    response_item: ResponseFunctionWebSearch,
) -> ChatCompletionMessage:
    if hasattr(response_item, "action") and response_item.action:
        if hasattr(response_item.action, "query"):
            search_query = response_item.action.query
        elif isinstance(response_item.action, dict):
            search_query = response_item.action.get("query", "unknown query")
        else:
            search_query = "unknown query"
    elif hasattr(response_item, "query"):
        search_query = response_item.query
    else:
        search_query = "unknown query"

    placeholder_content = f"<| Web Search tool call: '{search_query}' |>"

    return ChatCompletionMessage(
        role="assistant",
        content=placeholder_content,
    )


def _response_reasoning_item_to_chat_completion_message(
    response_item: ResponseReasoningItem,
) -> ChatCompletionMessage:
    reasoning_content = "\n".join([summary.text for summary in response_item.summary])

    return ChatCompletionMessage(
        role="assistant",
        content=reasoning_content,
    )


def _unsupported_response_to_chat_completion_message(
    response_item: ResponseFileSearchToolCall
    | ResponseComputerToolCall
    | ImageGenerationCall
    | ResponseCodeInterpreterToolCall
    | LocalShellCall
    | McpCall
    | McpListTools
    | McpApprovalRequest,
) -> ChatCompletionMessage:
    logger.warning(
        f"Response of type {type(response_item)} is not natively supported in ChatCompletionMessage."
        " Returning a JSON string representation of the response item in `content`."
    )
    return ChatCompletionMessage(
        content=json.dumps(response_item.model_dump(exclude_none=True)), role="assistant"
    )
