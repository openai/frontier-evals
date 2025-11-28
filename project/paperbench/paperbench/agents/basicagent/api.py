import json
from dataclasses import dataclass
from typing import Any, TypeGuard

import structlog
from openai import (
    BadRequestError,
    LengthFinishReasonError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.type_helpers import (
    chat_completion_message_to_message_param,
    is_chat_completion_message_param,
)

from paperbench.agents.basicagent.completer import TimeTrackingRetryConfig
from paperbench.agents.basicagent.tools.base import ToolCall

logger = structlog.stdlib.get_logger(component=__name__)


def parse_basic_agent_tool_calls(completion: list[ChatCompletionMessage]) -> list[ToolCall]:
    """
    Parses completion messages into BasicAgent ToolCalls
    """
    basic_agent_tool_calls: list[ToolCall] = []

    for item in completion:
        if item.tool_calls:
            for tool_call in item.tool_calls:
                assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
                basic_agent_tool_calls.append(
                    ToolCall(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                        tool_call.id,
                    )
                )

    return basic_agent_tool_calls


@dataclass(frozen=True)
class CompleterResponse:
    response_messages: list[ChatCompletionMessageParam]
    tool_calls: list[ToolCall]
    time_spent_retrying: float


async def make_completer_request(
    completer: TurnCompleter, messages: list[ChatCompletionMessageParam]
) -> CompleterResponse:
    """
    Makes request using the provided completer

    Returns:
        response_messages (list[dict[str, Any]]): list of generated messages
        tool_calls (list[ToolCall]): list of requested BasicAgent tool calls
        time_spent_retrying (float): total time spent retrying, not counting time to generate the response
    """
    time_spent_retrying = 0.0

    try:
        completion = await completer.async_completion(conversation=messages)
    except BadRequestError as e:
        if "input exceeds the context window" in e.message:
            raise LengthFinishReasonError(
                completion=ChatCompletion(
                    id="",
                    choices=[],
                    created=0,
                    model="",
                    object="chat.completion",
                )
            ) from e
        else:
            raise e

    if hasattr(completer, "retry_config") and isinstance(
        completer.retry_config, TimeTrackingRetryConfig
    ):
        time_spent_retrying = completer.retry_config.time_spent_retrying
        completer.retry_config.time_spent_retrying = 0.0

    tool_calls = parse_basic_agent_tool_calls(completion.output_messages)

    output_messages = [
        chat_completion_message_to_message_param(message) for message in completion.output_messages
    ]
    assert is_chat_completion_message_param_list(output_messages)

    return CompleterResponse(
        response_messages=output_messages,
        tool_calls=tool_calls,
        time_spent_retrying=time_spent_retrying,
    )


def is_chat_completion_message_param_list(obj: Any) -> TypeGuard[list[ChatCompletionMessageParam]]:
    if not isinstance(obj, list):
        return False
    return all(is_chat_completion_message_param(item) for item in obj)
