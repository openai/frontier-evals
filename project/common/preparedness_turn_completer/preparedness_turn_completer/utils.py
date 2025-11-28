from __future__ import annotations

import logging
import os
from typing import Unpack

import openai
import structlog.stdlib
import tenacity
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.type_helpers import (
    is_chat_completion_assistant_message_param,
    is_chat_completion_dev_message_param,
    is_chat_completion_function_message_param,
    is_chat_completion_sys_message_param,
    is_chat_completion_tool_message_param,
    is_chat_completion_user_message_param,
)
from pydantic import BaseModel

logger = structlog.stdlib.get_logger(component=__name__)

CONTEXT_WINDOW_LENGTHS: dict[str, int] = {
    "gpt-4o-mini": 128_000,
    "gpt-4o-mini-2024-07-18": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-2024-08-06": 128_000,
    "o1-mini": 128_000,
    "o1-mini-2024-09-12": 128_000,
    "o1": 200_000,
    "o1-2024-12-17": 200_000,
    "o3": 200_000,
    "o3-mini-2025-01-31": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "o4-mini-deep-research-2025-06-26": 200_000,
    "o4-mini-deep-research": 200_000,
    "o3-deep-research-2025-06-26": 200_000,
    "o3-deep-research": 200_000,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1": 1_047_576,
    "o1-preview": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "gpt-5-2025-08-07": 400_000,
    "gpt-5-mini-2025-08-07": 400_000,
    "gpt-5-nano-2025-08-07": 400_000,
    "gpt-5-codex": 400_000,
    "gpt-5-pro-2025-10-06": 400_000,
    "gpt-5-pro": 400_000,
}


def get_model_context_window_length(model: str) -> int:
    if model not in CONTEXT_WINDOW_LENGTHS:
        raise ValueError(f"Model {model} not found in context window lengths")
    return CONTEXT_WINDOW_LENGTHS[model]


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def is_image_download_timeout(exc: BaseException) -> bool:
    # retry only BadRequestErrors that mention a download timeout
    if isinstance(exc, openai.BadRequestError):
        return "Timeout while downloading" in str(exc)
    return False


retry_predicate = tenacity.retry_if_exception_type(
    OPENAI_TIMEOUT_EXCEPTIONS
) | tenacity.retry_if_exception(is_image_download_timeout)


class RetryConfig(BaseModel):
    wait_min: float = 1
    wait_max: float = 300
    stop_after: float = 3600 * 2

    def build(self: RetryConfig) -> tenacity.AsyncRetrying:
        return tenacity.AsyncRetrying(
            wait=tenacity.wait_random_exponential(min=self.wait_min, max=self.wait_max),
            stop=tenacity.stop_after_delay(self.stop_after),
            retry=retry_predicate,
            before_sleep=tenacity.before_sleep_log(logger._logger, logging.WARNING)
            if logger._logger
            else None,
            reraise=True,
        )


def text_from_completion(message: ChatCompletionMessageParam) -> str | list[str]:
    """
    Gets the text content from a chat completion message.
    If a particular message content(part) does not have text context, an empty string is
    returned for that message content(part).
    Useful for any truncation operations that require the text content of a message.
    """
    if "content" not in message or message["content"] is None:
        return ""
    elif (
        is_chat_completion_sys_message_param(message)
        or is_chat_completion_dev_message_param(message)
        or is_chat_completion_tool_message_param(message)
        or is_chat_completion_function_message_param(message)
    ):
        return (
            message["content"]
            if isinstance(message["content"], str)
            else [part["text"] for part in message["content"]]
        )
    elif is_chat_completion_user_message_param(message):
        return (
            message["content"]
            if isinstance(message["content"], str)
            else [part["text"] if part["type"] == "text" else "" for part in message["content"]]
        )
    elif is_chat_completion_assistant_message_param(message):
        return (
            message["content"]
            if isinstance(message["content"], str)
            else [
                part["text"] if part["type"] == "text" else part["refusal"]
                for part in message["content"]
            ]
        )
    else:
        raise ValueError(f"Unknown message role: {message['role']}")


def warn_about_non_empty_params(
    completer: TurnCompleter, **params: Unpack[TurnCompleter.Params]
) -> None:
    """
    We specifically don't want to use `TurnCompleter.Params` in `async_completion`
    because the base (non-abstract) `TurnCompleter.Params` is empty,
    and subclassing it will introduce conflicts or branching in the API.
    """
    if params and os.getenv("TC_DISABLE_EMPTY_PARAMS_WARNING", "false").lower() != "true":
        logger.warning(
            f"{completer.__class__} received params, but they are not used in async_completion."
            " You may disable this warning by setting the environment variable"
            " `TC_DISABLE_CONVERTER_WARNINGS` to `true`.",
            params=params,
        )
