from __future__ import annotations

import asyncio
import functools
import logging
from typing import Unpack

import openai
import structlog.stdlib
import tenacity
import tiktoken
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from preparedness_turn_completer.turn_completer import TurnCompleter

try:
    from nanoeval.recorder import get_recorder  # type: ignore
except ImportError:

    def get_recorder():
        raise LookupError("Recorder not available")


logger = structlog.stdlib.get_logger(component=__name__)


class OpenAITurnCompleter(TurnCompleter):
    def __init__(self, model: str, reasoning_effort: str | None = None):
        self.model: str = model
        self.reasoning_effort: str | None = reasoning_effort
        self.encoding_name: str = tiktoken.encoding_name_for_model(model)
        self.n_ctx: int = get_model_context_window_length(model)

    class Config(TurnCompleter.Config):
        model: str
        reasoning_effort: str | None = None

        def build(self) -> OpenAITurnCompleter:
            return OpenAITurnCompleter(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )

    class OpenAICompletion(TurnCompleter.Completion):
        usage: CompletionUsage | None = None

    @functools.cached_property
    def _client(self):
        return openai.AsyncClient()

    def _handle_kwargs(self, params: TurnCompleter.Params) -> TurnCompleter.Params:
        self._check_duplicate_param("model", params)
        self._check_duplicate_param("reasoning_effort", params)
        if "messages" in params:
            logger.warning("Found `messages` key in params, using `conversation` kwarg instead")
            del params["messages"]
        params["model"] = self.model
        params["reasoning_effort"] = self.reasoning_effort
        params = self._remove_none_values(params)
        return params

    def _remove_none_values(self, params: TurnCompleter.Params) -> TurnCompleter.Params:
        return {k: v for k, v in params.items() if v is not None}

    def _check_duplicate_param(
        self, param: str, params: TurnCompleter.Params
    ) -> TurnCompleter.Params:
        if param in params:
            logger.warning(
                f"Found duplicate key between self.{param} and params: {param} -> {params[param]}"
                f" using self.{param} value -> {getattr(self, param)}",
            )

    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> TurnCompleter.Completion:
        raise NotImplementedError("Not implemented, use async_completion instead")

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation | list[dict],
        **params: Unpack[TurnCompleter.Params],
    ) -> OpenAITurnCompleter.OpenAICompletion:
        # override params with self.completion_kwargs if duplicate keys
        # remove params[`messages`] since we have `conversation`
        kwargs = self._handle_kwargs(params)
        completion = await oai_chat_completion_create(
            self._client,
            messages=conversation,
            **kwargs,
        )
        return OpenAITurnCompleter.OpenAICompletion(
            input_conversation=conversation,
            output_messages=[completion.choices[0].message],
            usage=completion.usage,
        )


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=1, max=300),  # Max wait time of 5 minutes
    stop=tenacity.stop_after_delay(3600 * 2),  # Retry for up to 2 hours
    retry=tenacity.retry_if_exception_type(OPENAI_TIMEOUT_EXCEPTIONS),
    before_sleep=(
        tenacity.before_sleep_log(logger._logger, logging.WARNING) if logger._logger else None
    ),
    reraise=True,
)
async def oai_chat_completion_create(client: openai.AsyncClient, *args, **kwargs) -> ChatCompletion:
    # This is a bit of a hack. We replace "system" messages with
    # "developer" for models where "system" messages aren't supported.
    if "model" in kwargs and kwargs["model"] == "o1-redteam":
        new_messages = []

        for m in kwargs["messages"]:
            new_m = m
            if m["role"] == "system":
                new_m = {**m, "role": "developer"}
            new_messages.append(new_m)

        kwargs["messages"] = new_messages

    # TODO: this is messy - in most cases `parse` seems to be a superset of features in `create` so we use that,
    # but have found aisi-basic-agent with o1 crashing on this when we use `parse`. Something related to tool-use?
    try:
        res = await client.beta.chat.completions.parse(*args, **kwargs)
    except ValueError as e:
        # can't fallback to `create` if we're using a BaseModel for response_format
        response_format_kwarg = kwargs.get("response_format", None)
        if (
            response_format_kwarg is not None
            and isinstance(response_format_kwarg, type)
            and issubclass(response_format_kwarg, BaseModel)
        ):
            raise e
        res = await client.chat.completions.create(*args, **kwargs)

    try:
        # Attempt to record the sampling
        await asyncio.to_thread(
            get_recorder().record_sampling,
            prompt=kwargs["messages"],
            sampled=res.to_dict(),
        )
    except LookupError:
        # Recorder context variable is not set, skip recording
        pass

    return res


async def turn_completer_sample(
    solver: TurnCompleter, messages: list[ChatCompletionMessageParam] | list[dict]
) -> ChatCompletionMessage:
    response = await solver.async_completion(messages)
    return response.choices[0].message


def get_model_context_window_length(model: str | None) -> int:
    max_context_window_lengths: dict = {
        "gpt-4o-mini": 128000,
        "gpt-4o-mini-2024-07-18": 128000,
        "gpt-4o": 128000,
        "gpt-4o-2024-08-06": 128000,
        "o1-mini": 128000,
        "o1-mini-2024-09-12": 128000,
        "o1": 200000,
        "o1-2024-12-17": 200000,
        "o3-mini-2024-12-17": 128000,
        "o3-mini-2025-01-31": 200000,
        "o3-mini": 200000,
        "o1-preview": 128000,
        "gpt-4-turbo": 128000,
    }
    if model not in max_context_window_lengths:
        raise ValueError(f"Model {model} not found in context window lengths")
    return max_context_window_lengths[model]
