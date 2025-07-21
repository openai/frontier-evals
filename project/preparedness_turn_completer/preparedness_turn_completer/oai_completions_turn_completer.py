from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Unpack

import openai
import structlog
import tenacity
import tiktoken
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import get_model_context_window_length
from pydantic import BaseModel

try:
    from nanoeval.recorder import get_recorder
except ImportError:
    from nanoeval.recorder_protocol import RecorderProtocol

    def get_recorder() -> RecorderProtocol:
        raise LookupError("Recorder not available")


logger = structlog.stdlib.get_logger(component=__name__)


class OpenAICompletionsTurnCompleter(TurnCompleter):
    def __init__(self, model: str, reasoning_effort: str | None = None):
        self.model: str = model
        self.reasoning_effort: str | None = reasoning_effort
        self.encoding_name: str
        try:
            self.encoding_name = tiktoken.encoding_name_for_model(model)
        except KeyError:
            # Fallback to o200k_base
            logger.warning(f"Model {model} not found in tiktoken, using o200k_base")
            self.encoding_name = "o200k_base"
        self.n_ctx: int = get_model_context_window_length(model)

    class Config(TurnCompleter.Config):
        model: str
        reasoning_effort: str | None = None

        def build(self) -> OpenAICompletionsTurnCompleter:
            return OpenAICompletionsTurnCompleter(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )

    class Params(TurnCompleter.Params, total=False):
        response_format: type[BaseModel] | None
        temperature: float | None

    class Completion(TurnCompleter.Completion):
        usage: CompletionUsage | None = None

    @functools.cached_property
    def _client(self) -> openai.AsyncClient:
        return openai.AsyncClient()

    def _handle_kwargs(
        self,
        params: OpenAICompletionsTurnCompleter.Params,
        conversation: TurnCompleter.RuntimeConversation,
    ) -> dict[str, Any]:
        if "messages" in params:
            logger.warning("Found `messages` key in params, will use `conversation` kwarg instead")
        if "reasoning_effort" in params:
            logger.warning(
                "Found `reasoning_effort` key in params,"
                f" will use self.reasoning_effort={self.reasoning_effort} instead"
            )
        if "model" in params:
            logger.warning(f"Found `model` key in params, will use self.model={self.model} instead")
        merged_kwargs = {
            **params,
            "model": self.model,
            "messages": conversation,
        }
        if self.reasoning_effort is not None:
            merged_kwargs["reasoning_effort"] = self.reasoning_effort
        return merged_kwargs

    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[OpenAICompletionsTurnCompleter.Params],
    ) -> OpenAICompletionsTurnCompleter.Completion:
        raise NotImplementedError("Not implemented, use async_completion instead")

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[OpenAICompletionsTurnCompleter.Params],
    ) -> OpenAICompletionsTurnCompleter.Completion:
        # possibly override params such as `model`, `reasoning_effort`, and `messages`
        kwargs = self._handle_kwargs(params, conversation)
        completion = await oai_create(
            self._client,
            **kwargs,
        )
        assert isinstance(completion, ChatCompletion)
        return OpenAICompletionsTurnCompleter.Completion(
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


# TODO: make this configurable
@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=1, max=300),  # Max wait time of 5 minutes
    stop=tenacity.stop_after_delay(3600 * 2),  # Retry for up to 2 hours
    retry=tenacity.retry_if_exception_type(OPENAI_TIMEOUT_EXCEPTIONS),
    before_sleep=(
        tenacity.before_sleep_log(logger._logger, logging.WARNING) if logger._logger else None
    ),
    reraise=True,
)
async def oai_create(client: openai.AsyncClient, *args: Any, **kwargs: Any) -> ChatCompletion:
    # TODO: this is messy - in most cases `parse` seems to be a superset of features in `create` so we use that,
    # but have found basic-agent with o1 crashing on this when we use `parse`. Something related to tool-use?
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
