from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Iterable, Unpack

import openai
import structlog
import tenacity
import tiktoken
from openai.types.responses import (
    Response,
    ResponseUsage,
)
from openai.types.responses.tool_param import ParseableToolParam
from openai.types.shared.reasoning import Reasoning
from preparedness_turn_completer.oai_responses_turn_completer.converters import (
    convert_conversation_to_response_input,
    convert_response_to_completion_messages,
)
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


class OpenAIResponsesTurnCompleter(TurnCompleter):
    def __init__(self, model: str, reasoning: Reasoning | None = None):
        self.model: str = model
        self.reasoning: Reasoning | None = reasoning
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
        reasoning: Reasoning | None = None

        def build(self) -> OpenAIResponsesTurnCompleter:
            return OpenAIResponsesTurnCompleter(model=self.model, reasoning=self.reasoning)

    class Params(TurnCompleter.Params, total=False):
        text_format: type[BaseModel] | None
        tools: Iterable[ParseableToolParam] | None
        previous_response_id: str | None
        temperature: float | None

    class Completion(TurnCompleter.Completion):
        response_id: str
        usage: ResponseUsage | None = None

    @functools.cached_property
    def _client(self) -> openai.AsyncClient:
        return openai.AsyncClient()

    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> OpenAIResponsesTurnCompleter.Completion:
        raise NotImplementedError("Not implemented, use async_completion instead")

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[OpenAIResponsesTurnCompleter.Params],
    ) -> OpenAIResponsesTurnCompleter.Completion:
        conversation_input = convert_conversation_to_response_input(conversation)
        response: Response = await oai_create(
            client=self._client,
            model=self.model,
            reasoning=self.reasoning,
            input=conversation_input,
            **params,
        )
        completion_messages = convert_response_to_completion_messages(response)

        return OpenAIResponsesTurnCompleter.Completion(
            input_conversation=conversation,
            output_messages=completion_messages,
            response_id=response.id,
            usage=response.usage,
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
async def oai_create(client: openai.AsyncClient, *args: Any, **kwargs: Any) -> Response:
    res = await client.responses.parse(*args, **kwargs)

    try:
        # Attempt to record the sampling
        await asyncio.to_thread(
            get_recorder().record_sampling,
            prompt=kwargs["input"],
            sampled=res.to_dict(),
        )
    except LookupError:
        # Recorder context variable is not set, skip recording
        pass

    return res
