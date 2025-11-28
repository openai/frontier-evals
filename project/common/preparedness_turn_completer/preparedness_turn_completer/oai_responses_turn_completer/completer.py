from __future__ import annotations

import functools
from typing import Any, Literal, Unpack

import openai
import structlog
import tiktoken
from openai import NOT_GIVEN, NotGiven
from openai.types.responses import (
    Response,
    ResponseUsage,
)
from openai.types.responses.tool_param import ParseableToolParam
from openai.types.shared_params.reasoning import Reasoning
from preparedness_turn_completer.oai_responses_turn_completer.converters import (
    convert_conversation_to_response_input,
    convert_response_to_completion_messages,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import (
    RetryConfig,
    get_model_context_window_length,
    warn_about_non_empty_params,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = structlog.stdlib.get_logger(component=__name__)


class ReasoningConfig(BaseModel):
    """chz-friendly wrapper around openai.types.shared_params.reasoning:Reasoning"""

    effort: Literal["minimal", "low", "medium", "high"] | None = None
    generate_summary: Literal["auto", "concise", "detailed"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class OpenAIResponsesTurnCompleter(TurnCompleter):
    def __init__(
        self,
        model: str,
        reasoning: Reasoning | None | NotGiven = NOT_GIVEN,
        text_format: type[BaseModel] | NotGiven = NOT_GIVEN,
        tools: list[ParseableToolParam] | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_output_tokens: int | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        retry_config: RetryConfig | None = None,
    ):
        self.model = model
        self.reasoning = reasoning
        self.text_format = text_format
        self.tools = tools
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.encoding_name: str
        self.retry_config = retry_config or RetryConfig()
        try:
            self.encoding_name = tiktoken.encoding_name_for_model(model)
        except KeyError:
            # Fallback to o200k_base
            logger.warning(f"Model {model} not found in tiktoken, using o200k_base")
            self.encoding_name = "o200k_base"
        self.n_ctx: int = get_model_context_window_length(model)

    class Config(TurnCompleter.Config):
        """
        Responses configuration. Non-exhaustive.
        Add more configuration options as needed, in a backwards-compatible way.
        """

        # needed for NotGiven type hint
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_encoders={NotGiven: lambda v: "NOT_GIVEN"},
        )

        model: str
        reasoning: ReasoningConfig | None | NotGiven = NOT_GIVEN
        text_format: type[BaseModel] | NotGiven = NOT_GIVEN
        tools: list[ParseableToolParam] | NotGiven = NOT_GIVEN
        temperature: float | None | NotGiven = NOT_GIVEN
        max_output_tokens: int | None | NotGiven = NOT_GIVEN
        top_p: float | None | NotGiven = NOT_GIVEN
        retry_config: RetryConfig = Field(default_factory=RetryConfig)

        def build(self) -> OpenAIResponsesTurnCompleter:
            reasoning_param: Reasoning | None | NotGiven
            if isinstance(self.reasoning, ReasoningConfig):
                reasoning_param = Reasoning(
                    effort=self.reasoning.effort,
                    generate_summary=self.reasoning.generate_summary,
                    summary=self.reasoning.summary,
                )
            else:
                reasoning_param = self.reasoning

            return OpenAIResponsesTurnCompleter(
                model=self.model,
                reasoning=reasoning_param,
                text_format=self.text_format,
                tools=self.tools,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                top_p=self.top_p,
                retry_config=self.retry_config,
            )

        @field_validator("*", mode="before")
        @classmethod
        def _decode_not_given(cls: type[OpenAIResponsesTurnCompleter.Config], v: Any) -> Any:
            """
            Turn the string "NOT_GIVEN" back into our sentinel before validation.
            """
            if v == "NOT_GIVEN":
                return NOT_GIVEN
            return v

    class Completion(TurnCompleter.Completion):
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
        **params: Unpack[TurnCompleter.Params],
    ) -> OpenAIResponsesTurnCompleter.Completion:
        warn_about_non_empty_params(self, **params)

        conversation_input = convert_conversation_to_response_input(conversation)

        async for attempt in self.retry_config.build():
            with attempt:
                response: Response = await self._client.responses.parse(
                    input=conversation_input,
                    model=self.model,
                    reasoning=self.reasoning,
                    text_format=self.text_format,
                    tools=self.tools,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    top_p=self.top_p,
                )
        completion_messages = convert_response_to_completion_messages(response)

        return OpenAIResponsesTurnCompleter.Completion(
            input_conversation=conversation,
            output_messages=completion_messages,
            usage=response.usage,
        )
