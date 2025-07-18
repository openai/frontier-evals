from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias, TypedDict, Unpack

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_assistant_message_param import Audio
from openai.types.chat.chat_completion_message_tool_call_param import Function
from pydantic import BaseModel


class TurnCompleter(ABC):
    encoding_name: str
    n_ctx: int

    class Config(BaseModel, ABC):
        @abstractmethod
        def build(self) -> TurnCompleter:
            raise NotImplementedError

    class Params(TypedDict, total=False):
        pass

    RuntimeConversation: TypeAlias = list[ChatCompletionMessageParam]

    class Completion(BaseModel):
        input_conversation: TurnCompleter.RuntimeConversation
        output_messages: list[ChatCompletionMessage]

        @property
        def output_conversation(self) -> TurnCompleter.RuntimeConversation:
            converted_output: list[ChatCompletionMessageParam] = [
                ChatCompletionAssistantMessageParam(
                    role=msg.role,
                    audio=Audio(id=msg.audio.id) if msg.audio else None,
                    content=msg.content,
                    refusal=msg.refusal,
                    tool_calls=[
                        ChatCompletionMessageToolCallParam(
                            id=tool_call.id,
                            function=Function(
                                arguments=tool_call.function.arguments,
                                name=tool_call.function.name,
                            ),
                            type="function",
                        )
                        for tool_call in msg.tool_calls
                    ]
                    if msg.tool_calls
                    else [],
                )
                for msg in self.output_messages
            ]
            return self.input_conversation + converted_output

    @abstractmethod
    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> TurnCompleter.Completion:
        """Generate and return a finished completion."""
        ...

    @abstractmethod
    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> TurnCompleter.Completion:
        """Generate and return a finished completion."""
        ...
