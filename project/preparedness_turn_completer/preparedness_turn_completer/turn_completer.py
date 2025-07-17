from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias, TypedDict, Unpack

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
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
            return self.input_conversation.with_suffix(*self.output_messages)

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
