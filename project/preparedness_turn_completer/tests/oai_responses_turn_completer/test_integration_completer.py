import asyncio
import json
import os

import pytest
from preparedness_turn_completer.oai_responses_turn_completer.completer import (
    OpenAIResponsesTurnCompleter,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from pydantic import BaseModel, Field

pytestmark = pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="requires OPENAI_API_KEY",
)

# pick models that support the needed features
MODELS_JSON = ["gpt-4o-mini"]
MODELS_BASE = ["gpt-4.1-nano"]


@pytest.mark.parametrize("model", MODELS_BASE)
def test_async_completion_integration(model: str) -> None:
    # given
    completer = OpenAIResponsesTurnCompleter(model=model)
    conversation: TurnCompleter.RuntimeConversation = [{"role": "user", "content": "Hello"}]
    # when
    completion: OpenAIResponsesTurnCompleter.Completion = asyncio.run(
        completer.async_completion(conversation)
    )
    # then
    assert completion.output_messages
    assert completion.usage is not None
    assert completion.output_messages[0].role == "assistant"


class FooBarSchema(BaseModel):
    foo: str = Field(..., description="A simple string field")
    bar: int = Field(..., description="A non‐negative integer", ge=0)


@pytest.mark.parametrize("model", MODELS_JSON)
def test_structured_output(model: str) -> None:
    """
    Ask via the completer to return exactly {"foo": string, "bar": integer}
    by passing a Pydantic BaseModel directly as text_format. The SDK
    will convert it to a strict JSON Schema under the hood. :contentReference[oaicite:0]{index=0}
    """
    # given
    completer = OpenAIResponsesTurnCompleter(model=model, text_format=FooBarSchema)
    conversation: TurnCompleter.RuntimeConversation = [
        {"role": "user", "content": "Please reply with a valid object"}
    ]

    # when
    completion = asyncio.run(completer.async_completion(conversation))

    # then
    assert completion.output_messages, "no output messages"
    text = completion.output_messages[0].content
    assert isinstance(text, str) and text.strip(), "empty or non-string content"
    data = json.loads(text)
    assert set(data.keys()) == {"foo", "bar"}
    assert isinstance(data["foo"], str)
    assert isinstance(data["bar"], int)
    assert data["bar"] >= 0


@pytest.mark.parametrize("model", MODELS_JSON)
def test_tool_web_search_passes_through(model: str) -> None:
    """
    Ensure that passing a web_search tool doesn't break the abstraction
    and still returns an assistant message.
    """
    # given
    completer = OpenAIResponsesTurnCompleter(
        model=model, tools=[{"type": "web_search_preview", "search_context_size": "low"}]
    )
    conversation: TurnCompleter.RuntimeConversation = [
        {"role": "user", "content": "Who won the 2024 Tour de France?"}
    ]
    # when
    completion = asyncio.run(completer.async_completion(conversation))
    # then
    assert completion.output_messages, "no messages at all"
    last = completion.output_messages[-1]
    assert last.role == "assistant"
    assert isinstance(last.content, str)
    assert last.content.strip()


@pytest.mark.parametrize("model", MODELS_BASE)
def test_multimodal_image_input(model: str) -> None:
    """
    Send a mixed text+image prompt via the completer abstraction.
    Verify the assistant returns a descriptive text.
    """
    # given
    completer = OpenAIResponsesTurnCompleter(model=model)
    image_url = "https://images.pexels.com/photos/8538275/pexels-photo-8538275.jpeg"
    conversation: TurnCompleter.RuntimeConversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
            ],
        }
    ]
    # when
    completion = asyncio.run(completer.async_completion(conversation))
    msgs = [m for m in completion.output_messages if m.role == "assistant"]
    # then
    assert msgs, "no assistant messages"
    desc = msgs[0].content
    assert isinstance(desc, str) and len(desc.split()) > 3
    assert "cat" in desc.lower() or "kitten" in desc.lower()


def test_sync_completion_raises() -> None:
    """
    The blocking .completion() should raise NotImplemented.
    """
    completer = OpenAIResponsesTurnCompleter(model=MODELS_BASE[0])
    with pytest.raises(NotImplementedError):
        _ = completer.completion([{"role": "user", "content": "Sync test"}])
