from __future__ import annotations

import functools
import os
from typing import Any, Iterable, Literal, Unpack

import structlog
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    warn_about_non_empty_params,
)
from pydantic import BaseModel, ConfigDict, field_validator

logger = structlog.stdlib.get_logger(component=__name__)

# Import Google API with graceful fallback
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse
    GOOGLE_AVAILABLE = True
except ImportError:
    logger.warning("google-generativeai not available. Please install it to use Google models.")
    GOOGLE_AVAILABLE = False
    genai = None
    GenerateContentResponse = None


class GoogleCompletionsTurnCompleter(TurnCompleter):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        retry_config: RetryConfig = DEFAULT_RETRY_CONFIG,
        **kwargs: Any,  # Ignore other OpenAI-specific parameters
    ):
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "google-generativeai is required for GoogleCompletionsTurnCompleter. "
                "Install it with: pip install google-generativeai"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.retry_config = retry_config

        # Ignore OpenAI-specific parameters with warning
        ignored_params = [k for k in kwargs.keys() if k in ['reasoning_effort', 'response_format', 'tools', 'tool_choice']]
        if ignored_params:
            logger.warning(f"Ignoring OpenAI-specific parameters: {ignored_params}")

        # Use generic encoding for token estimation
        self.encoding_name = "cl100k_base"
        self.n_ctx = self._get_google_context_window(model)

        # Initialize Google client
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable or api_key parameter is required")

        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(model)

    def _get_google_context_window(self, model: str) -> int:
        """Get context window for Google models"""
        google_context_windows = {
            "gemini-2.5-pro": 2_000_000,  # Gemini 2.5 Pro has 2M context window
            "gemini-1.5-pro": 1_000_000,
            "gemini-1.5-pro-002": 1_000_000,
            "gemini-1.5-flash": 1_000_000,
            "gemini-2.0-flash": 1_000_000,
            "gemini-pro": 30_720,
            "gemini-pro-vision": 30_720,
        }
        return google_context_windows.get(model, 2_000_000)  # Default to 2.5 Pro context size

    class Config(TurnCompleter.Config):
        """Configuration for Google Completions"""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_encoders={NotGiven: lambda v: "NOT_GIVEN"},
        )

        model: str
        api_key: str | None = None
        temperature: float | None | NotGiven = NOT_GIVEN
        max_tokens: int | None | NotGiven = NOT_GIVEN
        top_p: float | None | NotGiven = NOT_GIVEN
        retry_config: RetryConfig = DEFAULT_RETRY_CONFIG

        def build(self) -> GoogleCompletionsTurnCompleter:
            return GoogleCompletionsTurnCompleter(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                retry_config=self.retry_config,
            )

        @field_validator("*", mode="before")
        @classmethod
        def _decode_not_given(cls, v: Any) -> Any:
            """Turn the string "NOT_GIVEN" back into our sentinel before validation."""
            if v == "NOT_GIVEN":
                return NOT_GIVEN
            return v

    class Completion(TurnCompleter.Completion):
        usage: CompletionUsage | None = None

    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> GoogleCompletionsTurnCompleter.Completion:
        raise NotImplementedError("Not implemented, use async_completion instead")

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> GoogleCompletionsTurnCompleter.Completion:
        warn_about_non_empty_params(self, **params)

        # Convert OpenAI conversation format to Google format
        google_prompt = self._convert_to_google_format(conversation)

        # Prepare generation config
        generation_config = {}
        if self.temperature is not NOT_GIVEN and self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.max_tokens is not NOT_GIVEN and self.max_tokens is not None:
            generation_config["max_output_tokens"] = self.max_tokens
        if self.top_p is not NOT_GIVEN and self.top_p is not None:
            generation_config["top_p"] = self.top_p

        # Call Google API with retry logic
        async for attempt in self.retry_config.build():
            with attempt:
                response = await self._client.generate_content_async(
                    google_prompt,
                    generation_config=genai.GenerationConfig(**generation_config) if generation_config else None,
                )

        # Convert response back to OpenAI format
        openai_message = self._convert_google_response_to_openai(response)

        # Estimate token usage
        usage = self._estimate_usage(conversation, response)

        return GoogleCompletionsTurnCompleter.Completion(
            input_conversation=conversation,
            output_messages=[openai_message],
            usage=usage,
        )

    def _convert_to_google_format(self, messages: list) -> str:
        """Convert OpenAI message format to Google format"""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"System Instructions: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"{role.title()}: {content}")

        return "\n\n".join(parts)

    def _convert_messages_to_gemini_contents(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        按 research-assistant/llm/gemini_provider.py 的方式，将OpenAI消息转换为Gemini contents：
        - 角色映射：assistant->model, tool->user, system->user；仅保留 user/model
        - 合并相邻同角色消息，使用 "\n\n" 拼接
        """
        contents: list[dict[str, Any]] = []
        current_role: str | None = None
        merged: list[str] = []

        for msg in messages:
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            if role == "assistant":
                role = "model"
            elif role in ("tool", "system"):
                role = "user"
            if role not in ("user", "model"):
                # skip unknown role
                continue
            if current_role is None:
                current_role = role
            if role == current_role:
                merged.append(content)
            else:
                if merged:
                    contents.append({
                        "role": current_role,
                        "parts": [{"text": "\n\n".join(merged)}],
                    })
                current_role = role
                merged = [content]

        if merged:
            contents.append({
                "role": current_role or "user",
                "parts": [{"text": "\n\n".join(merged)}],
            })
        return contents

    def _convert_google_response_to_openai(self, response: GenerateContentResponse) -> ChatCompletionMessage:
        """Convert Google response to OpenAI ChatCompletionMessage format"""
        # Extract text content from Google response
        try:
            content = response.text if hasattr(response, 'text') else ""
        except Exception as e:
            logger.warning(f"Failed to extract text from Google response: {e}")
            content = ""

        return ChatCompletionMessage(
            role="assistant",
            content=content,
            refusal=None,
            tool_calls=None,
            audio=None,
        )

    def _estimate_usage(self, conversation: list, response: GenerateContentResponse) -> CompletionUsage:
        """
        计算token：优先使用 Gemini 官方SDK 的 count_tokens，与 research-assistant/llm/gemini_provider.py 保持一致；
        若不可用则回退到字符/4估算。
        """
        # 1) 构造 contents（优先结构化）
        try:
            contents = self._convert_messages_to_gemini_contents(conversation)
        except Exception:
            contents = None

        prompt_tokens = None
        completion_tokens = None

        # 2) 尝试使用 google-genai (优先)
        try:
            try:
                from google import genai as genai_new  # type: ignore
            except Exception:
                genai_new = None  # type: ignore

            if genai_new is not None:
                # 优先以 GOOGLE_API_KEY，如果不存在则尝试 GEMINI_API_KEY
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                client = genai_new.Client(api_key=api_key) if api_key else genai_new.Client()

                # prompt tokens
                if contents is not None:
                    ct_resp = client.models.count_tokens(model=self.model, contents=contents)
                else:
                    prompt_text = " \n".join(str(m.get("content", "")) for m in conversation)
                    ct_resp = client.models.count_tokens(
                        model=self.model,
                        contents=[{"parts": [{"text": prompt_text}]}],
                    )
                if hasattr(ct_resp, "total_tokens"):
                    prompt_tokens = int(ct_resp.total_tokens)

                # completion tokens（基于文本再次计数）
                completion_text = getattr(response, "text", "") if response else ""
                if completion_text:
                    ct_out = client.models.count_tokens(
                        model=self.model,
                        contents=[{"parts": [{"text": str(completion_text)}]}],
                    )
                    if hasattr(ct_out, "total_tokens"):
                        completion_tokens = int(ct_out.total_tokens)
        except Exception:
            # 忽略，进入后续回退
            pass

        # 3) 尝试使用 google-generativeai 的 count_tokens
        if (prompt_tokens is None or completion_tokens is None) and hasattr(self._client, "count_tokens"):
            try:
                # prompt tokens
                prompt_input = contents if contents is not None else self._convert_to_google_format(conversation)
                ct2 = self._client.count_tokens(prompt_input)
                if hasattr(ct2, "total_tokens") and prompt_tokens is None:
                    prompt_tokens = int(ct2.total_tokens)
                # completion tokens
                completion_text = getattr(response, "text", "") if response else ""
                if completion_text:
                    ct2_out = self._client.count_tokens(completion_text)
                    if hasattr(ct2_out, "total_tokens") and completion_tokens is None:
                        completion_tokens = int(ct2_out.total_tokens)
            except Exception:
                pass

        # 4) 最终回退：字符/4
        if prompt_tokens is None:
            prompt_text = " ".join(str(msg.get("content", "")) for msg in conversation)
            prompt_tokens = max(len(prompt_text) // 4, 1)
        if completion_tokens is None:
            completion_text = getattr(response, 'text', "") if response else ""
            completion_tokens = max(len(completion_text) // 4, 0)

        return CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(prompt_tokens + completion_tokens),
        )
