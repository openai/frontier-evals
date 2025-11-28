from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable

import structlog.stdlib
import tenacity
from openai import NotGiven
from openai.types.responses.tool_param import ParseableToolParam
from preparedness_turn_completer.oai_responses_turn_completer.completer import (
    OpenAIResponsesTurnCompleter,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import RetryConfig, retry_predicate
from pydantic import Field

from paperbench.agents.basicagent.tools.base import Tool

logger = structlog.stdlib.get_logger(component=__name__)


class TimeTrackingRetryConfig(RetryConfig):
    time_spent_retrying: float = 0.0

    def build(self) -> tenacity.AsyncRetrying:
        # just before sleeping, we'll track how long we'll sleep for
        def _track_wait(rs: tenacity.RetryCallState) -> None:
            wait_time = rs.next_action.sleep if rs.next_action else 0
            self.time_spent_retrying += wait_time

        # and we also want to keep the base log behaviour
        base_log_callback = (
            tenacity.before_sleep_log(logger._logger, logging.WARNING) if logger._logger else None
        )

        # so we compose them together
        def _compose(
            a: Callable[[tenacity.RetryCallState], None] | None,
            b: Callable[[tenacity.RetryCallState], None] | None,
        ) -> Callable[[tenacity.RetryCallState], None] | None:
            if a and b:

                def _both(rs: tenacity.RetryCallState) -> None:
                    a(rs)
                    b(rs)

                return _both
            return a or b

        return tenacity.AsyncRetrying(
            wait=tenacity.wait_random_exponential(min=self.wait_min, max=self.wait_max),
            stop=tenacity.stop_after_delay(self.stop_after),
            retry=retry_predicate,
            before_sleep=_compose(_track_wait, base_log_callback),
            reraise=True,
        )


class BasicAgentTurnCompleterConfig(TurnCompleter.Config, ABC):
    """
    Light wrapper around the base `TurnCompleter.Config` Abstract Base Class.

    Adds two attributes expected by BasicAgent-aware completers:
    - basicagent_tools: Optional list of BasicAgent `Tool` instances that a completer
      should convert/forward into its native tool format (e.g., OpenAI tools).
    - retry_config: Defaults to `TimeTrackingRetryConfig`, enabling collection of the
      total time spent in API retries. Completers can ignore this field if unsupported;
      in that case, retry time will simply not be tracked.

    Implementations should handle `basicagent_tools` when present and, where possible,
    leverage `TimeTrackingRetryConfig` to report retry time back to the solver.

    See `OpenAIResponsesTurnCompleterConfig` in this module for a concrete example
    of how a completer integrates with these fields.
    """

    basicagent_tools: list[Tool] | None = None
    retry_config: RetryConfig = Field(default_factory=TimeTrackingRetryConfig)

    @abstractmethod
    def build(self) -> TurnCompleter: ...


class OpenAIResponsesTurnCompleterConfig(
    OpenAIResponsesTurnCompleter.Config, BasicAgentTurnCompleterConfig
):
    def build(self) -> OpenAIResponsesTurnCompleter:
        if self.basicagent_tools is not None:
            responses_basic_agent_tools = self._basicagent_to_responses_tools(self.basicagent_tools)

            if self._tools_is_set():
                self.tools: list[ParseableToolParam] = (
                    list(self.tools) + responses_basic_agent_tools
                )
            else:
                self.tools = responses_basic_agent_tools

        return OpenAIResponsesTurnCompleter.Config.build(self)

    def _basicagent_to_responses_tools(self, tools: list[Tool]) -> list[ParseableToolParam]:
        tools_responses: list[ParseableToolParam] = [tool.get_oai_tool_call() for tool in tools]

        return tools_responses

    def _tools_is_set(self) -> bool:
        if isinstance(self.tools, NotGiven):
            return False
        else:
            return len(list(self.tools)) > 0
