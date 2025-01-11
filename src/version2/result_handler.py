import json
from typing import List

from openai.types.chat import ChatCompletionMessageToolCall

from .types import (
    AgentFunction,
    TaskResponse,
    FuncResult,
    Agent,
)
from .utils import debug_print


class ResultHandler:

    @staticmethod
    def __handle_function_result(result) -> FuncResult:
        pass

    def handle_tool_calls(
            self,
            tool_calls: List[ChatCompletionMessageToolCall],
            functions: List[AgentFunction],
    ) -> TaskResponse:
        pass

    def __handle_call(self, tool_call:ChatCompletionMessageToolCall, function_map:dict, partial_response: TaskResponse):
        pass

    def __execute_tool(self, function_map, name, tool_call:ChatCompletionMessageToolCall):
        pass
