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


class ToolCallHandler:

    @staticmethod
    def __handle_function_result(result) -> FuncResult:
        if isinstance(result, FuncResult):
            return result

        if isinstance(result, Agent):
            agent: Agent = result
            return FuncResult(
                value=json.dumps({"assistant": agent.name}),
                agent=agent,
            )

        try:
            return FuncResult(value=str(result))
        except Exception as e:
            error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
            debug_print(error_message)
            raise TypeError(error_message)

    def handle_tool_calls(
            self,
            tool_calls: List[ChatCompletionMessageToolCall],
            functions: List[AgentFunction],
    ) -> TaskResponse:
        functions_map = {f.__name__: f for f in functions}
        partial_response = TaskResponse(
            messages=[],
            agent=None,
            context_variables={}
        )
        for tool_call in tool_calls:
            self.__handle_call(tool_call, functions_map, partial_response)
        return partial_response


    def __handle_call(self, tool_call:ChatCompletionMessageToolCall, function_map:dict, partial_response: TaskResponse):
        name = tool_call.function.name
        if name not in function_map:
            debug_print(f"Function {name} not found in function map")
            partial_response.messages.append({
                "role": "tool",
                "tool_name": name,
                "tool_call_id": tool_call.id,
                "content": f"Error: tool {name} not found"
            })
            return
        raw_result = self.__execute_tool(function_map, name, tool_call)
        result = self.__handle_function_result(raw_result)
        partial_response.messages.append({
            "role": "tool",
            "tool_name": name,
            "tool_call_id": tool_call.id,
            "content": result.value
        })
        if result.agent:
            partial_response.agent = result.agent


    @staticmethod
    def __execute_tool(function_map, name, tool_call:ChatCompletionMessageToolCall):
        args = json.loads(tool_call.function.arguments)
        debug_print(f"Executing tool {name} with args {args}")
        return function_map[name](**args)
