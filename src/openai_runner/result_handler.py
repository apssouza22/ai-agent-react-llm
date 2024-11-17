import json
from typing import List

from openai.types.chat import ChatCompletionMessageToolCall
from .types import (
    AgentFunction,
    Response,
    Result,
    Agent,
)
from openai_runner.util import debug_print

__CTX_VAR_NAME__ = "context_variables"


class ResultHandler:

    def __init__(self, debug=True):
        self.debug = debug

    @staticmethod
    def __handle_function_result(result, debug) -> Result:
        if isinstance(result, Result):
            return result

        if isinstance(result, Agent):
            agent: Agent = result
            return Result(
                value=json.dumps({"assistant": agent.name}),
                agent=agent,
            )

        try:
            return Result(value=str(result))
        except Exception as e:
            error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
            debug_print(debug, error_message)
            raise TypeError(error_message)

    def handle_tool_calls(
            self,
            tool_calls: List[ChatCompletionMessageToolCall],
            functions: List[AgentFunction],
            context_variables: dict,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(
            messages=[],
            agent=None,
            context_variables={}
        )

        for tool_call in tool_calls:
            self.__handle_call(tool_call, function_map, context_variables, partial_response)

        return partial_response

    def __handle_call(self, tool_call, function_map, context_variables, partial_response):
        name = tool_call.function.name
        # handle missing tool case, skip to next tool
        if name not in function_map:
            debug_print(self.debug, f"Tool {name} not found in function map.")
            partial_response.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": f"Error: Tool {name} not found.",
            })
            return

        raw_result = self.__execute_tool(context_variables, function_map, name, tool_call)
        result: Result = self.__handle_function_result(raw_result, self.debug)

        partial_response.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "tool_name": name,
            "content": result.value,
        })
        partial_response.context_variables.update(result.context_variables)
        if result.agent:
            partial_response.agent = result.agent

    def __execute_tool(self, context_variables, function_map, name, tool_call):
        args = json.loads(tool_call.function.arguments)
        debug_print(self.debug, f"Processing tool call: {name} with arguments {args}")
        func = function_map[name]
        # pass context_variables to agent functions
        if __CTX_VAR_NAME__ in func.__code__.co_varnames:
            args[__CTX_VAR_NAME__] = context_variables
        raw_result = function_map[name](**args)
        return raw_result
