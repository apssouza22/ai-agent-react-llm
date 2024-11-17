# Standard library imports
import copy
import json
from collections import defaultdict
from typing import List

# Package/library imports
from openai import OpenAI

from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)
# Local imports
from .util import function_to_json, debug_print, merge_chunk

__CTX_VARS_NAME__ = "context_variables"


class Swarm:
    def __init__(self, client=None, debug=False):
        self.debug = debug
        if not client:
            client = OpenAI()
        self.client = client

    def create_inference_request(
            self,
            agent: Agent,
            history: List,
            context_variables: dict,
            model_override: str
    ) -> dict:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(self.debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        return create_params

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
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
                continue
            args = json.loads(tool_call.function.arguments)
            debug_print(self.debug, f"Processing tool call: {name} with arguments {args}")

            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

            result: Result = self.handle_function_result(raw_result, self.debug)
            partial_response.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": result.value,
            })
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run(
            self,
            agent: Agent,
            messages: List,
            context_variables: dict = {},
            model_override: str = None,
            max_turns: int = float("inf"),
            execute_tools: bool = True,
    ) -> Response:
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)
        loop_count = 0

        while loop_count < max_turns and active_agent:
            # get completion with current history, agent
            create_params = self.create_inference_request(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override
            )
            completion = self.client.chat.completions.create(**create_params)
            message = completion.choices[0].message
            debug_print(self.debug, "Received completion:", message)
            message.sender = active_agent.name
            history_msg = json.loads(message.model_dump_json())  # to avoid OpenAI types (?)
            history.append(history_msg)
            loop_count = loop_count + 1
            if not message.tool_calls or not execute_tools:
                debug_print(self.debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                message.tool_calls,
                active_agent.functions,
                context_variables
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
