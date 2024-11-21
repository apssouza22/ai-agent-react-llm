import copy
import json
from collections import defaultdict
from typing import List

from .result_handler import ResultHandler
from .types import (
    Agent,
    Response,
)
from .util import function_to_json, debug_print

__CTX_VARS_NAME__ = "context_variables"


class AgentRunner:

    def __init__(self, client=None, debug=False):
        self.debug = debug
        self.client = client
        self.result_handler = ResultHandler(debug=debug)

    def __create_inference_request(
            self,
            agent: Agent,
            history: List,
            context_variables: dict
    ) -> dict:
        context_variables = defaultdict(str, context_variables)
        instructions = agent.get_instructions(context_variables)
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(self.debug, "Getting chat completion for...:", messages)

        tools = agent.tools_in_json()
        self.__hide_context_vars(tools)

        create_params = {
            "model": agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        return create_params

    @staticmethod
    def __hide_context_vars(tools):
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

    def run(
            self,
            agent: Agent,
            messages: List,
            context_variables: dict = {},
            max_turns: int = float("inf"),
            execute_tools: bool = True,
    ) -> Response:
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)
        loop_count = 0

        while loop_count < max_turns and active_agent:
            create_params = self.__create_inference_request(
                agent=active_agent,
                history=history,
                context_variables=context_variables
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

            partial_response = self.result_handler.handle_tool_calls(
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
