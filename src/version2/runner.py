import copy
import json
from collections import defaultdict


from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from common import Agent
from version2.result_handler import ToolCallHandler
from version2.types import TaskResponse
from version2.utils import debug_print


class AppRunner:
    def __init__(self, client: OpenAI):
        self.client = client
        self.tool_handler = ToolCallHandler()

    def run(self, agent: Agent, messages: list, variables: dict, max_interactions=10) -> TaskResponse:
        loop_count = 0
        active_agent = agent
        context_variables = copy.deepcopy(variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)
        while loop_count < max_interactions:
            llm_params = self.__create_inference_request(agent, history, variables)
            response = self.client.chat.completions.create(**llm_params)
            message: ChatCompletionMessage = response.choices[0].message
            debug_print( "Response from OpenAI:", str(response))
            message.sender = active_agent.name
            history_msg = json.loads(message.model_dump_json())
            history.append(history_msg)
            loop_count += 1
            if not message.tool_calls:
                debug_print( "No tool calls found in the response")
                break
            debug_print(message.tool_calls)
            response =self.tool_handler.handle_tool_calls(
                message.tool_calls,
                active_agent.functions,
            )
            debug_print( "Response from tool handler:", str(response))
            break
            # messages.extend(response.messages)
            # agent = response.agent

        return TaskResponse(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables
        )

    @staticmethod
    def __create_inference_request(agent: Agent, history: list, variables: dict) -> dict:
        context_variables = defaultdict(str, variables)
        instructions = agent.get_instructions(context_variables)
        messages = [{"role": "system", "content": instructions}] + history
        tools = agent.tools_in_json()
        debug_print( "Getting chat completion for...:", str(messages))

        params = {
            "model": agent.model,
            "messages": messages,
            "tool_choice": agent.tool_choice
        }
        if tools:
            params["parallel_tool_calls"] = agent.parallel_tool_calls
            params["tools"] = tools

        return params
