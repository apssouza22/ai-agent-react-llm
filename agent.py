import json
from typing import Callable, Optional, Type

from pydantic import BaseModel
import inspect

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


def llm_generate(prompt: str, output_format: Optional[Type[BaseModel]] = None):
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that assists the user in completing a task. Don't ask for user input. 
            Important! You are not good with math operations therefore you must to use a Calculator tool to perform math calculations.
            Important! You don't know the date of today, therefore you must use the Date_of_today tool to get the date of today.
             
        Given the following information from the context history 
        provide for the user's task using only this information\n\n """,
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]
    if output_format:
        completion = client.beta.chat.completions.parse(
            messages=messages,
            # model="gpt-4o-2024-08-06",
            model="gpt-4o-mini-2024-07-18",
            temperature=0.1,
            max_tokens=1000,
            response_format=output_format,
        )
        return completion.choices[0].message.parsed

    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.1
    )
    return response.choices[0].message.content


class ReactEnd(BaseModel):
    stop: bool
    final_answer: str


class ToolChoice(BaseModel):
    tool_name: str
    reason_of_choice: str


class Tool:
    def __init__(self, name: str, func: Callable, desc) -> None:
        self.desc = desc
        self.name = name
        self.func = func

    def act(self, **kwargs) -> str:
        return self.func(**kwargs)


class Agent:
    def __init__(self) -> None:
        self.tools = []
        self.messages = []
        self.request = ""
        self.token_count = 0
        self.token_limit = 5000

    def add_tool(self, tool: Tool) -> None:
        self.tools.append(tool)

    def append_message(self, message):
        self.messages.append(message)
        self.token_count += len(message)

        # Check if token_count exceeds the limit
        while self.token_count > self.token_limit and len(self.messages) > 1:
            # Remove messages from the end until token_count is within the limit
            removed_message = self.messages.pop(1)  # Keep the first message, remove the second one
            self.token_count -= len(removed_message)

    def background_info(self) -> str:
        join = "\n".join(self.messages[1:])
        return join

    def plan(self) -> None:
        prompt = f"""Answer the following request as best you can: {self.request}.
                    
        First think step by step about what to do. Plan step by step what to do.
        
        Your available tools are: 
        {[tool.name + " - " + tool.desc for tool in self.tools]}
        
        CONTEXT HISTORY:
        ---
        {self.background_info()}
"""
        response = llm_generate(prompt=prompt)
        print(f"Thought: {response}")
        self.append_message("Assistant: " + response)

    def choose_action(self) -> Tool:
        prompt = f"""To Answer the following request as best you can: {self.request}.
                    Choose the tool to use if need be. The tool should be among:
                    {[tool.name for tool in self.tools]}.
                    
                    CONTEXT HISTORY:
                    ---
                    {self.background_info()}

                    RESPONSE FORMAT:
                    {{
                        "tool_name": "ToolName",
                        "reason_of_choice": "Reason for choosing the tool"
                    }}
                    """
        self.append_message("User: Choose the tool to use if need be.")
        response: ToolChoice = llm_generate(prompt=prompt, output_format=ToolChoice)

        message = f"""Assistant: I should use this tool: {response.tool_name}. {response.reason_of_choice}"""

        print(message)
        self.append_message(message)

        tool = [tool for tool in self.tools if tool.name == response.tool_name].pop()
        return tool

    def action(self, tool: Tool) -> None:
        if tool is None:
            return
        parameters = inspect.signature(tool.func).parameters
        response = {}
        prompt = f"""To Answer the following request as best you can: {self.request}.
                    Determine the inputs to send to the tool: {tool.name}
                    Given that the function signature of the tool function is: {inspect.signature(tool.func)}.
                    
                    CONTEXT HISTORY:
                    ---
                    {self.background_info()}
                    """

        if len(parameters) > 0:
            prompt += f"""
                    RESPONSE FORMAT:
                    {{
                        {', '.join([f'"{param}": <function parameter>' for param in parameters])}
                    }}"""
            self.append_message("User: Determine the inputs to send to the tool:" + tool.name)
            response = llm_generate(prompt=prompt)
            self.append_message("Assistant: " + response)
            response = json.loads(response)

        action_result = tool.func(**response)
        message = f"Action Result: {action_result}"

        print("Action params: " + str(response))
        print(message)
        self.append_message(message)

    def observation(self) -> ReactEnd:
        prompt = f"""Is the context information  enough to finally answer to this request: {self.messages[0]}?
        
        CONTEXT HISTORY:
        ---
        {self.background_info()}
        
        RESPONSE FORMAT:
        {{
            "stop": "True if the context is enough to answer the request",
            "final_answer": "Final answer if the context is enough to answer the request"
        }}
        
"""
        check_final = llm_generate(prompt, output_format=ReactEnd)
        self.append_message("User: Is the context information enough to finally answer to this request?")
        self.append_message("Assistant: " + check_final.final_answer)
        print(f"Observation: {check_final.final_answer}")

        return check_final

    def react(self, input: str) -> str:
        self.append_message(input)
        self.request = input
        print(f"Request: {input}")
        while True:
            self.plan()
            tool = self.choose_action()
            if tool:
                self.action(tool)
            observation = self.observation()
            if observation.stop:
                print("Thought: I now know the final answer. \n")
                print(f"Final Answer: {observation.final_answer}")
                break
