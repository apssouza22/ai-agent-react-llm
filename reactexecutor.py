import inspect
import json
from brain import Brain, ReactEnd
from tools import Tool, ToolChoice


class ReActExecutor:
    def __init__(self, config) -> None:
        self.config = config
        self.request = ""
        self.brain = Brain(config)

    def plan(self) -> None:
        prompt = f"""Answer the following request as best you can: {self.request}.
                    
        First think step by step about what to do. Plan step by step what to do.
        
        Your available tools are: 
        {[tool.name + " - " + tool.desc for tool in self.config.tools]}
        
        CONTEXT HISTORY:
        ---
        {self.brain.recall()}
"""
        response = self.brain.think(prompt=prompt)
        print(f"Thought: {response}")
        self.brain.remember("Assistant: " + response)

    def choose_action(self) -> Tool:
        prompt = f"""To Answer the following request as best you can: {self.request}.
                    Choose the tool to use if need be. The tool should be among:
                    {[tool.name for tool in self.config.tools]}.
                    
                    CONTEXT HISTORY:
                    ---
                    {self.brain.recall()}

                    RESPONSE FORMAT:
                    {{
                        "tool_name": "ToolName",
                        "reason_of_choice": "Reason for choosing the tool"
                    }}
                    """
        self.brain.remember("User: Choose the tool to use if need be.")
        response: ToolChoice = self.brain.think(prompt=prompt, output_format=ToolChoice)

        message = f"""Assistant: I should use this tool: {response.tool_name}. {response.reason_of_choice}"""

        print(message)
        self.brain.remember(message)

        tool = [tool for tool in self.config.tools if tool.name == response.tool_name].pop()
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
                    {self.brain.recall()}
                    """

        if len(parameters) > 0:
            prompt += f"""
                    RESPONSE FORMAT:
                    {{
                        {', '.join([f'"{param}": <function parameter>' for param in parameters])}
                    }}"""
            self.brain.remember("User: Determine the inputs to send to the tool:" + tool.name)
            response = self.brain.think(prompt=prompt)
            self.brain.remember("Assistant: " + response)
            response = json.loads(response)

        action_result = tool.func(**response)
        message = f"Action Result: {action_result}"

        print("Action params: " + str(response))
        print(message)
        self.brain.remember(message)

    def observation(self) -> ReactEnd:
        prompt = f"""Is the context information  enough to finally answer to this request: {self.request}?
        
        CONTEXT HISTORY:
        ---
        {self.brain.recall()}
        
        RESPONSE FORMAT:
        {{
            "stop": "True if the context is enough to answer the request",
            "final_answer": "Final answer if the context is enough to answer the request"
        }}
        
"""
        resp = self.brain.think(prompt, output_format=ReactEnd)
        self.brain.remember("User: Is the context information enough to finally answer to this request?")
        self.brain.remember("Assistant: " + resp.final_answer)
        print(f"Observation: {resp.final_answer}")

        return resp

    def execute(self, input: str) -> str:
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



