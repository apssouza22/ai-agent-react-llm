import inspect
import json

from common.agent_base import Agent
from react.brain import Brain, ReactEnd
from react.cache import CacheHandler
from react.tools import Tool, ToolChoice


class ReActExecutor:
    def __init__(self, config, agent: Agent) -> None:
        self.agent = agent
        self.config = config
        self.request = ""
        self.brain = Brain(config)
        self.cache = CacheHandler()

    def plan(self, current_agent: Agent) -> None:
        tools = self.__get_tools(current_agent)

        prompt = f"""Answer the following request as best you can: {self.request}.
        
First think step by step about what to do. Plan step by step what to do.
Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress.
Your goal is to demonstrate a thorough, adaptive, and self-reflective problem-solving process, emphasizing dynamic thinking and learning from your own reasoning.

Make sure to include the available tools in your plan.

Your available tools are: 
{tools}

CONTEXT HISTORY:
---
{self.brain.recall()}
"""
        response = self.brain.think(prompt=prompt, agent=current_agent)
        print(f"============= \n Thought: {response} \n=============")
        self.brain.remember("Assistant: " + response)

    @staticmethod
    def __get_tools(current_agent) -> str:
        tools = [tool for tool in current_agent.functions if isinstance(tool, Tool)]
        str_tools = [tool.name + " - " + tool.desc for tool in tools]
        return "\n".join(str_tools)

    def choose_action(self, current_agent: Agent) -> Tool:
        tools = self.__get_tools(current_agent)
        prompt = f"""To Answer the following request as best you can: {self.request}.
        
Choose the tool to use if need be. The tool should be among:
{tools}

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
        response: ToolChoice = self.brain.think(prompt=prompt, agent=current_agent, output_format=ToolChoice)

        message = f"""Assistant: I should use this tool: {response.tool_name}. {response.reason_of_choice}"""

        print(message)
        self.brain.remember(message)

        tool = [tool for tool in current_agent.functions if tool.name == response.tool_name]
        return tool[0] if tool else None

    def action(self, tool: Tool, current_agent: Agent) -> None:
        if tool is None:
            return

        print(f"""============= Executing {tool.name} =============""")
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
            response = self.brain.think(prompt=prompt, agent=current_agent)
            self.brain.remember("Assistant: " + response)
            # replace "json" string from response
            response = json.loads(response.replace("json", ""))

        action_result = tool.func(**response)
        message = f"Action Result: {action_result}"
        print("Action params: " + str(response))
        print(message)
        self.brain.remember(message)

    def observation(self, current_agent: Agent) -> ReactEnd:
        prompt = f"""Is the context information  enough to finally answer to this request: {self.request}?
       
Assign a quality confidence score between 0.0 and 1.0 to guide your approach:
   - 0.8+: Continue current approach
   - 0.5-0.7: Consider minor adjustments
   - Below 0.5: Seriously consider backtracking and trying a different approach
   
CONTEXT HISTORY:
---
{self.brain.recall()}
"""
        resp: ReactEnd = self.brain.think(prompt, agent=current_agent, output_format=ReactEnd)
        self.brain.remember("User: Is the context information enough to finally answer to this request?")
        self.brain.remember("Assistant: " + resp.final_answer)
        self.brain.remember("Assistant: Approach confidence score - " + str(resp.confidence))

        print("============== Observation =============")
        print(f"Observation: {resp.final_answer}")
        print(f"Approach confident score: {resp.confidence}")

        return resp

    def execute(self, input: str) -> str:
        self.request = input
        print(f"Request: {input}")
        total_interactions = 0
        agent = self.agent
        while True:
            total_interactions += 1
            self.plan(agent)
            tool = self.choose_action(agent)
            if tool:
                if isinstance(tool.func, Agent):
                    agent = tool.func
                    print(f"Agent: {agent.name}")
                    continue

                self.action(tool, agent)
            else:
                print("Tool not found")
                agent = self.agent

            observation = self.observation(agent)
            if observation.stop:
                print("Thought: I now know the final answer. \n")
                print(f"Final Answer: {observation.final_answer}")
                return observation.final_answer

            if self.config.max_interactions <= total_interactions:
                print("Max interactions reached. Exiting.")
                return ""
