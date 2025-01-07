import inspect
import json

from version1.brain import Brain
from common import AgentConfig, Agent, ReactEnd, Tool, ToolChoice


class ReActExecutor:
    def __init__(self, config: AgentConfig, agent: Agent) -> None:
        self.base_agent = agent
        self.config = config
        self.request = ""
        self.brain = Brain(config)

    def execute(self, query_input: str) -> str:
        print(f"Request: {query_input}")
        self.request = query_input
        total_interactions = 0
        agent = self.base_agent
        while True:
            total_interactions += 1
            if self.config.max_interactions <= total_interactions:
                print("Max interactions reached. Exiting.")
                return ""

            print("Current Agent: ", agent.name)
            self.__thought(agent)
            agent, skip = self.__action(agent)
            if skip:
                continue
            observation = self.__observation(agent)
            if observation.stop:
                print("Thought: I now know the final answer. \n")
                print(f"Final Answer: {observation.final_answer}")
                return observation.final_answer

    @staticmethod
    def __get_tools(agent: Agent) -> str:
        tools = [tool for tool in agent.functions if isinstance(tool, Tool)]
        str_tools = [tool.name + " - " + tool.desc for tool in tools]
        return "\n".join(str_tools)

    def __thought(self, current_agent: Agent) -> None:
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
        print(f"============= Thought =============")
        print(f"Thought response: {response} \n")
        self.brain.remember("Assistant: " + response)

    def __action(self, agent: Agent) -> tuple[Agent, bool]:
        tool = self.__choose_action(agent)
        if tool:
            if isinstance(tool.func, Agent):
                agent = tool.func
                print(f"Switching to the Agent: {agent.name} \n")
                return agent, True

            self.__execute_action(tool, agent)
        else:
            print("No tool found")
            agent = self.base_agent
            return agent, True
        return agent, False

    def __observation(self, current_agent: Agent) -> ReactEnd:
        prompt = f"""Is the context information  enough to finally answer to this request: {self.request}?
       
Assign a quality confidence score between 0.0 and 1.0 to guide your approach:
   - 0.8+: Continue current approach
   - 0.5-0.7: Consider minor adjustments
   - Below 0.5: Seriously consider backtracking and trying a different approach
   
CONTEXT HISTORY:
---
{self.brain.recall()}
"""
        response: ReactEnd = self.brain.think(prompt=prompt, agent=current_agent, output_format=ReactEnd)
        self.brain.remember("Assistant: " + response.final_answer)
        self.brain.remember("Assistant: " + f"Confidence score: {response.confidence}")

        print("\n ============== Observation ============ \n")
        print(f"Observation: {response.final_answer} \n")
        print(f"Approach Confidence score: {response.confidence} \n")

        return response

    def __choose_action(self, agent: Agent) -> Tool:
        tools = self.__get_tools(agent)
        prompt = f"""To Answer the following request as best you can: {self.request}.
Choose the tool to use if need be. The tool should be among:
{tools}

CONTEXT HISTORY:
---
{self.brain.recall()}
"""
        response: ToolChoice = self.brain.think(prompt, agent=agent, output_format=ToolChoice)
        message = f""" Assistant: I should use this tool: {response.tool_name}. Reason: {response.reason_of_choice}"""
        self.brain.remember(message)

        tool = [tool for tool in agent.functions if tool.name == response.tool_name]
        return tool[0] if tool else None

    def __execute_action(self, tool: Tool, agent: Agent):
        if tool is None:
            return

        print(f"\n ================== Executing: {tool.name} ================\n")

        prompt = f"""To Answer the following request as best you can: {self.request}.
Determine the inputs to send to the tool: {tool.name}
Given that the function signature of the tool function is: {inspect.signature(tool.func)}.

CONTEXT HISTORY:
---
{self.brain.recall()}
"""
        parameters = inspect.signature(tool.func).parameters
        response = {}
        if len(parameters) > 0:
            prompt += f"""RESPONSE FORMAT:
            {{
                {', '.join([f'"{param}": <function parameter>' for param in parameters])}
            }}"""
            response = self.brain.think(prompt=prompt, agent=agent)
            self.brain.remember("Assistant: " + response)

            try:
                response = json.loads(response)
            except Exception as e:
                print(f"Error in parsing response: {e}")
                print(f"Invalid response: {response}")
                self.brain.remember("Assistant: Error in parsing json response")
                return

        action_result = tool.func(**response)
        msg = f"Tool Result: {action_result}"
        print(f"Tool Params: {response}")
        print(msg)
        self.brain.remember(f"Assistant: {msg}")

