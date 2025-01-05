import inspect
import json

from common import AgentConfig, Agent, ReactEnd, Tool, ToolChoice


class ReActExecutor:
    def __init__(self, config: AgentConfig, agent: Agent) -> None:
        self.base_agent = agent
        self.config = config
        self.request = ""

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

    def __thought(self, current_agent: Agent) -> None:
        pass

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
        return ReactEnd(stop=True, final_answer="This is the final answer", confidence=0.5)

    def __choose_action(self, agent: Agent) -> Tool:
        # TODO:  Implement the logic to choose the action
        # Given the current agent and the available tools, ask chatGPT to choose the best tool

        response: ToolChoice = ToolChoice(tool_name="People_search", reason_of_choice="reason_of_choice")
        tool = [tool for tool in agent.functions if tool.name == response.tool_name]
        return tool[0] if tool else None

    def __execute_action(self, tool: Tool, agent:Agent):
        if tool is None:
            return
        parameters = inspect.signature(tool.func).parameters

        #TODO: Ask chatGPT to set the parameters values
        response = f"""
        {{
            {', '.join([f'"{param}": <function parameter value>' for param in parameters])}
        }}
        """
        try:
            resp = json.loads(response)
        except json.JSONDecodeError:
            print("Error in setting the parameters")
            print(f"Invalid response: {response}")
            return

        action_result = tool.func(**resp)
        print(f"Action Result: {action_result}")

