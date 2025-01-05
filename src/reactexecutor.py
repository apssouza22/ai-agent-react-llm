from typing import Tuple

from common import AgentConfig, Agent, ReactEnd


class ReActExecutor:
    def __init__(self, config: AgentConfig, agent: Agent) -> None:
        self.base_agent = agent
        self.config = config
        self.request = ""


    def __thought(self, current_agent: Agent) -> None:
        pass


    def __action(self, agent: Agent) -> tuple[Agent, bool]:
        pass

    def __observation(self, current_agent: Agent) -> ReactEnd:
        pass


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

            self.__thought(agent)
            agent, skip = self.__action(agent)
            if skip:
                continue
            observation = self.__observation(agent)
            if observation.stop:
                print("Thought: I now know the final answer. \n")
                print(f"Final Answer: {observation.final_answer}")
                return observation.final_answer





