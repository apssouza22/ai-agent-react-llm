from common import AgentConfig, Agent


class ReActExecutor:
    def __init__(self, config: AgentConfig, agent: Agent) -> None:
        self.base_agent = agent
        self.config = config
        self.request = ""


    def __thought(self, current_agent: Agent) -> None:
        pass

    def __action(self, current_agent: Agent) -> None:
        pass

    def __observation(self, current_agent: Agent) -> None:
        pass


    def execute(self, request: str) -> str:
        self.request = request
        current_agent = self.base_agent
        self.__thought(current_agent)
        self.__action(current_agent)
        return self.__observation(current_agent)


