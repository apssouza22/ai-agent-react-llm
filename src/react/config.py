from openai import OpenAI
from react.tools import Tool


class AgentConfig:
    def __init__(self):
        self.max_interactions = 3
        self.model = None
        self.tools: list = []
        self.system_instructions: str = ""
        self.token_limit: int = 5000

    def with_model_client(self, model: OpenAI):
        self.model = model
        return self

    def with_system_instructions(self, instructions: str):
        self.system_instructions = instructions
        return self

    def with_tools(self, tools: list[Tool]):
        self.tools = tools
        return self

    def with_token_limit(self, token_limit: int):
        self.token_limit = token_limit
        return self

    def with_max_interactions(self, max_int: int):
        self.max_interactions = max_int
