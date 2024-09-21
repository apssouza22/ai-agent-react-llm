from openai import OpenAI

from tools import Tool


class AgentConfig:
    def __init__(self):
        self.model = None
        self.tools:list = []
        self.system_instructions:str = ""
        self.token_limit:int = 5000

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
