from typing import Union, Callable, List, Any

from openai import OpenAI
from pydantic import BaseModel, Field

from version2.utils import function_to_json


class ToolChoice(BaseModel):
    """Data model for the tool choice"""
    tool_name: str = Field(..., description="Name of the tool to use")
    reason_of_choice: str = Field(..., description="Reason for choosing the tool")

class ReactEnd(BaseModel):
    """Data model for the observation step"""
    stop: bool = Field(..., description="True if the context is enough to answer the request else False")
    final_answer: str = Field(..., description="Final answer if the context is enough to answer the request")
    confidence: float = Field(..., description="Confidence score of the final answer")


class Tool:
    def __init__(self, name: str, func, desc) -> None:
        self.desc = desc
        self.name = name
        self.func = func


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List = []
    parallel_tool_calls: bool = True
    tool_choice: str = None

    def tools_in_json(self):
        return [function_to_json(f) for f in self.functions]

    def get_instructions(self, context_variables: dict = {}) -> str:
        if callable(self.instructions):
            return self.instructions(context_variables)
        return self.instructions

class AgentConfig:
    def __init__(self):
        self.max_interactions = 3
        self.model = None
        self.token_limit: int = 5000

    def with_model_client(self, model: OpenAI):
        self.model = model
        return self

    def with_token_limit(self, token_limit: int):
        self.token_limit = token_limit
        return self

    def with_max_interactions(self, max_int: int):
        self.max_interactions = max_int

