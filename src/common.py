from typing import Union, Callable, List

from openai import OpenAI
from pydantic import BaseModel, Field


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

