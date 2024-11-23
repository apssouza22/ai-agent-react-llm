from typing import List, Callable, Union, Optional
from pydantic import BaseModel

from openai_runner.util import function_to_json

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True

    def get_instructions(self, context_variables: dict = {}) -> str:
        if callable(self.instructions):
            return self.instructions(context_variables)
        return self.instructions

    def tools_in_json(self)->list[dict]:
        return [function_to_json(f) for f in self.functions]

class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
