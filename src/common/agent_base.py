from typing import Callable, Union, List

from pydantic import BaseModel
from pydantic.v1.typing import AnyCallable

from common.utils import function_to_json

AgentFunction = Callable[[], AnyCallable]


class AgentBase(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-2024-08-06"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List = []
    tool_choice: str = None
    parallel_tool_calls: bool = True

    def get_instructions(self, context_variables: dict = {}) -> str:
        if callable(self.instructions):
            return self.instructions(context_variables)
        return self.instructions

    def tools_in_json(self)->list[dict]:
        return [function_to_json(f) for f in self.functions]