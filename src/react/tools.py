import datetime
from typing import Callable, Union
import wikipedia
from pydantic import BaseModel, Field


class ToolChoice(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to use")
    reason_of_choice: str = Field(..., description="Reason for choosing the tool")


class Tool:
    def __init__(self, name: str, func, desc) -> None:
        self.desc = desc
        self.name = name
        self.func = func

    def act(self, **kwargs) -> str:
        return self.func(**kwargs)

AgentFunction = Callable[[], Union[str, "Agent", dict, Tool]]

def perform_calculation(operation, a, b):
    if operation not in ['add', 'subtract', 'multiply', 'divide']:
        return f"Invalid operation: {operation}, should be among ['add', 'subtract', 'multiply', 'divide']"

    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b == 0:
            return "Division by zero"
        return a / b


def search_wikipedia(search_query):
    try:
        page = wikipedia.page(search_query)
        text = page.content
    except Exception as e:
        return ("Could not find any information on wikipedia for the search query: "
                + search_query + ". Please try another search term")
    return text[:300]


# Equivalent of the date_req function
def date_of_today():
    return datetime.date.today()
