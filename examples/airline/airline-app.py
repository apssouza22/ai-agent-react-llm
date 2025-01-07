from openai import OpenAI

from common import Agent
from version2.types import TaskResponse

main_agent = Agent(
    name="MainAgent",
    instructions=f"""
    You are a helpful assistant that assists the user in completing a task using multiple tools.
    """,
    functions=[]
)
context_variables = {
    "name": "Linus Torvalds",
    "age": 52
}


class AppRunner:
    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, query: str, agent:Agent, variables:dict, messages:list) -> TaskResponse:
        return TaskResponse()


if __name__ == "__main__":
    print("Starting the app")
    runner = AppRunner(client = OpenAI())
    messages = []
    agent = main_agent
    while True:
        # query = input("Enter your query: ")
        query = "test"
        response = runner.run(query, main_agent, context_variables, messages)
        messages.extend(response.messages)
        agent = response.agent
        break

    print("Finishing the app")