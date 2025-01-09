from openai import OpenAI

from common import Agent
from version2.runner import AppRunner
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





if __name__ == "__main__":
    print("Starting the app")
    runner = AppRunner(client = OpenAI())
    messages = []
    agent = main_agent
    while True:
        # query = input("Enter your query: ")
        query = "What day is today"
        messages.append({"role": "user", "content": query})
        response = runner.run(main_agent, messages, context_variables)
        messages.extend(response.messages)
        agent = response.agent
        break

    print("Finishing the app")