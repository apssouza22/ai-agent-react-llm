from openai import OpenAI

from common import Agent
from version2.runner import AppRunner
from version2.types import TaskResponse

def transfer_to_lost_baggage():
    return "I will transfer you to the lost baggage department"

def transfer_to_cancel_flights():
    return "I will transfer you to the flight cancellation department"

main_agent = Agent(
    name="MainAgent",
    instructions=f"""
    You are to triage a users request, and call a tool to transfer to the right intent.
    """,
    functions=[transfer_to_lost_baggage, transfer_to_cancel_flights]
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
        query = "I want to cancel my flight"
        messages.append({"role": "user", "content": query})
        response = runner.run(main_agent, messages, context_variables)
        messages.extend(response.messages)
        agent = response.agent
        break

    print("Finishing the app")