from openai import OpenAI

from agents import triage_agent
from common import Agent
from version2.runner import AppRunner
from version2.utils import pretty_print_messages

# Load this data from a database
context_variables = {
    "customer_context": """Here is what you know about the customer's details:
1. CUSTOMER_ID: customer_12345
2. NAME: John Doe
3. PHONE_NUMBER: (123) 456-7890
4. EMAIL: johndoe@example.com
5. STATUS: Premium
6. ACCOUNT_STATUS: Active
7. BALANCE: $0.00
8. LOCATION: 1234 Main St, San Francisco, CA 94123, USA
""",
    "flight_context": """The customer has an upcoming flight from LGA (Laguardia) in NYC to LAX in Los Angeles.
The flight # is 1919. The flight departure date is 3pm ET, 5/21/2024.""",
}

if __name__ == "__main__":
    print("Starting the app")
    runner = AppRunner(client = OpenAI())
    messages = []
    agent = triage_agent
    while True:
        query = input("Enter your query: ")
        messages.append({"role": "user", "content": query})
        response = runner.run(agent, messages, context_variables)
        messages.extend(response.messages)
        agent = response.agent
        pretty_print_messages(response.messages)

    print("Finishing the app")