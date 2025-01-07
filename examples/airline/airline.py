from configs.agents import *
from openai import OpenAI
from openai_runner import AutoRunner
from openai_runner.util import pretty_print_messages

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
    runner = AutoRunner(client=OpenAI(), debug=True)
    print("Starting Agent CLI")

    messages = []
    agent = triage_agent

    while True:
        user_input = input("\033[90mUser\033[0m: ")
        messages.append({"role": "user", "content": user_input})

        response = runner.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
        )

        pretty_print_messages(response.messages)
        messages.extend(response.messages)
        agent = response.agent
