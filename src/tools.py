from common import Tool, Agent


def perform_calculator(operation:str, a, b):
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        return a / b
    return "Invalid operation"

calculator_tool = Tool("Calculator", perform_calculator, "To perform math calculations")

people_search_agent = Agent(
    name="People_search_Agent",
    instructions=f"""You are a helpful assistant that help to find people information on the Wikipedia using its name.""",
    functions=[calculator_tool]
)

people_search_tool = Tool("People_search", people_search_agent, "To search for person information")
