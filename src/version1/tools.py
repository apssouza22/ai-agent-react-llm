import datetime

import wikipedia

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

def search_wikipedia(search_query:str):
    try:
        page = wikipedia.page(search_query)
        text = page.content
    except Exception as e:
        return ("Could not find any information on wikipedia for the search query: "
                + search_query + ". Please try another search term")
    return text[:300]

def date_of_today():
    return datetime.date.today()

people_search_agent = Agent(
    name="People_search_Agent",
    instructions=f"""You are a helpful assistant that help to find people information on the Wikipedia using its name.""",
)

calculator_tool = Tool("Calculator", perform_calculator, "To perform math calculations")
wikipedia_tool = Tool("Wikipedia_search", search_wikipedia, "To search for information on wikipedia")
today_tool = Tool("Date_of_today", date_of_today, "To get the date of today")
people_search_tool = Tool("People_search", people_search_agent, "To search for person information")
people_search_agent.functions = [wikipedia_tool]

