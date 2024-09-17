from agent import Tool, Agent
from tools import search_wikipedia, perform_calculation, date_of_today

wikipedia_search_tool = Tool("WikipediaSearch", search_wikipedia, "To search for information on wikipedia")
calculator_tool = Tool("Calculator", perform_calculation, "To perform math calculations")
date_request_tool = Tool("Date_of_today", date_of_today, "To get the date of today")

agent = Agent()
agent.add_tool(wikipedia_search_tool)
agent.add_tool(calculator_tool)
agent.add_tool(date_request_tool)
agent.react("What is the double of barack obama's age?")
