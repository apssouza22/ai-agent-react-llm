import os
from openai import OpenAI

from config import AgentConfig
from reactexecutor import ReActExecutor
from tools import search_wikipedia, perform_calculation, date_of_today, Tool

wikipedia_search_tool = Tool("WikipediaSearch", search_wikipedia, "To search for information on wikipedia")
calculator_tool = Tool("Calculator", perform_calculation, "To perform math calculations")
date_request_tool = Tool("Date_of_today", date_of_today, "To get the date of today")
tools = [wikipedia_search_tool, calculator_tool, date_request_tool]

agentConfig = AgentConfig()
agentConfig.with_model_client(OpenAI(api_key=os.environ.get('OPENAI_API_KEY')))
agentConfig.with_tools(tools)
agentConfig.with_token_limit(5000)
agentConfig.with_max_interactions(5)
agentConfig.with_system_instructions(
    "Important! You are not good with math operations therefore you must to use a Calculator tool to perform math calculations."
)

agent = ReActExecutor(agentConfig)
agent.execute("What is the double of Jair Bolsonaro age?")
