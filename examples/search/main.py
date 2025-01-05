import os

from openai import OpenAI

from common import Agent, AgentConfig
from reactexecutor import ReActExecutor
from tools import people_search_tool, calculator_tool, today_tool

open_ai = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

main_agent = Agent(
    name="MainAgent",
    instructions=f"""
    You are a helpful assistant that assists the user in completing a task using multiple tools.
    """,
    functions=[people_search_tool, calculator_tool, today_tool]
)


if __name__ == "__main__":
    query = "What is the double of Linus Torvalds age?"
    agent_config = AgentConfig()
    agent_config.with_model_client(open_ai)
    agent_config.with_token_limit(5000)
    agent_config.with_max_interactions(5)
    react_exec = ReActExecutor(agent_config, main_agent)
    react_exec.execute(query)


