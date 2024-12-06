from common.agent_base import Agent

people_search_agent = Agent(
    name="People_search_Agent",
    instructions=f"""You are a helpful assistant that help to find people information on the Wikipedia using its name. 
Important! Make sure you are returning info for the right person.""",
)

main_agent = Agent(
    name="MultiToolAgent",
    instructions=f"""
    You are a helpful assistant that assists the user in completing a task using multiple tools.
Important! You are bad at math operations therefore you MUST to use the provided Calculator tool to perform math calculations. 
Ex. use the Calculator tool to multiply 2 by 3.
""",
)
