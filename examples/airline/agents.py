from common import Agent
from lost_baggage import STARTER_PROMPT, LOST_BAGGAGE_POLICY
from tools import escalate_to_human, initiate_baggage_search, case_resolved


def transfer_to_lost_baggage():
    return lost_baggage_agent

def transfer_to_triage():
    return triage_agent

lost_baggage_agent = Agent(
    name="Lost Baggage Agent",
    instructions=STARTER_PROMPT + LOST_BAGGAGE_POLICY,
    functions=[
        escalate_to_human,
        initiate_baggage_search,
        transfer_to_triage,
        case_resolved,
    ],
)

def triage_instructions(context_variables):
    customer_context = context_variables.get("customer_context", None)
    flight_context = context_variables.get("flight_context", None)
    return f"""You are to triage a users request, and call a tool to transfer to the right intent.
    Once you are ready to transfer to the right intent, call the tool to transfer to the right intent.
    You dont need to know specifics, just the topic of the request.
    When you need more information to triage the request to an agent, ask a direct question without explaining why you're asking it.
    Do not share your thought process with the user! Do not make unreasonable assumptions on behalf of user.
    The customer context is here: {customer_context}, and flight context is here: {flight_context}"""

triage_agent = Agent(
    name="Triage Agent",
    instructions=triage_instructions,
    functions=[transfer_to_lost_baggage],
)
