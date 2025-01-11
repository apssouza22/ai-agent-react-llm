def escalate_to_human(reason=None):
    return f"Escalating to agent: {reason}" if reason else "Escalating to agent"


def case_resolved():
    return "Case resolved. No further questions."


def initiate_baggage_search():
    return "Baggage was found!"
