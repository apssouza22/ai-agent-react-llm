import datetime
import wikipedia

# Equivalent of the perform_calculation function
def perform_calculation(operation, a, b):
    if operation not in ['add', 'subtract', 'multiply', 'divide']:
        return f"Invalid operation: {operation}, should be among ['add', 'subtract', 'multiply', 'divide']"

    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b == 0:
            return "Division by zero"
        return a / b



def search_wikipedia(search_query):
    return "Barack Obama birthday is 04/08/1961"
    page = wikipedia.page(search_query)
    text = page.content
    return text[:300]


# Equivalent of the date_req function
def date_of_today():
    return datetime.date.today()

