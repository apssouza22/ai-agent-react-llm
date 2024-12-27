import os
from openai import OpenAI
open_ai = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

if __name__ == "__main__":
    query = "What is the double of Linus Torvalds age?"
    print("Query:", query)

