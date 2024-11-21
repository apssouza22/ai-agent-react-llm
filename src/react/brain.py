from typing import Optional, Type
from pydantic import BaseModel, Field
from react.config import AgentConfig


class Brain:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.messages = []
        self.token_count = 0

    def remember(self, message):
        self.messages.append(message)
        self.token_count += len(message)

        # Check if token_count exceeds the limit
        while self.token_count > self.config.token_limit and len(self.messages) > 1:
            # Remove messages from the end until token_count is within the limit
            removed_message = self.messages.pop()
            self.token_count -= len(removed_message)

    def recall(self):
        return "\n".join(self.messages)

    def think(self, prompt: str, output_format: Optional[Type[BaseModel]] = None):
        messages = [
            self._get_system_instructions(),
            {
                "role": "user",
                "content": prompt,
            }
        ]
        open_ai_params = {
            # model="gpt-4o-2024-08-06",
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 0.1,
            "max_tokens": 1000,
            "messages": messages,
        }
        if output_format:
            open_ai_params["response_format"] = output_format
            completion = self.config.model.beta.chat.completions.parse(
                **open_ai_params
            )
            return completion.choices[0].message.parsed

        response = self.config.model.chat.completions.create(**open_ai_params)
        return response.choices[0].message.content

    def _get_system_instructions(self):
        return {
            "role": "system",
            "content": f"""You are a helpful assistant that assists the user in completing a task. Don't ask for user input. 
Important! You don't know the date of today, therefore you must use the Date_of_today tool to get the date of today.
\n {self.config.system_instructions} \n         
Given the following information from the context history 
provide for the user's task using only this information.""",
        }


class ReactEnd(BaseModel):
    stop: bool = Field(..., description="True if the context is enough to answer the request else False")
    final_answer: str = Field(..., description="Final answer if the context is enough to answer the request")
    confidence: float = Field(..., description="Confidence score of the final answer")

