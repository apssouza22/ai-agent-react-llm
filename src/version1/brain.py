from typing import Optional, Type

from pydantic import BaseModel

from common import AgentConfig, Agent


class Brain:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.messages = []

    def remember(self, message: str):
        self.messages.append(message)

    def recall(self) -> str:
        return "\n".join(self.messages)

    @staticmethod
    def _get_system_instructions():
        return {
            "role": "system",
            "content": f"""You are a helpful assistant that assists the user in completing a task. Don't ask for user input. 
Important! You don't know the date of today, therefore you must use the Date_of_today tool to get the date of today. 
Important! You don't know math, therefore you must use the Calculator tool for math operations       
Given the following information from the context history provide for the user's task using only this information.
"""
        }

    def think(self, prompt: str,agent: Agent, output_format: Optional[Type[BaseModel]] = None):
        messages = [
            self._get_system_instructions(),
            {
                "role": "system",
                "content": agent.instructions
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        openai_params = {
            "model": agent.model,
            "temperature": 0.1,
            "max_tokens": self.config.token_limit,
            "messages": messages
        }
        if output_format:
            openai_params["response_format"] = output_format
            completion = self.config.model.beta.chat.completions.parse(**openai_params)
            return completion.choices[0].message.parsed

        completion = self.config.model.chat.completions.create(**openai_params)
        return completion.choices[0].message.content
