from ..state import State
from openai import OpenAI
from anthropic import Anthropic
from typing import Union, List, Dict
import os


def MessageGPT(query: Union[str, List[Dict[str, str]]], 
               system_prompt : str =  "", 
               state : State = None) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role" : "system", "content" : system_prompt})
    if isinstance(query, List):
        messages.extend(query)
    elif query:
        messages.append({"role": "user", "content": query})
    if state:
        for message in state.get("messages", []):
            messages.append(message)
    return messages


def MessageClaude(query: Union[str, List[Dict[str, str]]], 
                  system_prompt : str =  "", 
                  state : State = None)-> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role" : "assistant", "content" : system_prompt})
    if isinstance(query, List):
        messages.extend(query)
    elif query:
        messages.append({"role": "user", "content": query})
    if state:
        for message in state.get("messages", []):
            messages.append(message)
    return messages


class OpenAIClient:
    def __init__(self, model : str = "gpt-4o"):
        OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=OPEN_AI_KEY)
        self.model = model

    def invoke(
            self, 
            query : Union[str, List[Dict[str, str]]], 
            system_prompt : str = "", 
            state : State = None, 
            stop : List[str] = []) -> str:
        message = MessageGPT(query, system_prompt, state)
        response = self.client.chat.completions.create(model=self.model, messages=message, stop=stop)
        return response.choices[0].message.content
    
    def generate(self, query: Union[str, List[Dict[str, str]]], system_prompt: str = "", state: State = None, stop: List[str] = []) -> str:
        """Alias for invoke method for compatibility."""
        return self.invoke(query, system_prompt, state, stop)
    

class AnthropicClient:
    def __init__(self, model : str = "claude-3-sonnet-20240229"):
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = model

    def generate(self, 
               query : Union[str, List[Dict[str, str]]], 
               system_prompt : str = "", 
               state : State = None, 
               stop : List[str] = []) -> str:
        message = MessageClaude(query, system_prompt, state)
        response = self.client.messages.create(model=self.model, max_tokens=2000, messages=message, stop=stop)
        return response.content[0].text


def get_client(model_name: str):
    if "gpt" in model_name.lower():
        return OpenAIClient(model_name)
    elif "claude" in model_name.lower():
        return AnthropicClient(model_name)
    else:
        return OpenAIClient(model_name)
