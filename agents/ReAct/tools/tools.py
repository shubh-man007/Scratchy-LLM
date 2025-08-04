from abc import ABC, abstractmethod
from typing import Dict, Union


class Tool(ABC):
    def __init__(self, tool_name : str, tool_description : str):
        self.tool_name = tool_name
        self.tool_description = tool_description

    def get_tool_name(self) -> str:
        return self.tool_name
    
    def get_metadata(self) -> Dict[str, str]:
        return {"tool name" : self.tool_name, "tool_description" : self.tool_description}
    
    @abstractmethod
    def run(self, input : Union[str, Dict[str, str]]) -> Dict[str, str]:
        pass
        # Implemented _execute_tool method to execute tool locally.
