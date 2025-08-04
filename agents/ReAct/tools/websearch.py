import os
from tavily import TavilyClient
from tools import Tool
from typing import Dict

MAX_RESULTS = 2

class TavilySearch(Tool):
    def __init__(self, tool_name : str, tool_description : str):
        super().__init__(tool_name, tool_description)
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=TAVILY_API_KEY)

    def run(self, query : Dict[str, str]) -> Dict[str, str]:
        tavily_response = self.client.search(query, max_results = MAX_RESULTS)
        return {"tavily_response" : tavily_response}
