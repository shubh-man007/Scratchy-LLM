# import os
# from tavily import TavilyClient
# from .tools import Tool
# from typing import Dict

# from dotenv import load_dotenv
# load_dotenv()

# MAX_RESULTS = 2

# class TavilySearch(Tool):
#     def __init__(self, tool_name : str, tool_description : str):
#         super().__init__(tool_name, tool_description)
#         TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
#         self.client = TavilyClient(api_key=TAVILY_API_KEY)

#     def run(self, query : Dict[str, str]) -> Dict[str, str]:
#         tavily_response = self.client.search(query, max_results = MAX_RESULTS)
#         return {"tavily_response" : tavily_response}


import logging
import os
from tavily import TavilyClient
from .tools import Tool
from typing import Union, Dict

from dotenv import load_dotenv
load_dotenv()

MAX_RESULTS = 2
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

logger = logging.getLogger(__name__)


client = TavilyClient(TAVILY_API_KEY)


class TavilySearchTool(Tool):
    def __init__(self):
        super().__init__("search", "Useful for searching the web for information.")
        self.client = TavilyClient(TAVILY_API_KEY)

    def clean_results(self, response):
        results = response["results"]
        clean_results = []
        for result in results:
            clean_result = {}
            clean_result["title"] = result["title"]
            clean_result["url"] = result["url"]
            clean_result["content"] = result["content"]
            clean_result["score"] = result["score"]
            if result["raw_content"]:
                clean_result["raw_content"] = result["raw_content"]
            clean_results.append(clean_result)
        return clean_results

    def __call__(self, query: str) -> str:
        response = self.client.search(query=query, max_results=MAX_RESULTS)
        clean_results = self.clean_results(response)
        return ";".join(str(result) for result in clean_results)
    
    # According to our tool definition.
    def run(self, input: Union[str, Dict[str, str]]) -> Dict[str, str]:
        if isinstance(input, dict):
            query = input.get("query", "")
        else:
            query = input
        
        response = self.client.search(query=query, max_results=MAX_RESULTS)
        clean_results = self.clean_results(response)
        return {"results": clean_results, "query": query}
