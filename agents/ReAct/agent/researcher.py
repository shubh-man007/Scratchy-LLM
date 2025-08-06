from ..llm.llm import OpenAIClient, AnthropicClient
from ..state.state import State
from typing import Union
from ..agent.react_agent import ReactAgent
from ..tools.search import TavilySearchTool
from ..tools.crawler import CrawlerTool

class Researcher:
    def __init__(self, llm_client: Union[OpenAIClient, AnthropicClient]):
        researcher_tools = [TavilySearchTool(), CrawlerTool()]
        researcher_config = {"locale": "en-US"}
        self.research_agent = ReactAgent(
            name="researcher",
            llm_client=llm_client,
            tools=researcher_tools,
            config=researcher_config,
        )

    def research(self, query: str, state: State) -> str:
        """
        query: the specific task step description
        state: the state of the workflow
        """
        res = self.research_agent.run(query, state)
        return res
