from ..llm.llm import OpenAIClient, AnthropicClient
from ..state.state import State
from typing import Union
from ..agent.react_agent import ReactAgent
from ..tools.python_repl import PythonREPLTool


class Coder:
    def __init__(self, llm_client: Union[OpenAIClient, AnthropicClient]):
        coder_tools = [PythonREPLTool()]
        coder_config = {"locale": "en-US"}
        self.coder_agent = ReactAgent(
            name="coder",
            llm_client=llm_client,
            tools=coder_tools,
            config=coder_config,
        )

    def code(self, query: str, state: State) -> str:
        """
        query: the specific task step description
        state: the state of the workflow
        """
        res = self.coder_agent.run(query, state)
        return res