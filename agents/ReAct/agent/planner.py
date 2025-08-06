from typing import List, Union
from ..llm.llm import OpenAIClient, AnthropicClient
from ..prompt.planner_model import Plan
from ..state.state import State
from ..prompt.utils import load_prompt


class Planner:
    def __init__(
        self,
        llm_client: Union[OpenAIClient, AnthropicClient],
        max_step_num: int = 3,
        locale: str = "en-US",
    ):
        self.system_prompt = load_prompt(
            "planner", {"max_step_num": max_step_num, "locale": locale}
        )
        self.llm_client = llm_client

    def plan(self, query: str, state: State = None) -> List[str]:
        response = self.llm_client.generate(query, self.system_prompt, state)
        return Plan.model_validate_json(response)