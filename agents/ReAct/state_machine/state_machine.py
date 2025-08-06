from enum import Enum, auto
from typing import Union
from ..agent.planner import Planner
from ..state.state import State
from ..llm.llm import OpenAIClient, AnthropicClient
from ..agent.researcher import Researcher
from ..agent.coder import Coder
from ..agent.reporter import Reporter


class Node(Enum):
    """Enumeration of possible states in the workflow."""

    PLANNER = auto()
    RESEARCH_TEAM = auto()
    RESEARCH = auto()
    CODE = auto()
    REPORTER = auto()
    END = auto()  

class StateMachine:
    """Implementation of a state machine with predefined transitions."""

    def __init__(
        self,
        human_query: str,
        llm_client: Union[OpenAIClient, AnthropicClient],
    ):
        self.transitions = {
            Node.PLANNER: [Node.RESEARCH_TEAM, Node.REPORTER],
            Node.RESEARCH_TEAM: [Node.RESEARCH, Node.CODE],
            Node.RESEARCH: [Node.RESEARCH_TEAM],
            Node.CODE: [Node.RESEARCH_TEAM],
            Node.REPORTER: [Node.END],
            Node.END: [],  
        }

        self.state_actions = {
            Node.PLANNER: self._planner_action,
            Node.RESEARCH_TEAM: self._research_team_action,
            Node.RESEARCH: self._research_action,
            Node.CODE: self._code_action,
            Node.REPORTER: self._reporter_action,
        }

        self.current_node = Node.PLANNER
        self.planner_agent = Planner(llm_client)
        self.researcher = Researcher(llm_client)
        self.coder = Coder(llm_client)
        self.reporter = Reporter(llm_client)

        messages = [{"role": "user", "content": human_query}]
        self.state = State()
        self.state.set("messages", messages)
        self.max_plan_iters = 2
        self.plan_iter = 0

    def _planner_action(self) -> None:
        """Action performed when in PLANNER state."""
        plan = self.planner_agent.plan("", self.state)
        self.plan_iter += 1
        self.state.set("current_plan", plan)
        if self.plan_iter > self.max_plan_iters:
            return Node.REPORTER
        if plan.has_enough_context:
            return Node.REPORTER
        else:
            return Node.RESEARCH_TEAM

    def _research_team_action(self) -> None:
        """
        Loop through the steps in the current plan and perform the action for each step by delegating to the research agent or code agent
        """
        current_plan = self.state.get("current_plan")

        if all(step.execution_res for step in current_plan.steps):
            return Node.REPORTER
        for step in current_plan.steps:
            if not step.execution_res:
                break
        if step.step_type == "research":
            next_node = Node.RESEARCH
        elif step.step_type == "processing":
            next_node = Node.CODE
        else:
            raise ValueError(f"Invalid step type: {step.step_type}")
        return next_node

    def _research_action(self) -> None:
        """Action performed when in RESEARCH state."""

        current_plan = self.state.get("current_plan")
        for step in current_plan.steps:
            if not step.execution_res:
                break
        input = f"#Task\n\n##title\n\n{step.title}\n\n##description\n\n{step.description}\n\n##locale\n\n{self.state.get('locale', 'en-US')}"
        res = self.researcher.research(input, self.state)
        step.execution_res = res['messages'][-1]
        return Node.RESEARCH_TEAM

    def _code_action(self) -> None:
        """Action performed when in CODE state."""

        current_plan = self.state.get("current_plan")
        for step in current_plan.steps:
            if not step.execution_res:
                break
        input = f"#Task\n\n##title\n\n{step.title}\n\n##description\n\n{step.description}\n\n##locale\n\n{self.state.get('locale', 'en-US')}"
        res = self.coder.code(input, self.state)
        step.execution_res = res['messages'][-1]
        return Node.RESEARCH_TEAM

    def _reporter_action(self) -> None:
        report = self.reporter.report(self.state)
        self.state.set("report", report)
        return Node.END

    def step(self) -> bool:
        return self.state_actions[self.current_node]()

    def run_until_end(self) -> None:
        """
        Run the state machine until it reaches the END state or max_steps is reached.

        Args:
            max_steps: Maximum number of steps to prevent infinite loops.
        """
        while self.current_node != Node.END:
            next_node = self.step()
            self.current_node = next_node
        return self.state.get("report")
