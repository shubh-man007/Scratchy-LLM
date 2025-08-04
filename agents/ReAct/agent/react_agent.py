# ReAct paper: https://arxiv.org/abs/2210.03629
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from ..tools import Tool
from ..prompt.utils import load_prompt
from ..state.state import State


@dataclass
class AgentState:
    """State object for the React agent."""

    # Track conversation history
    messages: List[Dict[str, str]] = field(default_factory=list)
    # Track intermediate steps (action, observation pairs)
    intermediate_steps: List[Dict[str, str]] = field(default_factory=list)
    # Current action details
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    # Flag to determine if the agent is done
    is_done: bool = False


class ReactAgent:
    """Implementation of a React agent that reasons and acts in cycles."""

    def __init__(
        self, name: str, llm_client, tools: List[Tool], config: Dict[str, Any]
    ):
        """Initialize the React agent.

        Args:
            llm_client: Client for LLM API
            tools: List of tools available to the agent
            prompt: The prompt for the agent
        """
        self.name = name
        self.llm_client = llm_client
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        tool_config = [
            {"name": tool.name, "description": tool.description} for tool in tools
        ]
        config["tools"] = tool_config
        self.prompt = load_prompt(name, config)

    def _create_prompt(
        self, agent_input: str, intermediate_steps: List[Dict[str, str]]
    ) -> str:
        """Create the prompt for the LLM including the React format instructions.

        Args:
            agent_input: The input to the agent
            intermediate_steps: List of previous steps taken

        Returns:
            Complete prompt string for the LLM
        """
        # Tools section
        tool_descriptions = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in self.tools]
        )

        # Format intermediate steps for prompt
        steps_text = ""
        for step in intermediate_steps:
            steps_text += f"Thought: {step.get('thought', '')}\n"
            steps_text += f"Action: {step.get('action', '')}\n"
            steps_text += f"Action Input: {step.get('action_input', '')}\n"
            steps_text += f"Observation: {step.get('observation', '')}\n"

        # Construct the full prompt
        prompt = f"""{self.prompt}
Human query: {agent_input}
Follow this format:
Thought: Think about the current situation and what to do
Action: The action to take (must be one of: {', '.join([tool.name for tool in self.tools])})
Action Input: The input to the action (can be a string or a JSON object)
Observation: The result of the action
... (this Thought/Action/Observation cycle can repeat multiple times)
Thought: I now know the final answer
Final Answer: The final answer to the original input question

{steps_text}
"""
        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured components.

        Args:
            response: Raw text response from the LLM

        Returns:
            Dictionary containing parsed components (thought, action, action_input, etc.)
        """
        result = {}

        # Check if the response contains a final answer
        if "Final Answer:" in response:
            final_answer_idx = response.find("Final Answer:")
            # Extract everything after "Final Answer:"
            final_answer = response[final_answer_idx + len("Final Answer:") :].strip()
            result["final_answer"] = final_answer
            # Extract the last thought before final answer
            thought_match = response[:final_answer_idx].strip()
            if "Thought:" in thought_match:
                last_thought_idx = thought_match.rfind("Thought:")
                result["thought"] = thought_match[
                    last_thought_idx + len("Thought:") :
                ].strip()
        else:
            # Extract Thought
            if "Thought:" in response:
                thought_idx = response.find("Thought:")
                action_idx = response.find("Action:")
                if action_idx > thought_idx:
                    result["thought"] = response[
                        thought_idx + len("Thought:") : action_idx
                    ].strip()

            # Extract Action
            if "Action:" in response:
                action_idx = response.find("Action:")
                action_input_idx = response.find("Action Input:")
                if action_input_idx > action_idx:
                    result["action"] = response[
                        action_idx + len("Action:") : action_input_idx
                    ].strip()

            # Extract Action Input
            if "Action Input:" in response:
                action_input_idx = response.find("Action Input:")
                observation_idx = response.find("Observation:")

                # If observation isn't present yet (expecting it to be filled later)
                if observation_idx == -1:
                    action_input_text = (
                        response[action_input_idx + len("Action Input:") :]
                        .strip()
                        .strip('"')
                    )
                else:
                    action_input_text = (
                        response[
                            action_input_idx + len("Action Input:") : observation_idx
                        ]
                        .strip()
                        .strip('"')
                    )

                # Try to parse as JSON if it looks like JSON
                if (
                    action_input_text.startswith("{")
                    and action_input_text.endswith("}")
                ) or (
                    action_input_text.startswith("[")
                    and action_input_text.endswith("]")
                ):
                    try:
                        result["action_input"] = json.loads(action_input_text)
                    except json.JSONDecodeError:
                        result["action_input"] = action_input_text
                else:
                    result["action_input"] = action_input_text

        return result

    def _call_llm(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """Call the LLM with the given prompt.

        This is an abstract method that should be implemented based on the specific LLM client.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Raw text response from the LLM
        """
        # add stop to avoid LLM generating the observation as we want to use the tool
        response = self.llm_client.generate(
            prompt, system_prompt=self.prompt, stop=["Observation:"]
        )
        return response

    def _execute_tool(self, action: str, action_input: Any) -> str:
        """Execute a tool with the provided input.

        Args:
            action: Name of the tool to execute
            action_input: Input to pass to the tool

        Returns:
            Result from the tool execution as a string
        """
        if action not in self.tool_map:
            return f"Error: Tool '{action}' not found. Available tools: {', '.join(self.tool_map.keys())}"

        try:
            tool_result = self.tool_map[action](action_input)
            return tool_result
        except Exception as e:
            return f"Error executing tool '{action}': {str(e)}"

    def run(
        self, agent_input: str, workflow_state: State, max_iterations: int = 10
    ) -> Dict[str, Any]:
        """Run the React agent on a user query.

        Args:
            agent_input: the input to the agent
            max_iterations: Maximum number of thought-action-observation cycles

        Returns:
            Complete result with all intermediate steps and final answer
        """
        agent_state = AgentState()
        agent_state.messages.append({"role": "user", "content": agent_input})
        observations = workflow_state.get("observations", [])

        # Main React loop
        for i in range(max_iterations):
            # Create prompt with current state
            prompt = self._create_prompt(agent_input, agent_state.intermediate_steps)

            # Get response from LLM
            llm_response = self._call_llm(prompt)

            # Parse LLM response
            parsed_response = self._parse_llm_response(llm_response)

            # Check if we have a final answer
            if "final_answer" in parsed_response:
                agent_state.is_done = True
                agent_state.messages.append(
                    {"role": "assistant", "content": parsed_response["final_answer"]}
                )
                break

            # Extract thought, action, action_input
            thought = parsed_response.get("thought", "")
            action = parsed_response.get("action", "")
            action_input = parsed_response.get("action_input", "")

            # Execute tool
            observation = self._execute_tool(action, action_input)
            print(f"thought: {thought}")
            print(f"action: {action}")
            print(f"action_input: {action_input}")
            print(f"observation: {observation}")
            print(f"react agent iter: {i}, max_iterations: {max_iterations}")
            print("--------------------------------")
            observations.append(observation)
            # Record step
            step = {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            }
            agent_state.intermediate_steps.append(step)

        # If we've hit max iterations without completion
        if not agent_state.is_done:
            agent_state.messages.append(
                {
                    "role": "assistant",
                    "content": "I was unable to complete the task within the maximum number of iterations.",
                }
            )

        # Prepare result
        result = {
            "messages": agent_state.messages,
            "intermediate_steps": agent_state.intermediate_steps,
            "is_complete": agent_state.is_done,
        }
        workflow_state.set("observations", observations)
        messages = workflow_state.get("messages", [])
        messages.append(agent_state.messages[-1])
        workflow_state.set("messages", messages)
        return result
