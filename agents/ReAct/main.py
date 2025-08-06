from .state_machine.state_machine import StateMachine
from .llm.llm import get_client
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4.1")
    args = parser.parse_args()
    state_machine = StateMachine(
        human_query=args.query,
        llm_client=get_client(args.model_name),
    )
    state_machine.run_until_end()
    print("Report:")
    print(state_machine.state.get("report"))
