import os
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any


def load_prompt(agent_name: str, config: Dict[str, Any]) -> str:
    template_dir = os.path.join(os.path.dirname(__file__))
    env = Environment(loader=FileSystemLoader(template_dir))

    # Load and render the template
    template = env.get_template(f"{agent_name}.md")
    return template.render(**config)
