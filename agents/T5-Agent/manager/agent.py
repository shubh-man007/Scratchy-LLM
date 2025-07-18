from google.adk.agents import Agent

from .tools.model import generate_sql_query

root_agent = Agent(
    name = "manager",
    model = "gemini-2.0-flash",
    description = "An agent that takes performs Text to SQL conversions",
    tools = [generate_sql_query],
)
