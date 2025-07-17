from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from .sub_agents.news_analyst.agent import news_analyst
from .sub_agents.reddit_sum.agent import reddit_scrapper
from .tools.tools import get_current_time

root_agent = Agent(
    name = "manager",
    model="gemini-2.0-flash",
    description = "Manager Agent",
    instruction = '''
    You are a manager agent that is responsible for overseeing the work of the other agents.

    Always delegate the task to the appropriate agent. Use your best judgement 
    to determine which agent to delegate to.

    You are responsible for delegating tasks to the following agent:
    - reddit_scrapper

    You also have access to the following tools:
    - news_analyst
    - get_current_time
    ''',
    sub_agents = [reddit_scrapper],
    tools = [
        AgentTool(news_analyst),
        get_current_time
    ]
)
