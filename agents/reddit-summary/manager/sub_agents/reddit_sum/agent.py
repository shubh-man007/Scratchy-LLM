import os
import json
from dotenv import load_dotenv
import praw

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext

load_dotenv()
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')
AGENT_NAME os.getenv('AGENT_NAME')

# <----Initialize PRAW---->
reddit = praw.Reddit(client_id = CLIENT_ID,
                     client_secret = CLIENT_SECRET,
                     username = USERNAME,
                     password = PASSWORD,
                     user_agent = AGENT_NAME)


def get_reddit_posts(sub_name:str, tool_context: ToolContext, lim:int = 5) -> dict:
    '''Retrieves a given amount of posts from a given sub-reddit using the Python Reddit API Wrapper (PRAW).

    Args:
        sub_name (str): Name of the sub-reddit from which the posts are to be extracted.
        tool_context (ToolContext): To store or share persistent state, by taking in sub_name and the posts associated.
        lim (int): How many posts are to be extracted, by default 10.

    Returns:
        dict: status, name of the sub-reddit and the posts extracted from the sub-reddit.
    '''
    try:
        subreddit = reddit.subreddit(sub_name)
        posts_data = []
        for post in subreddit.hot(limit=lim):
            post_info = {
                "title": post.title,
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "text_snippet": post.selftext if post.selftext else ""
            }
            posts_data.append(post_info)
        
        tool_context.state["last_subreddit"] = sub_name
        tool_context.state["last_reddit_summary"] = posts_data
        return {
            "status": "success", 
            "subreddit": sub_name, 
            "submissions": posts_data
        }
    
    except Exception as e:
        return {
            "status": "error",
            "subreddit": sub_name,
            "error_message": str(e)
        }



reddit_scrapper = Agent(
    name="reddit_sum",
    model="gemini-2.0-flash",
    description="An agent that summarizes reddit posts from a specified sub-reddit.",
    instruction="""
        You are a Reddit insight agent that retrieves and summarizes the current mood and top discussions in a given subreddit.

        When asked about what's happening in a subreddit:
        1. Use the `get_reddit_posts` tool to fetch the most recent popular posts.
        2. If the user specifies a subreddit name, pass it directly as the `sub_name` argument.
        3. If the user doesn't mention a subreddit, politely ask them to specify one.
        4. Optionally, allow users to change how many posts are retrieved by setting the `lim` argument.
        5. Store the subreddit name and fetched post data using the provided `ToolContext` for future use by other tools.
        6. If the tool returns an error (e.g., a 403 Forbidden error), inform the user that you were unable to access that subreddit and suggest checking if it’s private or restricted.

        Each post will include:
        - Title
        - Score (how popular the post is)
        - Upvote ratio (the general positivity of feedback)
        - The post content

        When presenting the results:
        - Give a brief summary of the subreddit’s current top posts (go through all the posts and summarize them).
        - Highlight any trends you notice (e.g., high scores, common topics).
        - Keep the tone informative and neutral, suitable for someone researching or browsing subreddits.

        Example response format:
        "Here are the top discussions from r/<subreddit>:
        1. <Title 1> (Score: X, Upvotes: Y%)
        2. <Title 2> (Score: X, Upvotes: Y%)
        ...

        Let me know if you’d like a summary of another subreddit or more posts!"

        If you receive an error, respond with:
        "I'm unable to access r/<subreddit>. It might be private, banned, or restricted. Please check the subreddit name or try another one."

        If the user asks something unrelated to Reddit or subreddit summaries, you should delegate the task to the manager agent.
        """,
    tools=[get_reddit_posts],
)
