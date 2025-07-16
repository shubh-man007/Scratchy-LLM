# import asyncio
# from dataclasses import dataclass, field
# from typing import Union, cast

# import anthropic
# from anthropic.types import MessageParam, TextBlock, ToolUnionParam, ToolUseBlock
# from dotenv import load_dotenv
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client

# load_dotenv()


# anthropic_client = anthropic.AsyncAnthropic()


# # Create server parameters for stdio connection
# server_params = StdioServerParameters(
#     command="python",  # Executable
#     args=["./mcp_server.py"],  # Optional command line arguments
#     env=None,  # Optional environment variables
# )


# @dataclass
# class Chat:
#     messages: list[MessageParam] = field(default_factory=list)

#     system_prompt: str = """You are a master SQLite assistant. 
#     Your job is to use the tools at your dispoal to execute SQL queries and provide the results to the user."""

#     async def process_query(self, session: ClientSession, query: str) -> None:
#         response = await session.list_tools()
#         available_tools: list[ToolUnionParam] = [
#             {
#                 "name": tool.name,
#                 "description": tool.description or "",
#                 "input_schema": tool.inputSchema,
#             }
#             for tool in response.tools
#         ]

#         # Initial Claude API call
#         res = await anthropic_client.messages.create(
#             model="claude-3-5-sonnet-latest",
#             system=self.system_prompt,
#             max_tokens=8000,
#             messages=self.messages,
#             tools=available_tools,
#         )

#         assistant_message_content: list[Union[ToolUseBlock, TextBlock]] = []
#         for content in res.content:
#             if content.type == "text":
#                 assistant_message_content.append(content)
#                 print(content.text)
#             elif content.type == "tool_use":
#                 tool_name = content.name
#                 tool_args = content.input

#                 result = await session.call_tool(tool_name, cast(dict, tool_args))

#                 assistant_message_content.append(content)
#                 self.messages.append(
#                     {"role": "assistant", "content": assistant_message_content}
#                 )
#                 self.messages.append(
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "tool_result",
#                                 "tool_use_id": content.id,
#                                 "content": getattr(result.content[0], "text", ""),
#                             }
#                         ],
#                     }
#                 )
#                 # Get next response from Claude
#                 res = await anthropic_client.messages.create(
#                     model="claude-3-7-sonnet-latest",
#                     max_tokens=8000,
#                     messages=self.messages,
#                     tools=available_tools,
#                 )
#                 self.messages.append(
#                     {
#                         "role": "assistant",
#                         "content": getattr(res.content[0], "text", ""),
#                     }
#                 )
#                 print(getattr(res.content[0], "text", ""))


#     async def question_decompose(self, query:str):
#         decomposition_prompt : str =  '''Decompose the following question into simpler subquestions that can each be answered using a SQL query.
#         Respond with a numbered list of subquestions only.'''      

#         messages = {"role": "user", "content": f"{decomposition_prompt}\nThis is the user quer:\n{query}"}    
#         res = await anthropic_client.messages.create(
#             model = "claude-3-7-sonnet-latest",
#             max_tokens = 1000,
#             system = "You are a helpful SQL Assistant, who specializes in breaking down complex queries into simpler ones.",
#             messages = messages
#         )

#         content_text = res.content[0].text
#         return [line.split(". ", 1)[1] for line in content_text.strip().split("\n") if ". " in line]
    

#     async def compose_answer(self, query:str, subquestions:list[str], subanswers:list[str]):
#         merge_prompt = '''Given the following original question, subquestions, and their answers, summarize the final answer.'''

#         content = (
#         f"Original Question: {query}\n\n"
#         + "\n".join([f"Subquestion {i+1}: {subquestions[i]}\nAnswer: {subanswers[i]}" for i in range(len(subquestions))])
#         )

#         messages = [{"role": "user", "content": f"{merge_prompt}\n\n{content}"}]
#         res = await anthropic_client.messages.create(
#             model = "claude-3-7-sonnet-latest",
#             max_tokens = 1000,

#         )


#     async def chat_loop(self, session: ClientSession):
#         while True:
#             query = input("\nQuery: ").strip()
#             self.messages.append(
#                 MessageParam(
#                     role="user",
#                     content=query,
#                 )
#             )

#             await self.process_query(session, query)


#     async def run(self):
#         async with stdio_client(server_params) as (read, write):
#             async with ClientSession(read, write) as session:
#                 # Initialize the connection
#                 await session.initialize()

#                 await self.chat_loop(session)


# chat = Chat()

# asyncio.run(chat.run())


# # uv run mcp_client.py


import asyncio
from dataclasses import dataclass, field
from typing import Union, cast

import anthropic
from anthropic.types import MessageParam, TextBlock, ToolUnionParam, ToolUseBlock
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


anthropic_client = anthropic.AsyncAnthropic()


# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["./mcp_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)


@dataclass
class Chat:
    messages: list[MessageParam] = field(default_factory=list)

    system_prompt: str = """You are a master SQLite assistant. 
    Your job is to use the tools at your dispoal to execute SQL queries and provide the results to the user."""

    async def process_query(self, session: ClientSession, query: str) -> None:
        response = await session.list_tools()
        available_tools: list[ToolUnionParam] = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        # Initial Claude API call
        res = await anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest",
            system=self.system_prompt,
            max_tokens=8000,
            messages=self.messages,
            tools=available_tools,
        )

        assistant_message_content: list[Union[ToolUseBlock, TextBlock]] = []
        for content in res.content:
            if content.type == "text":
                assistant_message_content.append(content)
                print(content.text)
            elif content.type == "tool_use":
                tool_name = content.name
                tool_args = content.input

                result = await session.call_tool(tool_name, cast(dict, tool_args))

                assistant_message_content.append(content)
                self.messages.append(
                    {"role": "assistant", "content": assistant_message_content}
                )
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": getattr(result.content[0], "text", ""),
                            }
                        ],
                    }
                )
                # Get next response from Claude
                res = await anthropic_client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=8000,
                    messages=self.messages,
                    tools=available_tools,
                )
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": getattr(res.content[0], "text", ""),
                    }
                )
                print(getattr(res.content[0], "text", ""))

    async def chat_loop(self, session: ClientSession):
        while True:
            query = input("\nQuery: ").strip()
            self.messages.append(
                MessageParam(
                    role="user",
                    content=query,
                )
            )

            await self.process_query(session, query)

    async def run(self):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                await self.chat_loop(session)


chat = Chat()

asyncio.run(chat.run())


# uv run mcp_client.py
