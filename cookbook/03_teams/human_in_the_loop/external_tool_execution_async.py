"""Team HITL: Async member agent tool with external execution.

Same as external_tool_execution.py but uses async run/continue_run.
"""

import asyncio

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from agno.tools import tool


@tool(external_execution=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to someone. Executed externally.

    Args:
        to (str): The recipient email address
        subject (str): The email subject
        body (str): The email body
    """
    return ""


email_agent = Agent(
    name="EmailAgent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[send_email],
)

team = Team(
    name="CommunicationTeam",
    model=OpenAIChat(id="gpt-4o"),
    members=[email_agent],
)


async def main():
    response = await team.arun(
        "Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'Let's meet at 3pm.'"
    )

    if response.is_paused:
        print("Team paused - external execution needed")
        for req in response.requirements:
            if req.needs_external_execution:
                print(f"  Tool: {req.tool_execution.tool_name}")
                print(f"  Args: {req.tool_execution.tool_args}")
                # Simulate executing the tool externally
                req.set_external_execution_result("Email sent successfully")

        response = await team.acontinue_run(response)
        print(f"Result: {response.content}")
    else:
        print(f"Result: {response.content}")


asyncio.run(main())
