"""Team HITL Async Streaming: Member agent tool with external execution.

Same as external_tool_execution_stream.py but uses async run/continue_run.

Note: When streaming with member agents, use isinstance() with TeamRunPausedEvent
to distinguish the team's pause from member agent pauses.
"""

import asyncio

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.run.team import RunPausedEvent as TeamRunPausedEvent
from agno.team.team import Team
from agno.tools import tool
from agno.utils import pprint

db = SqliteDb(db_file="tmp/team_hitl_stream.db")


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
    role="Handles email operations",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[send_email],
    db=db,
)

team = Team(
    name="CommunicationTeam",
    members=[email_agent],
    model=OpenAIChat(id="gpt-4o-mini"),
    db=db,
)


async def main():
    async for run_event in team.arun(
        "Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'Let's meet at 3pm.'",
        stream=True,
    ):
        # Use isinstance to check for team's pause event (not the member agent's)
        if isinstance(run_event, TeamRunPausedEvent):
            print("Team paused - requires external execution")
            for req in run_event.active_requirements:
                if req.needs_external_execution:
                    print(f"  Tool: {req.tool_execution.tool_name}")
                    print(f"  Args: {req.tool_execution.tool_args}")

                    # Simulate executing the tool externally
                    req.set_external_execution_result("Email sent successfully")

            # Use apprint_run_response for async streaming
            response = team.acontinue_run(
                run_id=run_event.run_id,
                session_id=run_event.session_id,
                requirements=run_event.requirements,
                stream=True,
            )
            await pprint.apprint_run_response(response)


if __name__ == "__main__":
    asyncio.run(main())
