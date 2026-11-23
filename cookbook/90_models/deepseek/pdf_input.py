"""
DeepSeek PDF Input
==================

Send a PDF to DeepSeek. DeepSeek only reads documents through its Files API, so the
`DeepSeek` model uploads each PDF (purpose "text_recognition") and references it by
`file_id`. The DeepSeek file id is stored on the `File`, so a file replayed from history
is not uploaded again.

Set `document_upload_purpose="rich_layout_recognition"` to keep figures as well as text.
"""

import asyncio

from agno.agent import Agent
from agno.media import File
from agno.models.deepseek import DeepSeek

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

agent = Agent(model=DeepSeek(id="deepseek-v4-flash"), markdown=True)

pdf = File(url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf")

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Sync ---
    agent.print_response("Suggest me a recipe from the attached file.", files=[pdf])

    # --- Async + Streaming ---
    asyncio.run(
        agent.aprint_response(
            "List three ingredients used in the recipes of the attached file.",
            files=[
                File(url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf")
            ],
            stream=True,
        )
    )
