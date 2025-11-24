# backend/adk_ticket_agent.py

import os
import asyncio
from typing import Optional

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

# -----------------------
# Env & API key handling
# -----------------------
load_dotenv()

# ADK expects GOOGLE_API_KEY for Gemini.
# We already use GEMINI_API_KEY elsewhere, so bridge them.
if "GOOGLE_API_KEY" not in os.environ and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

APP_NAME = "ticketing_mas"
GEMINI_MODEL = "gemini-2.5-pro"  # this is the one you just verified

# -----------------------
# Define a simple ticket triage agent
# -----------------------
ticket_agent = LlmAgent(
    model=GEMINI_MODEL,
    name="ticket_triage_agent",
    description=(
        "Helps triage and draft responses for ServiceNow-style support tickets."
    ),
    instruction=(
        "You are a ServiceNow-style ticket assistant.\n"
        "Input: a ticket description and (optionally) past resolution notes.\n"
        "Tasks:\n"
        "1) Classify the ticket into one of: 'access', 'incident', 'request', 'billing', 'other'.\n"
        "2) Suggest a short resolution plan (1–3 bullet points).\n"
        "3) Draft a short reply to the user (2–4 sentences) in simple language.\n"
        "Output format (very important):\n"
        "TicketType: <one of access/incidents/request/billing/other>\n"
        "Plan:\n"
        "- step 1\n"
        "- step 2\n"
        "Reply:\n"
        "<user-facing reply here>\n"
    ),
)

# In-memory runner for local testing & for later FastAPI integration
runner = InMemoryRunner(agent=ticket_agent, app_name=APP_NAME)


# -----------------------
# Core helper: run agent once
# -----------------------
async def run_ticket_agent_once(
    prompt: str,
    session_id: str = "demo-session",
    user_id: str = "demo-user",
) -> str:
    """
    Run the ticket agent for a single prompt.
    This uses ADK's InMemoryRunner + sessions, which we'll later reuse in the backend.
    """
    # Create or ensure a session exists
    # API is async in latest ADK, so we await it.
    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    final_text: Optional[str] = None

    # Consume the event stream and grab the final response text
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_content,
    ):
        # We only care about the final LLM response for now
        if event.is_final_response() and event.content and event.content.parts:
            # Concatenate all text parts (usually just one)
            texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
            if texts:
                final_text = "\n".join(texts).strip()

    return final_text or ""


def run_ticket_agent(prompt: str) -> str:
    """
    Synchronous wrapper so FastAPI or simple scripts can call the ADK agent easily.
    """
    return asyncio.run(run_ticket_agent_once(prompt))


# -----------------------
# Local manual test
# -----------------------
if __name__ == "__main__":
    test_prompt = (
        "User cannot access the payroll portal. "
        "They get 'Access denied' even though they logged in yesterday."
    )
    print(">>> Sending test prompt to ADK ticket agent...\n")
    response = run_ticket_agent(test_prompt)
    print("=== Agent response ===")
    print(response)
