# backend/adk_ticket_agent.py
import os
import re
import asyncio
from typing import Optional, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

# -----------------------
# Env & API key handling
# -----------------------
load_dotenv()

# ADK expects GOOGLE_API_KEY for Gemini.
# Bridge GEMINI_API_KEY -> GOOGLE_API_KEY if needed
if "GOOGLE_API_KEY" not in os.environ and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

APP_NAME = "ticketing_mas"
GEMINI_MODEL = "gemini-2.5-pro"  # keep your verified model as default

# -----------------------
# Define a simple ticket triage agent (default)
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

# Global runner for the default agent/model (keeps backward compatibility)
runner = InMemoryRunner(agent=ticket_agent, app_name=APP_NAME)


# -----------------------
# Internal helper: sanitize model name -> valid identifier fragment
# -----------------------
def _sanitize_model_name_for_agent(model_name: str) -> str:
    """
    Turn arbitrary model name (e.g. 'gemini-2.5-flash') into a valid identifier:
      - only letters, digits, underscores
      - must start with letter or underscore
      - limited length to avoid extremely long names
    """
    if not isinstance(model_name, str):
        model_name = str(model_name or "")
    # replace non-alnum with underscore
    sanitized = re.sub(r'[^0-9A-Za-z_]', '_', model_name)
    # ensure starts with letter/underscore
    if not re.match(r'^[A-Za-z_]', sanitized):
        sanitized = 'm_' + sanitized
    # collapse repeated underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # trim
    return sanitized[:60]


# -----------------------
# Internal helper to build an agent/runner for arbitrary model
# -----------------------
def _make_agent_and_runner(model_name: str) -> Tuple[LlmAgent, InMemoryRunner]:
    """
    Build an LlmAgent + runner for the requested model name.
    Used when caller asks for a different model than default.
    Agent name is sanitized to obey ADK's identifier rules.
    """
    safe_fragment = _sanitize_model_name_for_agent(model_name)
    agent_name = f"ticket_triage_agent_{safe_fragment}"

    agent = LlmAgent(
        model=model_name,
        name=agent_name,
        description="Ticket triage agent (dynamic model).",
        instruction=ticket_agent.instruction,  # reuse same instruction
    )
    return agent, InMemoryRunner(agent=agent, app_name=APP_NAME)


# -----------------------
# Core helper: run agent once (async)
# -----------------------
async def run_ticket_agent_once(
    prompt: str,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: str = "demo-user",
) -> str:
    """
    Run the ticket agent once and return final response text.

    - Default behavior (no model passed): reuse global runner which uses GEMINI_MODEL.
    - If a different model is passed, create a temporary agent+runner for that model.
    - If session_id is None: generate a unique session id (prevents collisions).
      If session_id is provided: reuse it (preserves session history).
    """
    chosen_model = model or GEMINI_MODEL
    use_global_runner = chosen_model == GEMINI_MODEL

    # Decide runner & session id
    if use_global_runner:
        active_runner = runner
    else:
        # create temporary agent & runner for requested model
        _, active_runner = _make_agent_and_runner(chosen_model)

    # Unique session id by default to prevent "AlreadyExistsError".
    session_id = session_id or f"ticket-session-{uuid4().hex}"

    # create session (ignore if session already exists)
    try:
        await active_runner.session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except Exception:
        # non-fatal on create_session (session may already exist)
        pass

    user_content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    final_text: Optional[str] = None

    async for event in active_runner.run_async(
        user_id=user_id, session_id=session_id, new_message=user_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
            if texts:
                final_text = "\n".join(texts).strip()

    return final_text or ""


def run_ticket_agent(prompt: str, model: Optional[str] = None, session_id: Optional[str] = None) -> str:
    """
    Synchronous wrapper.

    - Keep signature backward-compatible: existing callers using run_ticket_agent(prompt)
      will keep working (uses GEMINI_MODEL and unique session id per call).
    - Callers who want persistent session or a specific model can pass them.
    """
    return asyncio.run(run_ticket_agent_once(prompt, model=model, session_id=session_id))


# -----------------------
# Local manual test
# -----------------------
if __name__ == "__main__":
    test_prompt = (
        "User cannot access the payroll portal. "
        "They get 'Access denied' even though they logged in yesterday."
    )
    print(">>> Sending test prompt to ADK ticket agent (default model)...\n")
    response = run_ticket_agent(test_prompt)
    print("=== Agent response (default) ===")
    print(response)

    # Example: run with explicit model and persistent session
    print("\n>>> Sending test prompt to ADK ticket agent (explicit model + session reuse)...\n")
    s = "persistent-session-demo"
    resp1 = run_ticket_agent("Please classify this ticket: forgot password and can't login", model=GEMINI_MODEL, session_id=s)
    resp2 = run_ticket_agent("Follow-up: user reset password but still cannot login", model=GEMINI_MODEL, session_id=s)
    print("Resp1:", resp1[:200])
    print("Resp2:", resp2[:200])
