# backend/adk_domain_agent.py

import os
import asyncio
import json
from typing import Optional, Dict, Any

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
GEMINI_MODEL = "gemini-2.5-pro"  # domain classifier uses PRO for better reasoning

# -----------------------
# Define the domain classifier agent
# -----------------------
domain_classifier_agent = LlmAgent(
    model=GEMINI_MODEL,
    name="domain_classifier_agent",
    description=(
        "Classifies IT/support tickets into domains and suggests which agent "
        "should handle them."
    ),
    instruction=(
        "You are a domain classifier for a multi-agent ticketing system.\n"
        "You receive a raw ticket description (user's text).\n\n"
        "You MUST respond ONLY with a single JSON object, no extra text, "
        "in the following format:\n\n"
        "{\n"
        '  \"domain\": \"access\" | \"billing\" | \"network\" | \"incident\" | \"request\" | \"other\",\n'
        "  \"confidence\": float between 0 and 1,\n"
        "  \"reason\": \"short explanation of why you chose this domain\",\n"
        "  \"summary\": \"one or two sentence summary of the ticket\",\n"
        "  \"suggested_agent\": \"access_agent\" | \"billing_agent\" | \"network_agent\" | \"generic_agent\"\n"
        "}\n\n"
        "Rules:\n"
        "- \"access\" covers login/auth/account access issues.\n"
        "- \"billing\" covers invoices, charges, payments, refunds.\n"
        "- \"network\" covers Wi-Fi, VPN, connectivity, latency.\n"
        "- \"incident\" covers unexpected outages or breakages not covered above.\n"
        "- \"request\" covers normal requests for features/access/resources.\n"
        "- If unsure, use domain=\"other\" and suggested_agent=\"generic_agent\".\n"
        "- Always ensure the JSON is valid and parseable.\n"
    ),
)

domain_runner = InMemoryRunner(agent=domain_classifier_agent, app_name=APP_NAME)


# -----------------------
# Core helper: run domain agent once
# -----------------------
async def _run_domain_classifier_once(
    description: str,
    session_id: str = "domain-session",
    user_id: str = "domain-user",
) -> Dict[str, Any]:
    """
    Run the domain classifier agent once and return a parsed dict:
    {
      "domain": str,
      "confidence": float,
      "reason": str,
      "summary": str,
      "suggested_agent": str,
      "raw_output": str
    }
    """
    # Ensure session exists
    await domain_runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    prompt = (
        "Classify the following ticket description into a support domain and "
        "follow the JSON-only response instructions.\n\n"
        f"Ticket Description:\n{description}\n"
    )

    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    final_text: Optional[str] = None

    async for event in domain_runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
            if texts:
                final_text = "\n".join(texts).strip()

    raw_output = final_text or ""

    # Try to parse JSON
    parsed: Dict[str, Any] = {
        "domain": "other",
        "confidence": 0.0,
        "reason": "",
        "summary": "",
        "suggested_agent": "generic_agent",
        "raw_output": raw_output,
    }

    if raw_output:
        try:
            obj = json.loads(raw_output)
            # Merge safely
            for key in ["domain", "confidence", "reason", "summary", "suggested_agent"]:
                if key in obj:
                    parsed[key] = obj[key]
        except json.JSONDecodeError:
            # If LLM messed up, keep defaults + raw_output
            pass

    return parsed


def run_domain_classifier(description: str) -> Dict[str, Any]:
    """
    Synchronous wrapper to call domain classifier from FastAPI or orchestrator.
    """
    return asyncio.run(_run_domain_classifier_once(description))


# -----------------------
# Local manual test
# -----------------------
if __name__ == "__main__":
    test_desc = "User can't connect to VPN and their internet keeps dropping in the office."
    print(">>> Sending test description to domain classifier...\n")
    result = run_domain_classifier(test_desc)
    print("=== Parsed result ===")
    for k, v in result.items():
        print(f"{k}: {v}")
