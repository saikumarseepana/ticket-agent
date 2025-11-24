# backend/adk_domain_agent.py
import os
import asyncio
import json
import logging
from typing import Optional, Dict, Any
from uuid import uuid4

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Env & API key handling
# -----------------------
load_dotenv()

# ADK expects GOOGLE_API_KEY for Gemini.
if "GOOGLE_API_KEY" not in os.environ and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

APP_NAME = "ticketing_mas"
GEMINI_MODEL = "gemini-2.5-pro"

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
# Helpers
# -----------------------
def _clean_and_parse_json(raw_output: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {
        "domain": "other",
        "confidence": 0.0,
        "reason": "",
        "summary": "",
        "suggested_agent": "generic_agent",
        "raw_output": raw_output or "",
    }

    if not raw_output:
        return parsed

    cleaned = raw_output.strip()

    # Strip triple backticks and optional leading language token like "json"
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    # Try JSON parse
    try:
        obj = json.loads(cleaned)
        parsed["domain"] = obj.get("domain", parsed["domain"])
        parsed["confidence"] = float(obj.get("confidence", parsed["confidence"]) or 0.0)
        parsed["reason"] = obj.get("reason", parsed["reason"])
        parsed["summary"] = obj.get("summary", parsed["summary"])
        parsed["suggested_agent"] = obj.get("suggested_agent", parsed["suggested_agent"])
        parsed["raw_output"] = cleaned
    except json.JSONDecodeError:
        parsed["raw_output"] = cleaned

    return parsed


# -----------------------
# Core helper: run domain agent once (robust)
# -----------------------
async def _run_domain_classifier_once(
    description: str,
    session_id: Optional[str] = None,
    user_id: str = "domain-user",
    max_retries: int = 1,
) -> Dict[str, Any]:
    """
    Run the domain classifier agent once and return parsed dict.
    Uses unique session ids by default to avoid collisions.
    Retries once on transient errors.
    """
    # Use a unique session id by default to avoid AlreadyExistsError
    session_id = session_id or f"domain-session-{uuid4().hex}"

    # Try to create session, but ignore errors (session may already exist in some environments)
    try:
        await domain_runner.session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except Exception as e:
        # Log and continue; if session exists it is not fatal
        logger.debug("create_session raised (ignored): %s", repr(e))

    prompt = (
        "Classify the following ticket description into a support domain and "
        "follow the JSON-only response instructions.\n\n"
        f"Ticket Description:\n{description}\n"
    )

    user_content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    final_text: Optional[str] = None

    attempt = 0
    last_exc = None
    while attempt <= max_retries:
        attempt += 1
        try:
            async for event in domain_runner.run_async(
                user_id=user_id, session_id=session_id, new_message=user_content
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    texts = [p.text for p in event.content.parts if getattr(p, "text", None)]
                    if texts:
                        final_text = "\n".join(texts).strip()
            break  # success
        except Exception as e:
            last_exc = e
            logger.warning(
                "domain_runner.run_async attempt %d failed: %s. Retrying if attempts remain.",
                attempt,
                repr(e),
            )
            await asyncio.sleep(0.3)
            continue

    if final_text is None:
        # If still nothing, raise with helpful message
        raise RuntimeError(f"Domain agent produced no output. Last exception: {repr(last_exc)}")

    parsed = _clean_and_parse_json(final_text)
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
    try:
        result = run_domain_classifier(test_desc)
        print("=== Parsed result ===")
        for k, v in result.items():
            print(f"{k}: {v}")
    except Exception as e:
        logger.exception("Domain classifier test failed: %s", e)
