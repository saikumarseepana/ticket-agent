# backend/orchestrator.py
from typing import Tuple

from backend.db import get_ticket, update_ticket_fields
from backend.agents.generic_agent import classify_domain, suggest_resolution
from backend.adk_ticket_agent import run_ticket_agent


# -----------------------------
# OLD STUB FLOW (kept as-is)
# -----------------------------
def run_ai_on_ticket(ticket_id: str) -> Tuple[dict, str, float]:
    """
    Phase 1 stub:
    - Loads ticket
    - Classifies domain (rule-based)
    - Calls generic_agent to get suggested resolution
    - Updates ticket in DB with domain, status='awaiting_human', and resolution
    - Returns (ticket_dict, agent_name, confidence)
    """
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise ValueError(f"Ticket {ticket_id} not found")

    description = ticket["description"]
    domain = classify_domain(description)

    suggestion = suggest_resolution(description, domain)
    agent_name = suggestion["agent_name"]
    resolution = suggestion["resolution"]
    confidence = suggestion["confidence"]

    # Update ticket: set domain, status awaiting human, and proposed resolution
    update_ticket_fields(
        ticket_id=ticket_id,
        domain=domain,
        status="awaiting_human",
        resolution=resolution,
    )

    updated_ticket = get_ticket(ticket_id)
    assert updated_ticket is not None

    return updated_ticket, agent_name, confidence


# -----------------------------
# NEW: ADK + Gemini flow
# -----------------------------
def _parse_ticket_agent_output(text: str) -> dict:
    """
    Parse the output from the ADK ticket_triage_agent.

    Expected format (approx):

    TicketType: access
    Plan:
    - step 1
    - step 2
    Reply:
    <some text>

    We parse out:
      - ticket_type (lowercased, no spaces)
      - plan (string)
      - reply (string)
    """
    lines = text.splitlines()
    ticket_type = "other"
    plan_lines = []
    reply_lines = []

    state = None  # None | "plan" | "reply"

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        # Section headers
        if line.lower().startswith("tickettype:"):
            ticket_type = line.split(":", 1)[1].strip().lower()
            continue

        if line.lower().startswith("plan:"):
            state = "plan"
            continue

        if line.lower().startswith("reply:"):
            state = "reply"
            continue

        # Content accumulation
        if state == "plan":
            plan_lines.append(line)
        elif state == "reply":
            reply_lines.append(line)

    plan = "\n".join(plan_lines).strip()
    reply = "\n".join(reply_lines).strip()

    return {
        "ticket_type": ticket_type or "other",
        "plan": plan,
        "reply": reply,
    }


def _map_ticket_type_to_domain(ticket_type: str) -> str:
    """
    Map ticket agent's TicketType to our ticket.domain values.

    TicketType examples: access, incident, request, billing, other
    Our domains: 'access', 'network', 'billing', 'generic', etc.

    For now:
      access  -> "access"
      billing -> "billing"
      everything else -> "generic"
    """
    tt = ticket_type.strip().lower()

    if tt == "access":
        return "access"
    if tt == "billing":
        return "billing"

    # You can extend this mapping later if you add more domains.
    return "generic"


def run_ai_on_ticket_llm(ticket_id: str) -> Tuple[dict, str, str]:
    """
    ADK + Gemini-powered flow:

    - Load ticket
    - Build a prompt from its description
    - Call the ADK ticket_triage_agent (Gemini 2.5 Flash)
    - Parse TicketType / Plan / Reply
    - Map TicketType to our ticket.domain
    - Update ticket with:
        domain
        status = 'awaiting_human'
        resolution = reply (or plan if reply missing)
    - Return (updated_ticket, ticket_type, raw_output)
    """
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise ValueError(f"Ticket {ticket_id} not found")

    description = ticket["description"]

    # You can enrich this with more context later (past resolutions, etc.)
    prompt = (
        "You are helping triage an IT/support ticket.\n\n"
        f"Ticket description:\n{description}\n\n"
        "Follow your instructions: classify the ticket, suggest a short plan, "
        "and draft a short user-facing reply. Use the required output format."
    )

    # Call ADK agent (Gemini)
    raw_output = run_ticket_agent(prompt)
    parsed = _parse_ticket_agent_output(raw_output)

    ticket_type = parsed["ticket_type"]
    domain = _map_ticket_type_to_domain(ticket_type)

    # Choose what we store as resolution
    resolution = parsed["reply"] or parsed["plan"] or raw_output

    # Update DB
    update_ticket_fields(
        ticket_id=ticket_id,
        domain=domain,
        status="awaiting_human",
        resolution=resolution,
    )

    updated_ticket = get_ticket(ticket_id)
    assert updated_ticket is not None

    return updated_ticket, ticket_type, raw_output
