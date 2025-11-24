# backend/orchestrator.py
from typing import Tuple

from backend.db import get_ticket, update_ticket_fields
from backend.agents.generic_agent import classify_domain, suggest_resolution


def run_ai_on_ticket(ticket_id: str) -> Tuple[dict, str, float]:
    """
    Main orchestration entrypoint for Phase 1.
    - Loads ticket
    - Classifies domain (rule-based for now)
    - Calls generic_agent to get a suggested resolution
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
