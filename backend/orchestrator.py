# backend/orchestrator.py
import re
import logging
from typing import Tuple, Dict, Any

from backend.db import get_ticket, update_ticket_fields
from backend.adk_ticket_agent import run_ticket_agent
from backend.adk_domain_agent import run_domain_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Helpers for parsing agent output
# -----------------------------
def _parse_ticket_agent_output(text: str) -> dict:
    """
    Parse the ticket agent output format:
    TicketType: <type>
    Plan:
    - ...
    Reply:
    <text>
    """
    lines = text.splitlines()
    ticket_type = "other"
    plan_lines = []
    reply_lines = []
    state = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("tickettype:"):
            ticket_type = line.split(":", 1)[1].strip().lower()
            continue
        if line.lower().startswith("plan:"):
            state = "plan"
            continue
        if line.lower().startswith("reply:"):
            state = "reply"
            continue
        if state == "plan":
            plan_lines.append(line)
        elif state == "reply":
            reply_lines.append(line)

    plan = "\n".join(plan_lines).strip()
    reply = "\n".join(reply_lines).strip()

    return {"ticket_type": ticket_type or "other", "plan": plan, "reply": reply}


def _map_ticket_type_to_domain(ticket_type: str) -> str:
    tt = ticket_type.strip().lower()
    if tt == "access":
        return "access"
    if tt == "billing":
        return "billing"
    if tt == "network":
        return "network"
    return "generic"


# -----------------------------
# Simple legacy flow (unchanged)
# -----------------------------
def run_ai_on_ticket(ticket_id: str) -> Tuple[dict, str, float]:
    """
    Backwards-compatible lightweight stub kept for existing callers.
    """
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise ValueError(f"Ticket {ticket_id} not found")
    description = ticket.get("description", "")
    # naive domain guess using classifier for better behavior
    classifier = run_domain_classifier(description)
    domain = classifier.get("domain", "other")
    # call default triage (default model inside adk_ticket_agent)
    prompt = (
        "You are helping triage an IT/support ticket.\n\n"
        f"Ticket description:\n{description}\n\n"
        "Follow your instructions: classify the ticket, suggest a short plan, "
        "and draft a short user-facing reply. Use the required output format."
    )
    raw_output = run_ticket_agent(prompt)  # default model (backwards compatible)
    parsed = _parse_ticket_agent_output(raw_output)
    ticket_type = parsed.get("ticket_type", "other")
    domain = _map_ticket_type_to_domain(ticket_type)
    resolution = parsed.get("reply") or parsed.get("plan") or raw_output

    update_ticket_fields(ticket_id=ticket_id, domain=domain, status="awaiting_human", resolution=resolution)
    return get_ticket(ticket_id), "ticket_triage_agent", 0.0


# -----------------------------
# LLM direct flow (unchanged available)
# -----------------------------
def run_ai_on_ticket_llm(ticket_id: str) -> Tuple[dict, str, str]:
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise ValueError(f"Ticket {ticket_id} not found")
    description = ticket.get("description", "")
    prompt = (
        "You are helping triage an IT/support ticket.\n\n"
        f"Ticket description:\n{description}\n\n"
        "Follow your instructions: classify the ticket, suggest a short plan, "
        "and draft a short user-facing reply. Use the required output format."
    )
    raw_output = run_ticket_agent(prompt)  # default model
    parsed = _parse_ticket_agent_output(raw_output)
    ticket_type = parsed["ticket_type"]
    domain = _map_ticket_type_to_domain(ticket_type)
    resolution = parsed.get("reply") or parsed.get("plan") or raw_output
    update_ticket_fields(ticket_id=ticket_id, domain=domain, status="awaiting_human", resolution=resolution)
    return get_ticket(ticket_id), ticket_type, raw_output


# -----------------------------
# HYBRID ROUTING
# -----------------------------
def _complexity_score(description: str) -> float:
    """
    Lightweight complexity heuristic: returns 0..1
    - length-based
    - sentence count
    - presence of certain keywords
    """
    desc = (description or "").strip()
    if not desc:
        return 0.0

    length = len(desc)
    sentences = max(1, len(re.split(r"[.!?]\s+", desc)))
    keywords = ["invoice", "refund", "payment", "attach", "attachment", "error code", "critical", "fail", "security", "password reset", "sso"]
    kw_count = sum(1 for kw in keywords if kw in desc.lower())

    a = min(1.0, length / 2000)
    b = min(1.0, sentences / 6)
    c = min(1.0, kw_count / 3)
    score = 0.5 * a + 0.3 * b + 0.2 * c
    return float(max(0.0, min(1.0, score)))


def run_ai_on_ticket_hybrid(ticket_id: str,
                             classifier_conf_threshold: float = 0.70,
                             complexity_threshold: float = 0.5) -> Tuple[dict, Dict[str, Any]]:
    """
    Hybrid routing logic:
     - Run domain classifier
     - Compute complexity score
     - Call cheap model (flash) first
     - If classifier_confidence < classifier_conf_threshold OR complexity > complexity_threshold:
         call pro model and prefer its output
     - Update ticket and return (updated_ticket, details)
    """
    ticket = get_ticket(ticket_id)
    if ticket is None:
        raise ValueError(f"Ticket {ticket_id} not found")

    description = ticket.get("description", "")

    # 1) classifier
    try:
        classifier = run_domain_classifier(description)
    except Exception as e:
        logger.warning("Domain classifier failed: %s. Falling back to defaults.", repr(e))
        classifier = {"domain": "other", "confidence": 0.0, "suggested_agent": "generic_agent", "raw_output": ""}

    cls_conf = float(classifier.get("confidence", 0.0) or 0.0)
    suggested_agent = classifier.get("suggested_agent", "generic_agent")

    # 2) complexity
    complexity = _complexity_score(description)

    # Build prompt
    prompt = (
        "You are helping triage an IT/support ticket.\n\n"
        f"Ticket description:\n{description}\n\n"
        "Follow your instructions: classify the ticket, suggest a short plan, "
        "and draft a short user-facing reply. Use the required output format."
    )

    flash_model = "gemini-2.5-flash"
    pro_model = "gemini-2.5-pro"

    flash_output = None
    pro_output = None
    model_used = flash_model
    fallback_used = False

    # Call flash
    try:
        flash_output = run_ticket_agent(prompt, model=flash_model)
    except Exception as e:
        logger.warning("Flash model call failed: %s. Will attempt pro. Using empty flash_output.", repr(e))
        flash_output = ""
        fallback_used = True

    # Decide whether to call pro
    need_pro = fallback_used or (cls_conf < classifier_conf_threshold) or (complexity > complexity_threshold)
    if need_pro:
        try:
            pro_output = run_ticket_agent(prompt, model=pro_model)
            model_used = pro_model
            fallback_used = True
        except Exception as e:
            logger.error("Pro model call failed: %s. Keeping flash output if available.", repr(e))
            pro_output = None

    final_output = pro_output or flash_output or ""
    parsed = _parse_ticket_agent_output(final_output)

    ticket_type = parsed.get("ticket_type", "other")
    domain = _map_ticket_type_to_domain(ticket_type)
    resolution = parsed.get("reply") or parsed.get("plan") or final_output

    # update ticket
    update_ticket_fields(ticket_id=ticket_id, domain=domain, status="awaiting_human", resolution=resolution)

    updated_ticket = get_ticket(ticket_id)
    details = {
        "classifier": classifier,
        "complexity_score": complexity,
        "model_used": model_used,
        "flash_output": flash_output,
        "pro_output": pro_output,
        "fallback_used": fallback_used,
        "ticket_type": ticket_type,
        "suggested_agent": suggested_agent,
    }
    return updated_ticket, details
