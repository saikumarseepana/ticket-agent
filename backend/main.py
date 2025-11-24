# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import traceback
import logging
from typing import Any, Dict

# Import DB + orchestrator + ADK helpers
from backend.db import init_db, create_ticket, get_ticket, list_tickets
from backend.orchestrator import run_ai_on_ticket, run_ai_on_ticket_llm
from backend.adk_domain_agent import run_domain_classifier

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized.")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Ticket Agent Backend (Phase 1)", lifespan=lifespan)


# -----------------------
# Pydantic models
# -----------------------
class TicketCreateRequest(BaseModel):
    description: str


class TicketResponse(BaseModel):
    ticket_id: str
    description: str
    domain: str | None
    status: str
    resolved_by: str | None
    resolution: str | None
    created_at: str
    updated_at: str


class RunAIResponse(BaseModel):
    ticket: TicketResponse
    agent_name: str
    confidence: float


class RunAILlmResponse(BaseModel):
    ticket: TicketResponse
    ticket_type: str
    raw_output: str


class DomainClassifyRequest(BaseModel):
    description: str


class DomainClassifyResponse(BaseModel):
    domain: str
    confidence: float
    reason: str
    summary: str
    suggested_agent: str
    raw_output: str


# -----------------------
# Ticket endpoints (unchanged)
# -----------------------
@app.post("/tickets", response_model=TicketResponse)
def create_ticket_endpoint(req: TicketCreateRequest):
    ticket_id = create_ticket(req.description)
    ticket = get_ticket(ticket_id)
    assert ticket is not None
    return ticket


@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket_endpoint(ticket_id: str):
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.get("/tickets")
def list_tickets_endpoint():
    return list_tickets()


@app.post("/tickets/{ticket_id}/run_ai", response_model=RunAIResponse)
def run_ai_endpoint(ticket_id: str):
    try:
        ticket, agent_name, confidence = run_ai_on_ticket(ticket_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {"ticket": ticket, "agent_name": agent_name, "confidence": confidence}


@app.post("/tickets/{ticket_id}/run_ai_llm", response_model=RunAILlmResponse)
def run_ai_llm_endpoint(ticket_id: str):
    try:
        ticket, ticket_type, raw_output = run_ai_on_ticket_llm(ticket_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Ticket not found")
    except Exception as e:
        # Log full traceback and return 502 with a helpful message
        tb = traceback.format_exc()
        logger.error("Error running run_ai_on_ticket_llm for ticket %s: %s", ticket_id, tb)
        raise HTTPException(
            status_code=502,
            detail=f"LLM orchestrator error. See server logs. Message: {repr(e)}"
        )
    return {"ticket": ticket, "ticket_type": ticket_type, "raw_output": raw_output}


# -----------------------
# NEW: Robust domain-classify endpoint with error handling
# -----------------------
@app.post("/classify_llm", response_model=DomainClassifyResponse)
def classify_llm_endpoint(req: DomainClassifyRequest):
    """
    Calls the ADK + Gemini domain_classifier_agent.
    Wrapped with robust error handling so we surface useful debug info.
    """
    description = req.description
    try:
        result = run_domain_classifier(description)
        # ensure types are safe for response_model validation
        return DomainClassifyResponse(
            domain=str(result.get("domain", "other")),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            reason=str(result.get("reason", "")),
            summary=str(result.get("summary", "")),
            suggested_agent=str(result.get("suggested_agent", "generic_agent")),
            raw_output=str(result.get("raw_output", "")),
        )
    except Exception as exc:
        tb = traceback.format_exc()
        # Log the error + the exact input that triggered it (important)
        logger.error("Exception in /classify_llm for input: %s\nTraceback:\n%s", description, tb)
        # Return a 502 with a helpful, non-sensitive message
        raise HTTPException(
            status_code=502,
            detail=(
                "Domain classifier failed. The server logged the full traceback for debugging. "
                "If this persists, paste the input and server logs to investigate."
            ),
        )
