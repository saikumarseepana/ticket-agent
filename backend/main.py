# backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from backend.orchestrator import run_ai_on_ticket, run_ai_on_ticket_llm
from backend.db import init_db, create_ticket, get_ticket, list_tickets
from backend.adk_domain_agent import run_domain_classifier


# New lifespan syntax (replaces on_event("startup"))
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    print("Database initialized.")
    yield
    # Shutdown (if needed later)
    print("Shutting down...")


app = FastAPI(
    title="Ticket Agent Backend (Phase 1)",
    lifespan=lifespan
)


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
    """
    Run the AI orchestrator on a given ticket.
    For now this uses a rule-based generic agent stub.
    Later we'll replace the internals with Gemini/ADK.
    """
    try:
        ticket, agent_name, confidence = run_ai_on_ticket(ticket_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Ticket not found")

    return {
        "ticket": ticket,
        "agent_name": agent_name,
        "confidence": confidence,
    }

@app.post("/tickets/{ticket_id}/run_ai_llm", response_model=RunAILlmResponse)
def run_ai_llm_endpoint(ticket_id: str):
    """
    Run the ADK + Gemini-based ticket_triage_agent on a given ticket.

    This:
      - Calls the LLM agent via ADK
      - Parses TicketType / Plan / Reply
      - Updates ticket.domain, ticket.status, ticket.resolution
      - Returns the updated ticket + ticket_type + raw raw_output
    """
    try:
        ticket, ticket_type, raw_output = run_ai_on_ticket_llm(ticket_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Ticket not found")

    return {
        "ticket": ticket,
        "ticket_type": ticket_type,
        "raw_output": raw_output,
    }

@app.post("/classify_llm", response_model=DomainClassifyResponse)
def classify_llm_endpoint(req: DomainClassifyRequest):
    """
    Test endpoint for the ADK + Gemini domain_classifier_agent.

    Input:
      - description: raw ticket text

    Output:
      - domain
      - confidence
      - reason
      - summary
      - suggested_agent
      - raw_output (raw JSON response as string)
    """
    result = run_domain_classifier(req.description)

    return DomainClassifyResponse(
        domain=str(result.get("domain", "other")),
        confidence=float(result.get("confidence", 0.0) or 0.0),
        reason=str(result.get("reason", "")),
        summary=str(result.get("summary", "")),
        suggested_agent=str(result.get("suggested_agent", "generic_agent")),
        raw_output=str(result.get("raw_output", "")),
    )
