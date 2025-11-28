# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
import traceback

from backend.db import init_db, create_ticket, get_ticket, list_tickets
from backend.orchestrator import run_ai_on_ticket, run_ai_on_ticket_llm, run_ai_on_ticket_hybrid
from backend.adk_domain_agent import run_domain_classifier
from backend.admin_api import router as admin_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized.")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Ticket Agent Backend", lifespan=lifespan)

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

@app.post("/tickets/{ticket_id}/run_ai", response_model=dict)
def run_ai_endpoint(ticket_id: str):
    try:
        ticket, agent_name, confidence = run_ai_on_ticket(ticket_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {"ticket": ticket, "agent_name": agent_name, "confidence": confidence}

@app.post("/tickets/{ticket_id}/run_ai_llm", response_model=dict)
def run_ai_llm_endpoint(ticket_id: str):
    try:
        ticket, ticket_type, raw_output = run_ai_on_ticket_llm(ticket_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Ticket not found")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error in run_ai_on_ticket_llm: %s", tb)
        raise HTTPException(status_code=502, detail=f"LLM orchestrator error. Message: {repr(e)}")
    return {"ticket": ticket, "ticket_type": ticket_type, "raw_output": raw_output}

@app.post("/tickets/{ticket_id}/run_ai_hybrid", response_model=dict)
def run_ai_hybrid_endpoint(ticket_id: str):
    """
    Safe hybrid endpoint. Returns the updated ticket plus 'details' describing model decisions.
    """
    try:
        updated_ticket, details = run_ai_on_ticket_hybrid(ticket_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Ticket not found")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Error in /run_ai_hybrid for ticket %s: %s", ticket_id, tb)
        raise HTTPException(status_code=502, detail="Hybrid orchestrator error. Check server logs.")
    return {"ticket": updated_ticket, "details": details}

app.include_router(admin_router)
