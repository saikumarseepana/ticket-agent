# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from backend.db import init_db, create_ticket, get_ticket, list_tickets


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
