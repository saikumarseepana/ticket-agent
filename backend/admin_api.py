# backend/admin_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import logging
from backend import db
from backend import agent_loader

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)


class ApprovePayload(BaseModel):
    agent_name: str
    model_type: str
    create_file: bool = True
    config: Dict[str, Any] = {}


class RejectPayload(BaseModel):
    admin_comment: str


@router.get("/proposals")
def list_proposals():
    return db.list_pending_proposals()


@router.get("/proposals/{proposal_id}")
def get_proposal(proposal_id: str):
    p = db.get_proposal(proposal_id)
    if not p:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return p


@router.post("/proposals/{proposal_id}/approve")
def approve_proposal_api(proposal_id: str, payload: ApprovePayload):
    p = db.get_proposal(proposal_id)
    if not p:
        raise HTTPException(status_code=404, detail="Proposal not found")

    # register in DB (agent registry) and mark proposal approved
    try:
        db.approve_proposal(proposal_id, agent_name=payload.agent_name, model_type=payload.model_type, config=payload.config)
    except Exception as e:
        logger.exception("Failed to approve proposal")
        raise HTTPException(status_code=500, detail=str(e))

    # Optionally create a file in backend/agents/
    agent_file_path = None
    if payload.create_file:
        reason = p.get("reason")
        # try to extract instruction or examples if present
        instruction = "Auto-generated agent. Customize instructions."
        try:
            rr = reason and (reason if isinstance(reason, str) else str(reason))
            instruction = f"Auto-generated for domain {p.get('domain')}.\nExamples: {p.get('reason')}"
        except Exception:
            pass
        agent_file_path = agent_loader.create_agent_file_from_template(payload.agent_name, instruction, payload.model_type)

    return {"status": "approved", "proposal_id": proposal_id, "agent_file": agent_file_path}


@router.post("/proposals/{proposal_id}/reject")
def reject_proposal_api(proposal_id: str, payload: RejectPayload):
    p = db.get_proposal(proposal_id)
    if not p:
        raise HTTPException(status_code=404, detail="Proposal not found")
    try:
        db.reject_proposal(proposal_id, payload.admin_comment)
    except Exception as e:
        logger.exception("Failed to reject proposal")
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "rejected", "proposal_id": proposal_id}


@router.get("/agents")
def list_agents_api():
    return db.list_agents()


@router.get("/agents/{agent_name}")
def get_agent_api(agent_name: str):
    a = db.get_agent_by_name(agent_name)
    if not a:
        raise HTTPException(status_code=404, detail="Agent not found")
    return a


@router.post("/agents/create")
def create_agent_api(payload: ApprovePayload):
    # create agent entry (admin manual creation)
    try:
        db.register_agent(agent_name=payload.agent_name, domain=payload.agent_name.split("_")[0], model_type=payload.model_type, config=payload.config)
    except Exception as e:
        logger.exception("Failed to create agent")
        raise HTTPException(status_code=500, detail=str(e))

    # optionally create file
    agent_file_path = None
    if payload.create_file:
        agent_file_path = agent_loader.create_agent_file_from_template(payload.agent_name, "Manual admin-created agent.", payload.model_type)

    return {"status": "created", "agent_file": agent_file_path}
