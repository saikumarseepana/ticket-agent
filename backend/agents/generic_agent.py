# backend/agents/generic_agent.py
"""
Generic fallback agent module.
Exposes handle_message(prompt: str) -> str
"""

from backend.adk_ticket_agent import run_ticket_agent

def handle_message(prompt: str) -> str:
    # uses default ADK ticket agent wrapper
    return run_ticket_agent(prompt)
