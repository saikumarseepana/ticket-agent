# backend/agent_loader.py
import importlib
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

AGENTS_PACKAGE = "backend.agents"
AGENTS_DIR = Path(__file__).parent / "agents"


def _safe_import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        logger.debug(f"Import failed for {module_name}: {e}")
        return None


def load_agent_module(agent_name: str):
    """
    Try to import backend.agents.<agent_name>.
    If not found, fall back to generic_agent.
    Returns module or None.
    """
    mod_name = f"{AGENTS_PACKAGE}.{agent_name}"
    mod = _safe_import_module(mod_name)
    if mod:
        logger.info(f"Loaded agent module: {mod_name}")
        return mod

    # fallback
    fallback = _safe_import_module(f"{AGENTS_PACKAGE}.generic_agent")
    if fallback:
        logger.warning(f"Agent '{agent_name}' not found. Falling back to generic_agent.")
        return fallback

    logger.error("No agent module available (generic_agent missing).")
    return None


def get_agent_callable(agent_name: str) -> Optional[Callable[[str], str]]:
    """
    Returns a synchronous callable handle_message(prompt:str)->str.
    Agent modules should expose:
      - handle_message(prompt: str) -> str   (preferred)
    Or:
      - run(prompt: str) -> str
    If neither exists, returns None.
    """
    mod = load_agent_module(agent_name)
    if not mod:
        return None

    if hasattr(mod, "handle_message"):
        return getattr(mod, "handle_message")
    if hasattr(mod, "run"):
        return getattr(mod, "run")

    logger.error(f"Agent module {mod.__name__} does not expose handle_message/run.")
    return None


def create_agent_file_from_template(agent_name: str, instruction: str, model_type: str = "gemini-2.5-pro") -> str:
    """
    Create a basic agent file under backend/agents/<agent_name>.py using a template.
    If file exists, it will NOT overwrite.
    Returns path to created file (or existing file).
    """
    AGENTS_DIR.mkdir(exist_ok=True)
    target = AGENTS_DIR / f"{agent_name}.py"
    if target.exists():
        return str(target)

    template = f'''# backend/agents/{agent_name}.py
"""
Auto-generated agent file (template).
Customize instruction, tools and behavior as needed.
This file exposes a synchronous function `handle_message(prompt: str) -> str`
which the orchestrator will call.
"""

from backend.adk_ticket_agent import run_ticket_agent  # reuse existing generic ADK wrapper

AGENT_NAME = "{agent_name}"
MODEL = "{model_type}"
INSTRUCTION = """{instruction}"""

def handle_message(prompt: str) -> str:
    \"\"\"Synchronous handler. Reuse run_ticket_agent that does async->sync for ADK.\"\"\"
    # You may customize this to call specialized code or other tools.
    response = run_ticket_agent(prompt, model=MODEL)
    return response
'''
    target.write_text(template)
    return str(target)
