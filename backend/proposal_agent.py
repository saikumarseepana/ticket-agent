# backend/proposal_agent.py
"""
Proposal Agent: scans domains periodically and creates agent proposals
when thresholds are met. Uses backend.db helpers implemented previously.

Usage:
  # run once:
  PYTHONPATH=. python backend/proposal_agent.py --once

  # run as a loop with default 60s interval:
  PYTHONPATH=. python backend/proposal_agent.py

  # run loop with custom interval (seconds):
  PYTHONPATH=. python backend/proposal_agent.py --interval 30
"""

import argparse
import logging
import time
from typing import Dict, Any

from backend import db

# -----------------------
# Config (tweakable)
# -----------------------
SCAN_INTERVAL_SECONDS = 60  # default periodic scan
MIN_TICKETS_FOR_PROPOSAL = 2     # your Q1
MIN_HRI_FOR_PROPOSAL = 0.50      # your Q2 (50%)
MAX_EXAMPLES = 3                 # Option B: store up to 3 recent tickets

# If admin rejected with "not needed" we only override after these:
OVERRIDE_MIN_TICKETS = 100
OVERRIDE_MIN_HRI = 0.97

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -----------------------
# Helper: interpret rejection policy (basic)
# -----------------------
def is_blocked_by_rejection_policy(domain: str) -> bool:
    pol = db.get_rejection_policy(domain)
    if not pol:
        return False
    # If there's a record in rejection_policies, treat as blocked unless override thresholds are met
    try:
        min_tickets = int(pol.get("min_tickets_threshold", OVERRIDE_MIN_TICKETS))
        min_hir = float(pol.get("min_hir_threshold", OVERRIDE_MIN_HRI)) / 100.0 if pol.get("min_hir_threshold") and int(pol.get("min_hir_threshold")) > 1 else float(pol.get("min_hir_threshold", OVERRIDE_MIN_HRI))
    except Exception:
        min_tickets = OVERRIDE_MIN_TICKETS
        min_hir = OVERRIDE_MIN_HRI

    stats = db.domain_stats(domain)
    total = stats.get("ticket_count", 0)
    hri = stats.get("human_involvement_ratio", 0.0)

    # If policy exists, block until override thresholds are met
    if total >= min_tickets and hri >= min_hir:
        logging.info(f"Domain {domain} had a rejection policy, but override thresholds met (tickets={total}, hri={hri:.2f}). Will allow proposal.")
        return False
    logging.info(f"Domain {domain} is blocked by rejection policy (tickets={total}, hri={hri:.2f}).")
    return True


# -----------------------
# Core check logic
# -----------------------
def should_propose(domain: str, stats: Dict[str, Any]) -> bool:
    """
    Decide whether to propose:
      - We require minimum ticket count AND HRI
      - There must be no existing agent for domain
      - There must be no pending proposal for domain
      - Not blocked by rejection_policies (unless override)
    """
    ticket_count = stats.get("ticket_count", 0)
    hri = stats.get("human_involvement_ratio", 0.0)

    if ticket_count < MIN_TICKETS_FOR_PROPOSAL:
        logging.debug(f"Domain {domain} skipped: not enough tickets ({ticket_count}<{MIN_TICKETS_FOR_PROPOSAL}).")
        return False

    if hri < MIN_HRI_FOR_PROPOSAL:
        logging.debug(f"Domain {domain} skipped: HRI too low ({hri:.2f}<{MIN_HRI_FOR_PROPOSAL}).")
        return False

    # If agent already exists for domain -> no propose
    agent = db.get_agent_for_domain(domain)
    if agent:
        logging.debug(f"Domain {domain} skipped: agent already exists ({agent['agent_name']}).")
        return False

    # if pending proposal exists -> don't create duplicate
    pending = [p for p in db.list_pending_proposals(limit=200) if p["domain"] == domain]
    if pending:
        logging.info(f"Domain {domain} skipped: pending proposal already exists (count={len(pending)}).")
        return False

    # Rejection policy check (block unless override criteria)
    if is_blocked_by_rejection_policy(domain):
        return False

    return True


def build_suggested_config(domain: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a basic suggested configuration for the proposed agent.
    Admin can edit after seeing the proposal.
    """
    examples = stats.get("examples", [])[:MAX_EXAMPLES]
    suggested_name = f"{domain}_agent"
    suggestion = {
        "suggested_agent_name": suggested_name,
        "suggested_model": "gemini-2.5-pro",  # default suggestion; admin can change
        "suggested_instruction": (
            f"You are an agent specialized in the '{domain}' domain. "
            "You should handle typical tickets in this domain, consult KB/examples provided, and escalate to human when unsure."
        ),
        "suggested_tools": ["kb_lookup", "ticket_update", "escalation"],
        "example_tickets": examples,
        "metrics": stats,
    }
    return suggestion


def scan_once() -> None:
    """
    Single-pass scan: inspect known domains and create proposals if needed.
    """
    logging.info("Proposal Agent: starting scan_once()")
    # Approach: collect domains from tickets table
    # We'll query distinct domains
    conn = db._connect()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT domain FROM tickets WHERE domain IS NOT NULL;")
    rows = cur.fetchall()
    domains = [r["domain"] for r in rows if r["domain"]]
    conn.close()

    if not domains:
        logging.info("No domains found in tickets table. Nothing to propose.")
        return

    logging.info(f"Found domains: {domains}")

    for domain in domains:
        stats = db.domain_stats(domain)
        logging.info(f"Stats for domain '{domain}': tickets={stats['ticket_count']}, hri={stats['human_involvement_ratio']:.2f}")

        try:
            if should_propose(domain, stats):
                # build suggestion
                cfg = build_suggested_config(domain, stats)
                reason_metrics = cfg["metrics"]
                example_tickets = cfg["example_tickets"]

                # Submit the proposal to DB (Proposal Agent doesn't approve)
                pid = db.submit_proposal(domain=domain, reason=reason_metrics, example_tickets=example_tickets)
                logging.info(f"Submitted proposal {pid} for domain '{domain}' with {len(example_tickets)} examples.")
            else:
                logging.debug(f"No proposal for domain: {domain}")
        except Exception as e:
            logging.exception(f"Error while processing domain '{domain}': {e}")

    logging.info("Proposal Agent: scan_once() finished.")


# -----------------------
# Loop runner
# -----------------------
def run_loop(interval_seconds: int = SCAN_INTERVAL_SECONDS):
    logging.info(f"Proposal Agent: starting main loop with interval={interval_seconds}s")
    try:
        while True:
            scan_once()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logging.info("Proposal Agent: interrupted. Exiting.")


# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    parser.add_argument("--interval", type=int, default=SCAN_INTERVAL_SECONDS, help="Scan interval seconds")
    args = parser.parse_args()

    # Ensure DB schema exists
    db.ensure_schema()

    if args.once:
        scan_once()
    else:
        run_loop(args.interval)


if __name__ == "__main__":
    main()
