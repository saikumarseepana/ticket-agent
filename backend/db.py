# backend/db.py
import sqlite3
import json
import uuid
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from io import BytesIO

DB_PATH = Path("ticketing.db")


# ---------------------------
# Low-level helpers
# ---------------------------
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    # Return rows as dict-like objects
    conn.row_factory = sqlite3.Row
    return conn


def _to_blob(nd: np.ndarray) -> bytes:
    bio = BytesIO()
    # Use np.save to preserve dtype & shape
    np.save(bio, nd, allow_pickle=False)
    bio.seek(0)
    return bio.read()


def _from_blob(blob: bytes) -> np.ndarray:
    bio = BytesIO(blob)
    bio.seek(0)
    return np.load(bio, allow_pickle=False)


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


# ---------------------------
# Initialization convenience
# ---------------------------
def ensure_schema():
    """Ensure DB schema exists (delegates to backend/db_schema.py)."""
    # If the DB file exists and has expected tables, skip.
    conn = _connect()
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets';")
        if cur.fetchone():
            conn.close()
            return True
    finally:
        conn.close()

    # Otherwise call schema initializer
    # Import local script (it will create ticketing.db with schema)
    import backend.db_schema as schema  # type: ignore
    schema.init_schema()
    return True


# ---------------------------
# TICKETS
# ---------------------------
def insert_ticket(ticket_id: Optional[str], description: str, domain: Optional[str] = None,
                  status: str = "open", resolution: Optional[str] = None) -> str:
    """
    Insert a ticket. If ticket_id is None, a new UUID is generated and returned.
    """
    ensure_schema()
    tid = ticket_id or f"T-{uuid.uuid4().hex[:8]}"
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO tickets (
            ticket_id, description, domain, status, resolution, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
        """,
        (tid, description, domain, status, resolution),
    )
    conn.commit()
    conn.close()
    return tid


def get_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tickets WHERE ticket_id = ?;", (ticket_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def list_tickets_by_domain(domain: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tickets WHERE domain = ? ORDER BY created_at DESC LIMIT ?;", (domain, limit))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_all_tickets(limit: int = 100) -> List[Dict[str, Any]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tickets ORDER BY created_at DESC LIMIT ?;", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_ticket_status(ticket_id: str, status: str, resolution: Optional[str] = None):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE tickets
        SET status = ?, resolution = ?, updated_at = CURRENT_TIMESTAMP
        WHERE ticket_id = ?;
        """,
        (status, resolution, ticket_id),
    )
    conn.commit()
    conn.close()


# ---------------------------
# AGENT REGISTRY
# ---------------------------
def register_agent(agent_name: str, domain: str, model_type: str, config: Dict[str, Any]):
    """
    Add or update an agent in registry.
    'config' can contain instructions, tools list, permissions, etc.
    """
    ensure_schema()
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO agent_registry (
            agent_name, domain, model_type, config, created_at
        ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP);
        """,
        (agent_name, domain, model_type, json.dumps(config)),
    )
    conn.commit()
    conn.close()


def get_agent_by_name(agent_name: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM agent_registry WHERE agent_name = ?;", (agent_name,))
    r = cur.fetchone()
    conn.close()
    return dict(r) if r else None


def get_agent_for_domain(domain: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM agent_registry WHERE domain = ? LIMIT 1;", (domain,))
    r = cur.fetchone()
    conn.close()
    return dict(r) if r else None


def list_agents() -> List[Dict[str, Any]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM agent_registry ORDER BY created_at DESC;")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def remove_agent(agent_name: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM agent_registry WHERE agent_name = ?;", (agent_name,))
    conn.commit()
    conn.close()


# ---------------------------
# PROPOSALS
# ---------------------------
def submit_proposal(domain: str, reason: Dict[str, Any], example_tickets: List[str]) -> str:
    """
    Submit a new proposal (created by Proposal Agent).
    'reason' should be JSON-serializable metrics (ticket_count, HRI, notes, etc).
    'example_tickets' is a list (we'll store as JSON).
    Returns proposal_id.
    """
    ensure_schema()
    pid = f"P-{uuid.uuid4().hex[:8]}"
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO agent_proposals (
            proposal_id, domain, status, reason, admin_comment, created_at
        ) VALUES (?, ?, 'pending', ?, NULL, CURRENT_TIMESTAMP);
        """,
        (pid, domain, json.dumps({"metrics": reason, "examples": example_tickets})),
    )
    conn.commit()
    conn.close()
    return pid


def list_pending_proposals(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM agent_proposals WHERE status = 'pending' ORDER BY created_at ASC LIMIT ?;", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM agent_proposals WHERE proposal_id = ?;", (proposal_id,))
    r = cur.fetchone()
    conn.close()
    return dict(r) if r else None


def approve_proposal(proposal_id: str, agent_name: str, model_type: str, config: Dict[str, Any]):
    """
    Approve a proposal: moves it to agent_registry and marks proposal as approved.
    """
    ensure_schema()
    prop = get_proposal(proposal_id)
    if not prop:
        raise ValueError("Proposal not found")

    domain = prop["domain"]
    # create agent record
    register_agent(agent_name=agent_name, domain=domain, model_type=model_type, config=config)

    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE agent_proposals
        SET status = 'approved', admin_comment = ?, decided_at = CURRENT_TIMESTAMP
        WHERE proposal_id = ?;
        """,
        (f"Approved and registered as {agent_name}", proposal_id),
    )
    conn.commit()
    conn.close()


def reject_proposal(proposal_id: str, admin_comment: str):
    """
    Reject a proposal and store admin_comment (reason). This comment may be parsed by Proposal Agent later.
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE agent_proposals
        SET status = 'rejected', admin_comment = ?, decided_at = CURRENT_TIMESTAMP
        WHERE proposal_id = ?;
        """,
        (admin_comment, proposal_id),
    )
    conn.commit()
    conn.close()


# ---------------------------
# REJECTION POLICIES
# ---------------------------
def set_rejection_policy(domain: str, rejection_reason: str,
                         min_tickets_threshold: int = 100, min_hir_threshold: int = 97):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO rejection_policies (
            domain, rejected_at, rejection_reason, min_tickets_threshold, min_hir_threshold
        ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?);
        """,
        (domain, rejection_reason, min_tickets_threshold, min_hir_threshold),
    )
    conn.commit()
    conn.close()


def get_rejection_policy(domain: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM rejection_policies WHERE domain = ?;", (domain,))
    r = cur.fetchone()
    conn.close()
    return dict(r) if r else None


# ---------------------------
# EMBEDDINGS (MiniLM local)
# ---------------------------
def insert_embedding(embed_id: Optional[str], ticket_id: str, chunk_text: str, vector: np.ndarray) -> str:
    """
    Store an embedding. `vector` is a 1D numpy array (float32/float64).
    """
    ensure_schema()
    eid = embed_id or f"E-{uuid.uuid4().hex[:8]}"
    blob = _to_blob(vector)
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO kb_embeddings (embed_id, ticket_id, chunk_text, embedding, created_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP);
        """,
        (eid, ticket_id, chunk_text, sqlite3.Binary(blob)),
    )
    conn.commit()
    conn.close()
    return eid


def get_all_embeddings() -> List[Tuple[str, str, str, np.ndarray]]:
    """
    Returns list of tuples: (embed_id, ticket_id, chunk_text, vector)
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT embed_id, ticket_id, chunk_text, embedding FROM kb_embeddings;")
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        vec = _from_blob(r["embedding"])
        out.append((r["embed_id"], r["ticket_id"], r["chunk_text"], vec))
    return out


def simple_embedding_search(query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Simple CPU cosine-similarity search over stored embeddings.
    Returns top_k entries with 'score' (higher is better).
    """
    all_emb = get_all_embeddings()
    if len(all_emb) == 0:
        return []
    q = query_vector.astype(np.float32)
    scores = []
    for eid, tid, text, vec in all_emb:
        v = vec.astype(np.float32)
        # cosine similarity
        den = (np.linalg.norm(q) * np.linalg.norm(v))
        score = float(np.dot(q, v) / den) if den > 0 else 0.0
        scores.append((eid, tid, text, score))
    scores.sort(key=lambda x: x[3], reverse=True)
    results = []
    for eid, tid, text, score in scores[:top_k]:
        results.append({"embed_id": eid, "ticket_id": tid, "chunk_text": text, "score": score})
    return results


# ---------------------------
# CHAT SESSIONS
# ---------------------------
def insert_chat_message(ticket_id: str, sender: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    ensure_schema()
    cid = f"C-{uuid.uuid4().hex[:8]}"
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_sessions (chat_id, ticket_id, sender, message, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
        """,
        (cid, ticket_id, sender, message, json.dumps(metadata or {})),
    )
    conn.commit()
    conn.close()
    return cid


def get_chat_history(ticket_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM chat_sessions WHERE ticket_id = ? ORDER BY created_at ASC LIMIT ?;",
        (ticket_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------
# UTILITY: domain stats used by Proposal Agent
# ---------------------------
def domain_stats(domain: str) -> Dict[str, Any]:
    """
    Returns:
      - ticket_count
      - human_resolved_count
      - human_involvement_ratio (HRI)
      - latest_examples (list of up to 3 ticket descriptions)
    """
    conn = _connect()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) as cnt FROM tickets WHERE domain = ?;", (domain,))
    total = cur.fetchone()["cnt"]

    # We'll interpret 'resolved by human' as status == 'human_resolved' or resolution contains 'human'
    cur.execute("SELECT COUNT(*) as cnt FROM tickets WHERE domain = ? AND status = 'human_resolved';", (domain,))
    human_resolved = cur.fetchone()["cnt"]

    # latest 3 examples
    cur.execute("SELECT description FROM tickets WHERE domain = ? ORDER BY created_at DESC LIMIT 3;", (domain,))
    rows = cur.fetchall()
    examples = [r["description"] for r in rows]

    conn.close()
    hri = (human_resolved / total) if total > 0 else 0.0
    return {
        "ticket_count": total,
        "human_resolved_count": human_resolved,
        "human_involvement_ratio": hri,
        "examples": examples,
    }


# ---------------------------
# Simple test helper (not run automatically)
# ---------------------------
def _smoke_test():
    ensure_schema()
    t = insert_ticket(None, "User can't login to payroll portal", domain="access")
    print("Inserted ticket:", t)
    insert_chat_message(t, "user", "I can't login even after password reset")
    print("Chat history:", get_chat_history(t))
    # embedding test
    vec = np.random.randn(384).astype(np.float32)
    eid = insert_embedding(None, t, "resolution note chunk", vec)
    print("Inserted embedding id:", eid)
    print("Embedding search:", simple_embedding_search(vec, top_k=1))


# ---------------------------
# Compatibility wrappers (for existing main.py imports)
# ---------------------------
def init_db():
    """
    Backwards-compatible initializer (previous main.py used init_db()).
    """
    ensure_schema()
    return True


def create_ticket(description: str, domain: Optional[str] = None, status: str = "open") -> str:
    """
    Backwards-compatible create_ticket.
    """
    return insert_ticket(None, description, domain=domain, status=status)


def list_tickets(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Backwards-compatible listing. If main.py expects a domain-agnostic list, return recent tickets.
    """
    return list_all_tickets(limit=limit)


def update_ticket_fields(ticket_id: str, fields: Dict[str, Any]) -> bool:    ## check
    """
    Update arbitrary fields on a ticket row.
    - ticket_id: the ticket id to update.
    - fields: dict of column -> value. Allowed columns: description, domain, status, resolution, updated_at.
              If a value is a dict, it will be JSON-serialized.
    Returns True if update performed, False if no fields or ticket not found.
    """
    if not fields:
        return False

    allowed = {"description", "domain", "status", "resolution", "updated_at"}
    # Filter fields to allowed set to avoid accidental injection
    update_items = {k: v for k, v in fields.items() if k in allowed}
    if not update_items:
        return False

    # serialize any dict values
    params = []
    set_clauses = []
    for col, val in update_items.items():
        if isinstance(val, dict):
            val = json.dumps(val)
        set_clauses.append(f"{col} = ?")
        params.append(val)

    # always update updated_at timestamp unless explicitly provided
    if "updated_at" not in update_items:
        set_clauses.append("updated_at = CURRENT_TIMESTAMP")

    params.append(ticket_id)

    sql = f"UPDATE tickets SET {', '.join(set_clauses)} WHERE ticket_id = ?;"
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM tickets WHERE ticket_id = ?;", (ticket_id,))
    if not cur.fetchone():
        conn.close()
        return False

    cur.execute(sql, tuple(params))
    conn.commit()
    conn.close()
    return True


# get_ticket already exists with same name; keep as alias
# def get_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:  # already defined above
#     return get_ticket(ticket_id)


if __name__ == "__main__":
    print("Running smoke test for backend/db.py ...")
    _smoke_test()
    print("Done.")
