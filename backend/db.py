# backend/db.py
import sqlite3
from pathlib import Path
from datetime import datetime
import os

from dotenv import load_dotenv

# Load .env so we get DATABASE_URL
load_dotenv()

# For simplicity, we extract the SQLite path from DATABASE_URL like: sqlite:///./tickets.db
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tickets.db")

if not DATABASE_URL.startswith("sqlite:///"):
    raise ValueError("Only sqlite:/// URLs are supported in this project for now.")

DB_PATH = DATABASE_URL.replace("sqlite:///", "")

# Ensure the directory for DB exists
db_path_obj = Path(DB_PATH)
if db_path_obj.parent != Path("."):
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    """
    Returns a new SQLite connection.
    We set row_factory to sqlite3.Row so we can access columns by name.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize the database with the base tickets table.
    We will add more tables (sessions, kb, proposals) later in separate steps.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Basic tickets table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            domain TEXT,
            status TEXT NOT NULL,
            resolved_by TEXT,
            resolution TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )

    conn.commit()
    conn.close()


def create_ticket(description: str) -> str:
    """
    Insert a new ticket with status 'new' and no domain yet.
    Returns the generated ticket_id.
    """
    from uuid import uuid4

    ticket_id = f"T-{uuid4().hex[:8]}"
    now = datetime.utcnow().isoformat()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tickets (
            ticket_id, description, domain, status, resolved_by, resolution, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ticket_id, description, None, "new", None, None, now, now),
    )
    conn.commit()
    conn.close()

    return ticket_id


def get_ticket(ticket_id: str):
    """
    Fetch a ticket row by ID. Returns dict or None.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)


def list_tickets():
    """
    Return all tickets as list of dicts (for debugging / UI listing).
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tickets ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]
