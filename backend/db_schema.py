# backend/db_schema.py

import sqlite3

DB_PATH = "ticketing.db"

def init_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        domain TEXT,
        status TEXT DEFAULT 'open',
        resolution TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS agent_registry (
        agent_name TEXT PRIMARY KEY,
        domain TEXT NOT NULL,
        model_type TEXT NOT NULL,
        config JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS agent_proposals (
        proposal_id TEXT PRIMARY KEY,
        domain TEXT NOT NULL,
        status TEXT NOT NULL,
        reason TEXT,
        admin_comment TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        decided_at TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS rejection_policies (
        domain TEXT PRIMARY KEY,
        rejected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        rejection_reason TEXT,
        min_tickets_threshold INTEGER DEFAULT 100,
        min_hir_threshold INTEGER DEFAULT 97
    );

    CREATE TABLE IF NOT EXISTS kb_embeddings (
        embed_id TEXT PRIMARY KEY,
        ticket_id TEXT NOT NULL,
        chunk_text TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS chat_sessions (
        chat_id TEXT PRIMARY KEY,
        ticket_id TEXT NOT NULL,
        sender TEXT NOT NULL,
        message TEXT NOT NULL,
        metadata JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    return True


if __name__ == "__main__":
    init_schema()
    print("DB schema initialized.")
