"""SQLite usage tracking."""
import sqlite3
import time
from config.settings import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_type TEXT NOT NULL,
            topic TEXT,
            brand_voice TEXT,
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_usage(content_type: str, topic: str = "", brand_voice: str = ""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO usage_log (content_type, topic, brand_voice, created_at) VALUES (?,?,?,?)",
        (content_type, topic, brand_voice, time.time())
    )
    conn.commit()
    conn.close()

def get_usage_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT content_type, COUNT(*) FROM usage_log GROUP BY content_type")
    rows = cur.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}
