"""Database — async SQLite for content storage and DNA profiles.

FIX: Semaphore to prevent concurrent write corruption on single connection.
"""
import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiosqlite


class Database:
    def __init__(self, db_path: str = "contentai.db"):
        self._path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._write_lock = asyncio.Semaphore(1)  # Serialize writes

    async def init(self):
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS content (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                content_type TEXT DEFAULT 'blog_post',
                stage TEXT DEFAULT 'draft',
                body TEXT,
                metadata TEXT DEFAULT '{}',
                dna_score REAL DEFAULT 0.0,
                debate_passed INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS dna_profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                samples_count INTEGER DEFAULT 0,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS atomized (
                id TEXT PRIMARY KEY,
                content_id TEXT REFERENCES content(id),
                platform TEXT NOT NULL,
                variant TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS debate_logs (
                id TEXT PRIMARY KEY,
                content_id TEXT REFERENCES content(id),
                round INTEGER,
                advocate TEXT,
                critic TEXT,
                judge_score REAL,
                judge_verdict TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS content_versions (
                id TEXT PRIMARY KEY,
                content_id TEXT REFERENCES content(id),
                stage TEXT NOT NULL,
                body TEXT,
                metadata TEXT DEFAULT '{}',
                version_num INTEGER DEFAULT 1,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_content_versions_content_id ON content_versions(content_id);
        """)

    async def save_content(self, topic: str, body: str, content_type: str = "blog_post",
                           stage: str = "draft", metadata: dict = None, dna_score: float = 0.0,
                           debate_passed: bool = False) -> str:
        cid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        async with self._write_lock:
            await self._conn.execute(
                "INSERT INTO content (id, topic, content_type, stage, body, metadata, dna_score, debate_passed, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (cid, topic, content_type, stage, body, json.dumps(metadata or {}), dna_score, int(debate_passed), now, now)
            )
            await self._conn.commit()
        return cid

    async def get_content(self, cid: str) -> Optional[Dict]:
        cursor = await self._conn.execute("SELECT * FROM content WHERE id = ?", (cid,))
        row = await cursor.fetchone()
        if row:
            d = dict(row)
            d["metadata"] = json.loads(d["metadata"])
            return d
        return None

    async def save_dna_profile(self, name: str, fingerprint: dict, samples_count: int) -> str:
        pid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        async with self._write_lock:
            await self._conn.execute(
                "INSERT INTO dna_profiles (id, name, fingerprint, samples_count, created_at) VALUES (?,?,?,?,?)",
                (pid, name, json.dumps(fingerprint), samples_count, now)
            )
            await self._conn.commit()
        return pid

    async def save_debate_log(self, content_id: str, round_num: int, advocate: str,
                               critic: str, judge_score: float, verdict: str) -> str:
        lid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        async with self._write_lock:
            await self._conn.execute(
                "INSERT INTO debate_logs (id, content_id, round, advocate, critic, judge_score, judge_verdict, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (lid, content_id, round_num, advocate, critic, judge_score, verdict, now)
            )
            await self._conn.commit()
        return lid

    async def save_atomized(self, content_id: str, platform: str, variant: str, metadata: dict = None) -> str:
        aid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        async with self._write_lock:
            await self._conn.execute(
                "INSERT INTO atomized (id, content_id, platform, variant, metadata, created_at) VALUES (?,?,?,?,?,?)",
                (aid, content_id, platform, variant, json.dumps(metadata or {}), now)
            )
            await self._conn.commit()
        return aid

    async def save_version(self, content_id: str, stage: str, body: str,
                            metadata: dict = None) -> str:
        async with self._write_lock:
            # Compute next version number for this content_id
            cursor = await self._conn.execute(
                "SELECT COALESCE(MAX(version_num), 0) + 1 FROM content_versions WHERE content_id = ?",
                (content_id,)
            )
            row = await cursor.fetchone()
            next_version = row[0] if row else 1
            vid = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            await self._conn.execute(
                "INSERT INTO content_versions (id, content_id, stage, body, metadata, version_num, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (vid, content_id, stage, body, json.dumps(metadata or {}), next_version, now)
            )
            await self._conn.commit()
            return vid

    async def get_content_history(self, content_id: str) -> List[Dict]:
        cursor = await self._conn.execute(
            "SELECT * FROM content_versions WHERE content_id = ? ORDER BY version_num ASC",
            (content_id,)
        )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d["metadata"])
            result.append(d)
        return result

    async def restore_version(self, content_id: str, version_id: str) -> bool:
        """Restore content body from a specific version and record the restoration."""
        async with self._write_lock:
            cursor = await self._conn.execute(
                "SELECT body, stage, metadata FROM content_versions WHERE id = ? AND content_id = ?",
                (version_id, content_id)
            )
            row = await cursor.fetchone()
            if not row:
                return False
            now = datetime.now(timezone.utc).isoformat()
            await self._conn.execute(
                "UPDATE content SET body = ?, updated_at = ? WHERE id = ?",
                (row["body"], now, content_id)
            )
            # Record the restoration as a new version entry for audit trail
            cursor = await self._conn.execute(
                "SELECT COALESCE(MAX(version_num), 0) + 1 FROM content_versions WHERE content_id = ?",
                (content_id,)
            )
            version_row = await cursor.fetchone()
            next_version = version_row[0] if version_row else 1
            vid = str(uuid.uuid4())
            await self._conn.execute(
                "INSERT INTO content_versions (id, content_id, stage, body, metadata, version_num, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (vid, content_id, row["stage"], row["body"],
                 row["metadata"] or "{}", next_version, now)
            )
            await self._conn.commit()
        return True

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def get_content_list(self, limit: int = 20, offset: int = 0) -> list:
        if not self._conn:
            raise RuntimeError("Database is not initialized. Call init() first.")
        cursor = await self._conn.execute(
            "SELECT id, topic, content_type, stage, dna_score, debate_passed, created_at FROM content "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


db = Database()
