from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import Iterable
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("SONG_RECOMMENDER_EVALUATION_DB_PATH", str(ROOT / "data" / "evaluation.sqlite3")))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _evaluation_responses_needs_migration(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "evaluation_responses"):
        return False

    columns = conn.execute("PRAGMA table_info(evaluation_responses)").fetchall()
    expected_names = {"session_id", "model_id", "recommendation_spotify_id", "rating", "responded_at"}
    actual_names = {str(column["name"]) for column in columns}
    expected_pk = {
        "session_id": 1,
        "model_id": 2,
        "recommendation_spotify_id": 3,
    }
    actual_pk = {str(column["name"]): int(column["pk"]) for column in columns if int(column["pk"]) > 0}
    return actual_names != expected_names or actual_pk != expected_pk


def _migrate_evaluation_responses(conn: sqlite3.Connection) -> None:
    legacy_table = "evaluation_responses_legacy"
    conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")
    conn.execute("ALTER TABLE evaluation_responses RENAME TO evaluation_responses_legacy")
    conn.execute(
        """
        CREATE TABLE evaluation_responses (
            session_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            recommendation_spotify_id TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK (rating IN (-1, 0, 1, 2)),
            responded_at TEXT NOT NULL,
            PRIMARY KEY (session_id, model_id, recommendation_spotify_id),
            FOREIGN KEY (session_id, model_id, recommendation_spotify_id)
                REFERENCES evaluation_items(session_id, model_id, recommendation_spotify_id)
                ON DELETE CASCADE
        )
        """
    )

    legacy_columns = {
        str(column["name"])
        for column in conn.execute(f"PRAGMA table_info({legacy_table})").fetchall()
    }
    if "model_id" in legacy_columns:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO evaluation_responses (
                session_id,
                model_id,
                recommendation_spotify_id,
                rating,
                responded_at
            )
            SELECT
                session_id,
                model_id,
                recommendation_spotify_id,
                rating,
                responded_at
            FROM {legacy_table}
            """
        )
    else:
        conn.execute(
            f"""
            INSERT OR IGNORE INTO evaluation_responses (
                session_id,
                model_id,
                recommendation_spotify_id,
                rating,
                responded_at
            )
            SELECT
                legacy.session_id,
                item.model_id,
                legacy.recommendation_spotify_id,
                legacy.rating,
                legacy.responded_at
            FROM {legacy_table} AS legacy
            JOIN evaluation_items AS item
              ON item.session_id = legacy.session_id
             AND item.recommendation_spotify_id = legacy.recommendation_spotify_id
            """
        )
    conn.execute(f"DROP TABLE {legacy_table}")


def ensure_schema() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluation_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                user_agent TEXT
            );

            CREATE TABLE IF NOT EXISTS evaluation_items (
                session_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_label TEXT NOT NULL,
                query_split TEXT NOT NULL,
                query_spotify_id TEXT NOT NULL,
                query_name TEXT NOT NULL,
                query_artist TEXT NOT NULL,
                recommendation_spotify_id TEXT NOT NULL,
                recommendation_name TEXT NOT NULL,
                recommendation_artist TEXT NOT NULL,
                recommendation_rank INTEGER NOT NULL,
                similarity REAL,
                PRIMARY KEY (session_id, model_id, recommendation_spotify_id),
                FOREIGN KEY (session_id) REFERENCES evaluation_sessions(session_id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_responses (
                session_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                recommendation_spotify_id TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating IN (-1, 0, 1, 2)),
                responded_at TEXT NOT NULL,
                PRIMARY KEY (session_id, model_id, recommendation_spotify_id),
                FOREIGN KEY (session_id, model_id, recommendation_spotify_id)
                    REFERENCES evaluation_items(session_id, model_id, recommendation_spotify_id)
                    ON DELETE CASCADE
            )
            """
        )
        if _evaluation_responses_needs_migration(conn):
            _migrate_evaluation_responses(conn)


def create_session(items: Iterable[dict[str, object]], user_agent: str | None = None) -> tuple[str, str]:
    ensure_schema()
    session_id = str(uuid4())
    created_at = utc_now_iso()
    rows = list(items)

    with connect() as conn:
        conn.execute(
            "INSERT INTO evaluation_sessions (session_id, created_at, user_agent) VALUES (?, ?, ?)",
            (session_id, created_at, user_agent or ""),
        )
        conn.executemany(
            """
            INSERT INTO evaluation_items (
                session_id,
                model_id,
                model_label,
                query_split,
                query_spotify_id,
                query_name,
                query_artist,
                recommendation_spotify_id,
                recommendation_name,
                recommendation_artist,
                recommendation_rank,
                similarity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    row["model_id"],
                    row["model_label"],
                    row["query_split"],
                    row["query_spotify_id"],
                    row["query_name"],
                    row["query_artist"],
                    row["recommendation_spotify_id"],
                    row["recommendation_name"],
                    row["recommendation_artist"],
                    row["recommendation_rank"],
                    row["similarity"],
                )
                for row in rows
            ],
        )
    return session_id, created_at


def save_response(session_id: str, model_id: str, recommendation_spotify_id: str, rating: int) -> dict[str, int]:
    ensure_schema()
    responded_at = utc_now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO evaluation_responses (
                session_id,
                model_id,
                recommendation_spotify_id,
                rating,
                responded_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, model_id, recommendation_spotify_id)
            DO UPDATE SET
                rating = excluded.rating,
                responded_at = excluded.responded_at
            """,
            (session_id, model_id, recommendation_spotify_id, int(rating), responded_at),
        )
        counts = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM evaluation_items WHERE session_id = ?) AS total_items,
                (SELECT COUNT(*) FROM evaluation_responses WHERE session_id = ?) AS rated_items
            """,
            (session_id, session_id),
        ).fetchone()
    return {
        "total_items": int(counts["total_items"]),
        "rated_items": int(counts["rated_items"]),
    }
