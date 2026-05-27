#!/usr/bin/env python3
"""Store parsed actions in SQLite (default) or PostgreSQL under parsed_data/."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from action_types import FAMILY_DB_NAMES, family_for_kind
from parse_data import ParsedFile, ParsedItem

REPO_ROOT = Path(__file__).resolve().parents[2]
PARSED_DATA_DIR = REPO_ROOT / "parsed_data"
DEFAULT_SQLITE_PATH = PARSED_DATA_DIR / "actions.db"

Backend = Literal["sqlite", "postgres"]

CREATE_SQLITE = """
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app TEXT NOT NULL,
    source_path TEXT NOT NULL,
    category TEXT NOT NULL,
    project_key TEXT,
    session_id TEXT,
    session_date TEXT,
    source_size_bytes INTEGER,
    source_modified_at TEXT,
    line_no INTEGER,
    kind TEXT NOT NULL,
    family TEXT NOT NULL,
    role TEXT,
    action_timestamp TEXT,
    text TEXT NOT NULL,
    extra_json TEXT NOT NULL DEFAULT '{}',
    text_hash TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    UNIQUE (source_path, line_no, kind, role, text_hash)
);
CREATE INDEX IF NOT EXISTS idx_actions_app ON actions(app);
CREATE INDEX IF NOT EXISTS idx_actions_project ON actions(project_key);
CREATE INDEX IF NOT EXISTS idx_actions_session ON actions(session_id);
CREATE INDEX IF NOT EXISTS idx_actions_kind ON actions(kind);
CREATE INDEX IF NOT EXISTS idx_actions_family ON actions(family);
CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON actions(action_timestamp);
"""

CREATE_POSTGRES = """
CREATE TABLE IF NOT EXISTS actions (
    id BIGSERIAL PRIMARY KEY,
    app TEXT NOT NULL,
    source_path TEXT NOT NULL,
    category TEXT NOT NULL,
    project_key TEXT,
    session_id TEXT,
    session_date TEXT,
    source_size_bytes BIGINT,
    source_modified_at TEXT,
    line_no INTEGER,
    kind TEXT NOT NULL,
    family TEXT NOT NULL,
    role TEXT,
    action_timestamp TIMESTAMPTZ,
    text TEXT NOT NULL,
    extra_json JSONB NOT NULL DEFAULT '{}',
    text_hash TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL,
    UNIQUE (source_path, line_no, kind, role, text_hash)
);
CREATE INDEX IF NOT EXISTS idx_actions_app ON actions(app);
CREATE INDEX IF NOT EXISTS idx_actions_project ON actions(project_key);
CREATE INDEX IF NOT EXISTS idx_actions_session ON actions(session_id);
CREATE INDEX IF NOT EXISTS idx_actions_kind ON actions(kind);
CREATE INDEX IF NOT EXISTS idx_actions_family ON actions(family);
CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON actions(action_timestamp);
"""

INSERT_SQL = """
INSERT INTO actions (
    app, source_path, category, project_key, session_id, session_date,
    source_size_bytes, source_modified_at, line_no, kind, family, role,
    action_timestamp, text, extra_json, text_hash, ingested_at
) VALUES ({ph})
ON CONFLICT (source_path, line_no, kind, role, text_hash) DO NOTHING
"""

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_sqlite_path(path: Path | str | None) -> Path:
    if path is None:
        return DEFAULT_SQLITE_PATH
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def connect_sqlite(path: Path | str | None = None) -> sqlite3.Connection:
    db_path = resolve_sqlite_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    existing = {r[1] for r in conn.execute("PRAGMA table_info(actions)")}
    if existing and "family" not in existing:
        conn.execute("DROP TABLE actions")
        conn.commit()
    conn.executescript(CREATE_SQLITE)
    conn.commit()
    return conn


def connect_postgres(url: str) -> Any:
    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit(
            "PostgreSQL requires psycopg: pip install 'psycopg[binary]'"
        ) from exc
    conn = psycopg.connect(url)
    with conn.cursor() as cur:
        cur.execute(CREATE_POSTGRES)
    conn.commit()
    return conn


def connect(
    *,
    backend: Backend = "sqlite",
    sqlite_path: Path | str | None = None,
    postgres_url: str | None = None,
) -> tuple[Any, Backend]:
    if backend == "postgres":
        if not postgres_url:
            raise ValueError("postgres_url required for postgres backend")
        return connect_postgres(postgres_url), "postgres"
    return connect_sqlite(sqlite_path), "sqlite"


def _row_from_item(pf: ParsedFile, item: ParsedItem, ingested_at: str) -> tuple:
    src = pf.source
    extra = json.dumps(item.extra, ensure_ascii=False)
    family = family_for_kind(item.kind, item.role, item.text)
    return (
        item.app,
        item.source_path,
        item.category,
        item.project_key,
        item.session_id,
        src.session_date,
        src.size_bytes,
        src.modified_at,
        item.line_no,
        item.kind,
        family,
        item.role,
        item.timestamp,
        item.text,
        extra,
        text_hash(item.text),
        ingested_at,
    )


def clear_actions(conn: Any, backend: Backend) -> None:
    conn.execute("DELETE FROM actions")
    conn.commit()


def store_parsed(
    parsed: list[ParsedFile],
    conn: Any,
    backend: Backend,
    *,
    replace: bool = False,
) -> dict[str, int]:
    if replace:
        clear_actions(conn, backend)

    ingested_at = utc_now()
    ph = ", ".join(["%s"] * 17) if backend == "postgres" else ", ".join(["?"] * 17)
    sql = INSERT_SQL.format(ph=ph)

    rows: list[tuple] = []
    for pf in parsed:
        for item in pf.items:
            rows.append(_row_from_item(pf, item, ingested_at))

    if not rows:
        return {"inserted": 0, "skipped_files": len(parsed), "actions": 0}

    before = _count_actions(conn, backend)
    if backend == "postgres":
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
    else:
        conn.executemany(sql, rows)
    conn.commit()
    after = _count_actions(conn, backend)
    inserted = after - before if not replace else after
    return {
        "inserted": inserted,
        "total_rows": after,
        "actions_parsed": len(rows),
        "files": len(parsed),
    }


def _count_actions(conn: Any, backend: Backend) -> int:
    if backend == "postgres":
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM actions")
            row = cur.fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) FROM actions").fetchone()
    return int(row[0])


def store_to_sqlite(
    parsed: list[ParsedFile],
    path: Path | str | None = None,
    *,
    replace: bool = False,
) -> dict[str, int]:
    conn = connect_sqlite(path)
    try:
        return store_parsed(parsed, conn, "sqlite", replace=replace)
    finally:
        conn.close()


def store_split_by_family(
    parsed: list[ParsedFile],
    base_dir: Path | None = None,
    *,
    replace: bool = True,
) -> dict[str, dict[str, int]]:
    """Write one SQLite DB per family under parsed_data/."""
    base = base_dir or PARSED_DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    ingested_at = utc_now()

    buckets: dict[str, list[tuple]] = {fam: [] for fam in FAMILY_DB_NAMES}
    for pf in parsed:
        for item in pf.items:
            fam = family_for_kind(item.kind, item.role, item.text)
            if fam not in buckets:
                fam = "conversation"
            buckets[fam].append(_row_from_item(pf, item, ingested_at))

    stats: dict[str, dict[str, int]] = {}
    ph = ", ".join(["?"] * 17)
    sql = INSERT_SQL.format(ph=ph)

    for fam, db_name in FAMILY_DB_NAMES.items():
        rows = buckets.get(fam, [])
        path = base / db_name
        conn = connect_sqlite(path)
        try:
            if replace:
                clear_actions(conn, "sqlite")
            if rows:
                conn.executemany(sql, rows)
            conn.commit()
            total = _count_actions(conn, "sqlite")
            stats[fam] = {"db": str(path), "rows": total, "written": len(rows)}
        finally:
            conn.close()

    return stats


def store_to_postgres(
    parsed: list[ParsedFile],
    url: str,
    *,
    replace: bool = False,
) -> dict[str, int]:
    conn, backend = connect(backend="postgres", postgres_url=url)
    try:
        return store_parsed(parsed, conn, backend, replace=replace)
    finally:
        conn.close()
