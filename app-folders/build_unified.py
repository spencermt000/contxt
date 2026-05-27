#!/usr/bin/env python3
"""Build parsed_data/unified.db — ordered action chains, tool detail, file-history links."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from action_types import family_for_kind  # noqa: E402
from find_data import APP_ROOTS, discover  # noqa: E402
from tool_classify import classify_tool  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_DB = REPO_ROOT / "parsed_data" / "unified.db"
CLAUDE_FILE_HISTORY = APP_ROOTS["claude"] / "file-history"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    raw_session_id TEXT,
    app TEXT NOT NULL,
    project_key TEXT,
    source_path TEXT NOT NULL,
    cwd TEXT,
    action_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chain (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    line_no INTEGER,
    sub_seq INTEGER NOT NULL DEFAULT 0,
    app TEXT NOT NULL,
    kind TEXT NOT NULL,
    family TEXT NOT NULL,
    role TEXT,
    action_timestamp TEXT,
    message_uuid TEXT,
    parent_uuid TEXT,
    tool_name TEXT,
    tool_action TEXT,
    target_path TEXT,
    command TEXT,
    tool_use_id TEXT,
    text TEXT NOT NULL,
    file_history_backup TEXT,
    file_history_target TEXT,
    extra_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS file_history_index (
    session_id TEXT NOT NULL,
    backup_name TEXT NOT NULL,
    disk_path TEXT NOT NULL,
    linked_file_path TEXT,
    message_uuid TEXT,
    version INTEGER,
    backup_time TEXT,
    PRIMARY KEY (disk_path)
);

CREATE TABLE IF NOT EXISTS tool_usage (
    app TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_action TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (app, tool_name, tool_action)
);

CREATE INDEX IF NOT EXISTS idx_chain_session ON chain(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_chain_tool ON chain(tool_name, tool_action);
CREATE INDEX IF NOT EXISTS idx_chain_target ON chain(target_path);
CREATE INDEX IF NOT EXISTS idx_chain_uuid ON chain(message_uuid);
CREATE INDEX IF NOT EXISTS idx_fh_session ON file_history_index(session_id);
"""


@dataclass
class ChainStep:
    session_id: str
    seq: int
    line_no: int | None
    sub_seq: int
    app: str
    kind: str
    family: str
    role: str | None
    action_timestamp: str | None
    message_uuid: str | None = None
    parent_uuid: str | None = None
    tool_name: str | None = None
    tool_action: str | None = None
    target_path: str | None = None
    command: str | None = None
    tool_use_id: str | None = None
    text: str = ""
    file_history_backup: str | None = None
    file_history_target: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


def index_claude_file_history(conn: sqlite3.Connection) -> int:
    if not CLAUDE_FILE_HISTORY.exists():
        return 0
    n = 0
    for session_dir in CLAUDE_FILE_HISTORY.iterdir():
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        for fpath in session_dir.iterdir():
            if not fpath.is_file():
                continue
            backup_name = fpath.name
            version = None
            if "@v" in backup_name:
                try:
                    version = int(backup_name.rsplit("@v", 1)[1])
                except ValueError:
                    pass
            conn.execute(
                """
                INSERT OR IGNORE INTO file_history_index
                (session_id, backup_name, disk_path, linked_file_path, message_uuid, version, backup_time)
                VALUES (?, ?, ?, NULL, NULL, ?, NULL)
                """,
                (session_id, backup_name, str(fpath), version),
            )
            n += 1
    conn.commit()
    return n


def link_snapshots_to_file_history(conn: sqlite3.Connection) -> int:
    """Update file_history_index from file-history-snapshot rows in chain extras."""
    updated = 0
    rows = conn.execute(
        """
        SELECT session_id, message_uuid, extra_json FROM chain
        WHERE kind = 'file_history_snapshot'
        """
    ).fetchall()
    for session_id, message_uuid, extra_json in rows:
        row = conn.execute(
            "SELECT raw_session_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        raw_sid = row[0] if row else session_id.split("|", 2)[1] if "|" in session_id else session_id
        extra = json.loads(extra_json)
        backups = extra.get("trackedFileBackups") or {}
        for file_path, info in backups.items():
            if not isinstance(info, dict):
                continue
            backup_name = info.get("backupFileName")
            if not backup_name:
                continue
            disk = CLAUDE_FILE_HISTORY / raw_sid / backup_name
            conn.execute(
                """
                UPDATE file_history_index SET
                    linked_file_path = ?,
                    message_uuid = ?,
                    backup_time = ?
                WHERE session_id = ? AND backup_name = ?
                """,
                (
                    file_path,
                    message_uuid,
                    info.get("backupTime"),
                    raw_sid,
                    backup_name,
                ),
            )
            if not disk.exists():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO file_history_index
                    (session_id, backup_name, disk_path, linked_file_path, message_uuid, version, backup_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        raw_sid,
                        backup_name,
                        str(disk),
                        file_path,
                        message_uuid,
                        info.get("version"),
                        info.get("backupTime"),
                    ),
                )
            updated += 1
    conn.commit()
    return updated


def _append_step(steps: list[ChainStep], step: ChainStep) -> None:
    step.seq = len(steps)
    steps.append(step)


def _base_extra(obj: dict) -> dict[str, Any]:
    return {
        k: obj[k]
        for k in ("uuid", "parentUuid", "sessionId", "cwd", "gitBranch", "version")
        if k in obj and obj[k] is not None
    }


def build_claude_chain(path: Path, session_id: str, project_key: str | None) -> list[ChainStep]:
    steps: list[ChainStep] = []
    cwd: str | None = None
    # latest backup per target file at each point (for linking edits)
    latest_backup: dict[str, str] = {}

    for line_no, obj in iter_jsonl(path):
        ts = obj.get("timestamp")
        uuid = obj.get("uuid")
        parent = obj.get("parentUuid")
        if obj.get("cwd"):
            cwd = obj["cwd"]
        otype = obj.get("type")

        if otype == "file-history-snapshot":
            snap = obj.get("snapshot") or {}
            backups = snap.get("trackedFileBackups") or {}
            for fp, info in backups.items():
                if isinstance(info, dict) and info.get("backupFileName"):
                    latest_backup[fp] = info["backupFileName"]
            _append_step(
                steps,
                ChainStep(
                    session_id=session_id,
                    seq=0,
                    line_no=line_no,
                    sub_seq=0,
                    app="claude",
                    kind="file_history_snapshot",
                    family="context",
                    role=None,
                    action_timestamp=ts or snap.get("timestamp"),
                    message_uuid=obj.get("messageId") or uuid,
                    text=f"snapshot:{len(backups)} file(s)",
                    extra={"trackedFileBackups": backups, "isSnapshotUpdate": obj.get("isSnapshotUpdate")},
                ),
            )
            continue

        if otype in ("queue-operation",):
            continue

        if otype == "attachment":
            att = obj.get("attachment") or {}
            _append_step(
                steps,
                ChainStep(
                    session_id=session_id,
                    seq=0,
                    line_no=line_no,
                    sub_seq=0,
                    app="claude",
                    kind="attachment",
                    family="context",
                    role=None,
                    action_timestamp=ts,
                    message_uuid=uuid,
                    parent_uuid=parent,
                    text=str(att.get("type", "attachment")),
                    extra={**_base_extra(obj), "attachment": att},
                ),
            )
            continue

        msg = obj.get("message") or {}
        role = msg.get("role") or otype
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue

        sub = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")

            if btype == "thinking":
                _append_step(
                    steps,
                    ChainStep(
                        session_id=session_id,
                        seq=0,
                        line_no=line_no,
                        sub_seq=sub,
                        app="claude",
                        kind="thinking",
                        family="thinking",
                        role="assistant",
                        action_timestamp=ts,
                        message_uuid=uuid,
                        parent_uuid=parent,
                        text=str(block.get("thinking", "")),
                        extra=_base_extra(obj),
                    ),
                )
                sub += 1
            elif btype == "tool_use":
                name = block.get("name", "tool")
                classified = classify_tool("claude", name, block.get("input"))
                target = classified.get("target_path")
                fh_backup = latest_backup.get(target) if target else None
                disk_path = (
                    str(CLAUDE_FILE_HISTORY / session_id / fh_backup) if fh_backup else None
                )
                _append_step(
                    steps,
                    ChainStep(
                        session_id=session_id,
                        seq=0,
                        line_no=line_no,
                        sub_seq=sub,
                        app="claude",
                        kind="tool_call",
                        family="tools",
                        role="assistant",
                        action_timestamp=ts,
                        message_uuid=uuid,
                        parent_uuid=parent,
                        tool_name=classified["tool_name"],
                        tool_action=classified["tool_action"],
                        target_path=target,
                        command=classified.get("command"),
                        tool_use_id=block.get("id"),
                        text=classified.get("input_summary") or f"[tool:{name}]",
                        file_history_backup=disk_path,
                        file_history_target=target,
                        extra={**_base_extra(obj), "input": classified.get("input_json")},
                    ),
                )
                sub += 1
            elif btype == "tool_result":
                _append_step(
                    steps,
                    ChainStep(
                        session_id=session_id,
                        seq=0,
                        line_no=line_no,
                        sub_seq=sub,
                        app="claude",
                        kind="tool_result",
                        family="tools",
                        role="user",
                        action_timestamp=ts,
                        message_uuid=uuid,
                        parent_uuid=parent,
                        tool_use_id=block.get("tool_use_id"),
                        text=str(block.get("content", ""))[:8000],
                        extra={
                            **_base_extra(obj),
                            "is_error": block.get("is_error"),
                        },
                    ),
                )
                sub += 1
            elif btype == "text":
                text = str(block.get("text", ""))
                if text.strip():
                    _append_step(
                        steps,
                        ChainStep(
                            session_id=session_id,
                            seq=0,
                            line_no=line_no,
                            sub_seq=sub,
                            app="claude",
                            kind="message",
                            family=family_for_kind("message", role, text),
                            role=role,
                            action_timestamp=ts,
                            message_uuid=uuid,
                            parent_uuid=parent,
                            text=text,
                            extra=_base_extra(obj),
                        ),
                    )
                    sub += 1

    return steps


def build_cursor_chain(path: Path, session_id: str, project_key: str | None) -> list[ChainStep]:
    steps: list[ChainStep] = []
    for line_no, obj in iter_jsonl(path):
        role = obj.get("role")
        msg = obj.get("message") or {}
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue
        sub = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = str(block.get("text", ""))
                if text.strip():
                    _append_step(
                        steps,
                        ChainStep(
                            session_id=session_id,
                            seq=0,
                            line_no=line_no,
                            sub_seq=sub,
                            app="cursor",
                            kind="message",
                            family=family_for_kind("message", role, text),
                            role=role,
                            action_timestamp=None,
                            text=text,
                            extra={},
                        ),
                    )
                    sub += 1
            elif block.get("type") == "tool_use":
                name = block.get("name", "tool")
                classified = classify_tool("cursor", name, block.get("input"))
                _append_step(
                    steps,
                    ChainStep(
                        session_id=session_id,
                        seq=0,
                        line_no=line_no,
                        sub_seq=sub,
                        app="cursor",
                        kind="tool_call",
                        family="tools",
                        role=role or "assistant",
                        action_timestamp=None,
                        tool_name=classified["tool_name"],
                        tool_action=classified["tool_action"],
                        target_path=classified.get("target_path"),
                        command=classified.get("command"),
                        text=classified.get("input_summary") or f"[tool:{name}]",
                        extra={"input": classified.get("input_json")},
                    ),
                )
                sub += 1
    return steps


def build_codex_chain(path: Path, session_id: str) -> list[ChainStep]:
    steps: list[ChainStep] = []
    cwd: str | None = None
    for line_no, obj in iter_jsonl(path):
        ts = obj.get("timestamp")
        t = obj.get("type")
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}

        if t == "session_meta":
            cwd = payload.get("cwd")
            _append_step(
                steps,
                ChainStep(
                    session_id=session_id,
                    seq=0,
                    line_no=line_no,
                    sub_seq=0,
                    app="codex",
                    kind="meta",
                    family="context",
                    role=None,
                    action_timestamp=ts,
                    text=f"cwd={cwd}",
                    extra={"cwd": cwd},
                ),
            )
            continue

        if t == "response_item" and payload.get("type") == "message":
            role = payload.get("role")
            if role == "developer":
                continue
            sub = 0
            for block in payload.get("content") or []:
                if not isinstance(block, dict):
                    continue
                txt = block.get("text") or block.get("input_text") or block.get("output_text")
                if txt and str(txt).strip():
                    text = str(txt)
                    _append_step(
                        steps,
                        ChainStep(
                            session_id=session_id,
                            seq=0,
                            line_no=line_no,
                            sub_seq=sub,
                            app="codex",
                            kind="message",
                            family=family_for_kind("message", role, text),
                            role=role,
                            action_timestamp=ts,
                            text=text,
                            extra={"cwd": cwd},
                        ),
                    )
                    sub += 1
            continue

        if t == "event_msg":
            ev = payload.get("type")
            if ev == "user_message":
                text = str(payload.get("message", ""))
                _append_step(
                    steps,
                    ChainStep(
                        session_id=session_id,
                        seq=0,
                        line_no=line_no,
                        sub_seq=0,
                        app="codex",
                        kind="message",
                        family=family_for_kind("message", "user", text),
                        role="user",
                        action_timestamp=ts,
                        text=text,
                        extra={"cwd": cwd},
                    ),
                )
            elif ev == "agent_message":
                text = str(payload.get("message", ""))
                _append_step(
                    steps,
                    ChainStep(
                        session_id=session_id,
                        seq=0,
                        line_no=line_no,
                        sub_seq=0,
                        app="codex",
                        kind="message",
                        family=family_for_kind("message", "assistant", text),
                        role="assistant",
                        action_timestamp=ts,
                        text=text,
                        extra={"cwd": cwd},
                    ),
                )
    return steps


def unified_session_id(app: str, session_id: str, source_path: str) -> str:
    """Globally unique session key (one chain per source file)."""
    digest = hashlib.sha1(source_path.encode()).hexdigest()[:8]
    return f"{app}|{session_id}|{digest}"


def write_chain(conn: sqlite3.Connection, session_id: str, app: str, project_key: str | None, source_path: str, steps: list[ChainStep]) -> None:
    uid = unified_session_id(app, session_id, source_path)
    for s in steps:
        s.session_id = uid
    cwd = None
    for s in steps:
        if s.extra.get("cwd"):
            cwd = s.extra["cwd"]
            break
    conn.execute(
        """
        INSERT OR REPLACE INTO sessions (session_id, raw_session_id, app, project_key, source_path, cwd, action_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (uid, session_id, app, project_key, source_path, cwd, len(steps)),
    )
    for s in steps:
        conn.execute(
            """
            INSERT INTO chain (
                session_id, seq, line_no, sub_seq, app, kind, family, role, action_timestamp,
                message_uuid, parent_uuid, tool_name, tool_action, target_path, command,
                tool_use_id, text, file_history_backup, file_history_target, extra_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                s.session_id,
                s.seq,
                s.line_no,
                s.sub_seq,
                s.app,
                s.kind,
                s.family,
                s.role,
                s.action_timestamp,
                s.message_uuid,
                s.parent_uuid,
                s.tool_name,
                s.tool_action,
                s.target_path,
                s.command,
                s.tool_use_id,
                s.text,
                s.file_history_backup,
                s.file_history_target,
                json.dumps(s.extra, ensure_ascii=False),
            ),
        )


def aggregate_tool_usage(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM tool_usage")
    conn.execute(
        """
        INSERT INTO tool_usage (app, tool_name, tool_action, count)
        SELECT app, tool_name, tool_action, COUNT(*) FROM chain
        WHERE kind = 'tool_call' AND tool_name IS NOT NULL
        GROUP BY app, tool_name, tool_action
        """
    )
    conn.commit()


def print_report(conn: sqlite3.Connection) -> None:
    total = conn.execute("SELECT COUNT(*) FROM chain").fetchone()[0]
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    fh = conn.execute("SELECT COUNT(*) FROM file_history_index").fetchone()[0]
    linked = conn.execute(
        "SELECT COUNT(*) FROM file_history_index WHERE linked_file_path IS NOT NULL"
    ).fetchone()[0]

    print(f"\nunified.db: {total:,} chain steps across {sessions} sessions")
    print(f"file-history: {fh:,} backup files on disk, {linked:,} linked to source paths\n")

    print("Tool usage (top 20):")
    for row in conn.execute(
        """
        SELECT app, tool_name, tool_action, count FROM tool_usage
        ORDER BY count DESC LIMIT 20
        """
    ):
        print(f"  {row[0]:8} {row[1]:16} {row[2]:20} {row[3]:5}")

    print("\nBash sub-actions (top 12):")
    for row in conn.execute(
        """
        SELECT tool_action, COUNT(*) FROM chain
        WHERE kind='tool_call' AND tool_name IN ('Bash','Shell')
        GROUP BY tool_action ORDER BY 2 DESC LIMIT 12
        """
    ):
        print(f"  {row[0]:20} {row[1]:5}")

    print("\nFile-targeted tools (top paths):")
    for row in conn.execute(
        """
        SELECT tool_action, target_path, COUNT(*) AS n FROM chain
        WHERE kind='tool_call' AND target_path IS NOT NULL
        GROUP BY tool_action, target_path ORDER BY n DESC LIMIT 8
        """
    ):
        print(f"  {row[0]:16} {row[2]:3}  {row[1][:70]}")

    print("\nSample chain (yompute session if present):")
    sid_row = conn.execute(
        """
        SELECT session_id FROM sessions
        WHERE project_key LIKE '%yompute%' OR source_path LIKE '%yompute%'
        LIMIT 1
        """
    ).fetchone()
    if sid_row:
        sid = sid_row[0]
        for row in conn.execute(
            """
            SELECT seq, kind, tool_name, tool_action, substr(text,1,70), file_history_backup IS NOT NULL
            FROM chain WHERE session_id = ? ORDER BY seq LIMIT 15
            """,
            (sid,),
        ):
            fh = " [FH]" if row[5] else ""
            tool = f" {row[2]}/{row[3]}" if row[2] else ""
            print(f"  {row[0]:3} {row[1]:22}{tool}{fh}  {row[4]}")


def build_unified(
    *,
    sources: list[str],
    project_filter: str | None = None,
    replace: bool = True,
) -> Path:
    if replace and UNIFIED_DB.exists():
        UNIFIED_DB.unlink()

    conn = sqlite3.connect(UNIFIED_DB)
    conn.executescript(CREATE_SQL)

    files = discover(sources)
    if project_filter:
        needle = project_filter.lower()
        files = [f for f in files if f.project_key and needle in f.project_key.lower()]

    session_files = [f for f in files if f.path.endswith(".jsonl") and f.category in ("session", "sessions", "agent-transcripts")]

    for df in session_files:
        path = Path(df.path)
        raw_session_id = df.session_id or path.stem
        if df.app == "claude":
            steps = build_claude_chain(path, raw_session_id, df.project_key)
        elif df.app == "cursor":
            steps = build_cursor_chain(path, raw_session_id, df.project_key)
        elif df.app == "codex":
            steps = build_codex_chain(path, raw_session_id)
        else:
            continue
        if steps:
            write_chain(conn, raw_session_id, df.app, df.project_key, df.path, steps)

    conn.commit()
    index_claude_file_history(conn)
    link_snapshots_to_file_history(conn)
    aggregate_tool_usage(conn)

    # Link chain tool_calls to file_history (raw session id = part after first colon)
    conn.execute(
        """
        UPDATE chain SET file_history_backup = (
            SELECT disk_path FROM file_history_index fh
            WHERE fh.session_id = (
                SELECT raw_session_id FROM sessions s WHERE s.session_id = chain.session_id
            )
              AND fh.linked_file_path = chain.target_path
            ORDER BY fh.version DESC LIMIT 1
        )
        WHERE kind = 'tool_call'
          AND file_history_backup IS NULL
          AND target_path IS NOT NULL
          AND chain.app = 'claude'
        """
    )
    conn.commit()
    conn.close()
    return UNIFIED_DB


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unified action chain database")
    ap.add_argument("--sources", default="codex,cursor,claude")
    ap.add_argument("--project", help="Filter by project_key substring")
    ap.add_argument("--no-replace", action="store_true")
    ap.add_argument("--query", help="Run SQL after build and print rows")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    db = build_unified(sources=sources, project_filter=args.project, replace=not args.no_replace)
    print(f"Built {db}")

    conn = sqlite3.connect(db)
    print_report(conn)

    if args.query:
        print(f"\nQuery: {args.query}\n")
        for row in conn.execute(args.query):
            print(row)
    conn.close()


if __name__ == "__main__":
    main()
