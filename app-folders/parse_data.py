#!/usr/bin/env python3
"""Parse .json / .jsonl / .txt / .md files discovered under Codex, Cursor, and Claude."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Allow running as a script from repo root or this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_data import DataFile, discover  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "parsed_data" / "actions.db"

PARSEABLE_SUFFIXES = {".json", ".jsonl", ".txt", ".md"}
DEFAULT_SKIP_CATEGORIES = {"todos", "file-history"}

# Skip Codex developer/system blobs unless requested (huge instruction prompts)
CODEX_SKIP_ROLES = {"developer"}


@dataclass
class ParsedItem:
    """One extracted unit from a source file."""

    app: str
    source_path: str
    category: str
    project_key: str | None
    session_id: str | None
    line_no: int | None
    kind: str
    role: str | None
    timestamp: str | None
    text: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedFile:
    source: DataFile
    items: list[ParsedItem] = field(default_factory=list)
    error: str | None = None


def _trunc(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n… [{len(text) - max_chars} chars truncated]"


def _item(
    df: DataFile,
    *,
    line_no: int | None,
    kind: str,
    role: str | None,
    timestamp: str | None,
    text: str,
    extra: dict[str, Any] | None = None,
    max_chars: int | None = None,
) -> ParsedItem | None:
    text = text.strip()
    if not text:
        return None
    return ParsedItem(
        app=df.app,
        source_path=df.path,
        category=df.category,
        project_key=df.project_key,
        session_id=df.session_id,
        line_no=line_no,
        kind=kind,
        role=role,
        timestamp=timestamp,
        text=_trunc(text, max_chars),
        extra=extra or {},
    )


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, {"__parse_error__": str(exc), "__raw__": line[:500]}


def blocks_to_text(blocks: Any) -> str:
    """Plain text blocks only (no thinking/tools)."""
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(str(block.get("text", "")))
        elif btype in ("input_text", "output_text"):
            parts.append(str(block.get("text") or block.get("input_text") or block.get("output_text") or ""))
    return "\n".join(p for p in parts if p)


def items_from_content_blocks(
    df: DataFile,
    *,
    line_no: int,
    role: str | None,
    timestamp: str | None,
    content: Any,
    extra: dict[str, Any] | None,
    max_chars: int | None,
) -> list[ParsedItem]:
    """Split message content into separate action rows by block type."""
    items: list[ParsedItem] = []
    if not isinstance(content, list):
        return items

    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")

        if btype == "thinking":
            if it := _item(
                df,
                line_no=line_no,
                kind="thinking",
                role="assistant",
                timestamp=timestamp,
                text=str(block.get("thinking", "")),
                extra=extra,
                max_chars=max_chars,
            ):
                items.append(it)
        elif btype == "tool_use":
            name = block.get("name", "tool")
            if it := _item(
                df,
                line_no=line_no,
                kind="tool_call",
                role=role or "assistant",
                timestamp=timestamp,
                text=f"[tool:{name}]",
                extra={**(extra or {}), "tool": name, "input": block.get("input")},
                max_chars=max_chars,
            ):
                items.append(it)
        elif btype == "tool_result":
            if it := _item(
                df,
                line_no=line_no,
                kind="tool_result",
                role="user",
                timestamp=timestamp,
                text=str(block.get("content", block.get("output", "")))[:max_chars or 8000],
                extra={**(extra or {}), "tool": block.get("name", "tool")},
                max_chars=max_chars,
            ):
                items.append(it)
        elif btype == "text":
            text_parts.append(str(block.get("text", "")))

    combined = "\n".join(text_parts)
    if combined.strip():
        if it := _item(
            df,
            line_no=line_no,
            kind="message",
            role=role,
            timestamp=timestamp,
            text=combined,
            extra=extra,
            max_chars=max_chars,
        ):
            items.append(it)
    return items


# --- Codex ---


def parse_codex_jsonl(df: DataFile, path: Path, max_chars: int | None, include_developer: bool) -> list[ParsedItem]:
    items: list[ParsedItem] = []
    session_cwd: str | None = None

    for line_no, obj in iter_jsonl(path):
        if "__parse_error__" in obj:
            if it := _item(
                df,
                line_no=line_no,
                kind="parse_error",
                role=None,
                timestamp=None,
                text=obj["__parse_error__"],
                extra={"raw": obj.get("__raw__")},
                max_chars=max_chars,
            ):
                items.append(it)
            continue

        ts = obj.get("timestamp")
        t = obj.get("type")
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}

        if t == "session_meta":
            session_cwd = payload.get("cwd")
            if it := _item(
                df,
                line_no=line_no,
                kind="meta",
                role=None,
                timestamp=ts,
                text=f"session {payload.get('id', '')} cwd={session_cwd or ''}".strip(),
                extra={"cwd": session_cwd, "originator": payload.get("originator")},
                max_chars=max_chars,
            ):
                items.append(it)
            continue

        if t == "response_item" and payload.get("type") == "message":
            role = payload.get("role")
            if role in CODEX_SKIP_ROLES and not include_developer:
                continue
            text = blocks_to_text(payload.get("content"))
            if it := _item(
                df,
                line_no=line_no,
                kind="message",
                role=role,
                timestamp=ts,
                text=text,
                extra={"cwd": session_cwd},
                max_chars=max_chars,
            ):
                items.append(it)
            continue

        if t == "event_msg":
            ev = payload.get("type")
            if ev == "user_message":
                if it := _item(df, line_no=line_no, kind="message", role="user", timestamp=ts, text=str(payload.get("message", "")), max_chars=max_chars):
                    items.append(it)
            elif ev == "agent_message":
                if it := _item(df, line_no=line_no, kind="message", role="assistant", timestamp=ts, text=str(payload.get("message", "")), max_chars=max_chars):
                    items.append(it)

    return items


# --- Cursor ---


def parse_cursor_jsonl(df: DataFile, path: Path, max_chars: int | None) -> list[ParsedItem]:
    items: list[ParsedItem] = []
    for line_no, obj in iter_jsonl(path):
        if "__parse_error__" in obj:
            if it := _item(df, line_no=line_no, kind="parse_error", role=None, timestamp=None, text=obj["__parse_error__"], max_chars=max_chars):
                items.append(it)
            continue

        role = obj.get("role")
        msg = obj.get("message") or {}
        block_items = items_from_content_blocks(
            df,
            line_no=line_no,
            role=role,
            timestamp=None,
            content=msg.get("content"),
            extra=None,
            max_chars=max_chars,
        )
        items.extend(block_items)

    return items


def parse_cursor_terminal(df: DataFile, path: Path, max_chars: int | None) -> list[ParsedItem]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, str] = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = parts[2].lstrip("\n")
    if it := _item(df, line_no=None, kind="terminal", role=None, timestamp=None, text=body, extra=meta, max_chars=max_chars):
        return [it]
    return []


# --- Claude ---


def parse_claude_jsonl(df: DataFile, path: Path, max_chars: int | None) -> list[ParsedItem]:
    items: list[ParsedItem] = []
    for line_no, obj in iter_jsonl(path):
        if "__parse_error__" in obj:
            if it := _item(df, line_no=line_no, kind="parse_error", role=None, timestamp=None, text=obj["__parse_error__"], max_chars=max_chars):
                items.append(it)
            continue

        ts = obj.get("timestamp")
        otype = obj.get("type")

        if otype in ("queue-operation",):
            continue

        if otype == "attachment":
            att = obj.get("attachment") or {}
            if it := _item(
                df,
                line_no=line_no,
                kind="attachment",
                role=None,
                timestamp=ts,
                text=str(att.get("type", "attachment")),
                extra={"attachment": att},
                max_chars=max_chars,
            ):
                items.append(it)
            continue

        msg = obj.get("message") or {}
        role = msg.get("role") or otype
        cwd = obj.get("cwd")
        extra = {"cwd": cwd} if cwd else {}
        items.extend(
            items_from_content_blocks(
                df,
                line_no=line_no,
                role=role,
                timestamp=ts,
                content=msg.get("content"),
                extra=extra,
                max_chars=max_chars,
            )
        )

    return items


def parse_claude_history(df: DataFile, path: Path, max_chars: int | None) -> list[ParsedItem]:
    items: list[ParsedItem] = []
    for line_no, obj in iter_jsonl(path):
        if "__parse_error__" in obj:
            continue
        text = str(obj.get("display", ""))
        if it := _item(
            df,
            line_no=line_no,
            kind="history",
            role="user",
            timestamp=str(obj.get("timestamp")) if obj.get("timestamp") else None,
            text=text,
            extra={"project": obj.get("project"), "sessionId": obj.get("sessionId")},
            max_chars=max_chars,
        ):
            items.append(it)
    return items


# --- Generic ---


def parse_plain_document(df: DataFile, path: Path, max_chars: int | None) -> list[ParsedItem]:
    text = path.read_text(encoding="utf-8", errors="replace")
    kind = "memory" if df.category == "memory" else "document"
    if it := _item(df, line_no=None, kind=kind, role="memory" if kind == "memory" else None, timestamp=None, text=text, max_chars=max_chars):
        return [it]
    return []


def parse_json_file(df: DataFile, path: Path, max_chars: int | None) -> list[ParsedItem]:
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        if it := _item(df, line_no=None, kind="parse_error", role=None, timestamp=None, text=str(exc), max_chars=max_chars):
            return [it]
        return []

    if isinstance(data, list):
        if not data:
            return []
        text = json.dumps(data, indent=2, ensure_ascii=False)
        kind = "todo_list" if df.category == "todos" else "json_array"
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False)
        kind = "json_object"

    if it := _item(df, line_no=None, kind=kind, role=None, timestamp=None, text=text, max_chars=max_chars):
        return [it]
    return []


def parse_file(
    df: DataFile,
    *,
    max_chars: int | None = 8000,
    include_developer: bool = False,
) -> ParsedFile:
    path = Path(df.path)
    suffix = path.suffix.lower()

    if suffix not in PARSEABLE_SUFFIXES:
        return ParsedFile(source=df, error=f"unsupported suffix: {suffix}")

    try:
        if df.app == "codex" and suffix == ".jsonl":
            items = parse_codex_jsonl(df, path, max_chars, include_developer)
        elif df.app == "cursor" and suffix == ".jsonl":
            items = parse_cursor_jsonl(df, path, max_chars)
        elif df.app == "cursor" and df.category == "terminals" and suffix == ".txt":
            items = parse_cursor_terminal(df, path, max_chars)
        elif df.app == "claude" and suffix == ".jsonl":
            if df.category == "history":
                items = parse_claude_history(df, path, max_chars)
            else:
                items = parse_claude_jsonl(df, path, max_chars)
        elif suffix in (".md", ".txt"):
            items = parse_plain_document(df, path, max_chars)
        elif suffix == ".json":
            items = parse_json_file(df, path, max_chars)
        else:
            items = parse_plain_document(df, path, max_chars)

        return ParsedFile(source=df, items=items)
    except OSError as exc:
        return ParsedFile(source=df, error=str(exc))


def is_parseable(df: DataFile, skip_categories: set[str]) -> bool:
    if df.category in skip_categories:
        return False
    return Path(df.path).suffix.lower() in PARSEABLE_SUFFIXES


def parse_discovered(
    files: list[DataFile],
    *,
    skip_categories: set[str],
    max_chars: int | None,
    include_developer: bool,
    limit_files: int | None,
) -> list[ParsedFile]:
    targets = [f for f in files if is_parseable(f, skip_categories)]
    if limit_files is not None:
        targets = targets[:limit_files]

    results: list[ParsedFile] = []
    for df in targets:
        results.append(parse_file(df, max_chars=max_chars, include_developer=include_developer))
    return results


def print_summary(parsed: list[ParsedFile]) -> None:
    total_items = sum(len(p.items) for p in parsed)
    errors = [p for p in parsed if p.error]
    by_kind: dict[str, int] = {}
    by_app: dict[str, int] = {}

    for pf in parsed:
        by_app[pf.source.app] = by_app.get(pf.source.app, 0) + len(pf.items)
        for it in pf.items:
            key = f"{it.app}/{it.kind}"
            by_kind[key] = by_kind.get(key, 0) + 1

    print(f"Parsed {len(parsed)} files → {total_items} items")
    if errors:
        print(f"  {len(errors)} file(s) failed")
    print("\nItems by app:")
    for app, n in sorted(by_app.items()):
        print(f"  {app}: {n}")
    print("\nItems by app/kind:")
    for label, n in sorted(by_kind.items()):
        print(f"  {label}: {n}")

    print("\nSample messages:")
    shown = 0
    for pf in parsed:
        for it in pf.items:
            if it.kind != "message":
                continue
            preview = it.text[:120].replace("\n", " ")
            proj = it.project_key or it.session_id or "?"
            print(f"  [{it.app}] {it.role} | {proj}")
            print(f"    {preview}{'…' if len(it.text) > 120 else ''}")
            shown += 1
            if shown >= 8:
                return


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse agent data files (.json/.jsonl/.txt/.md)")
    ap.add_argument("--sources", default="codex,cursor,claude")
    ap.add_argument("--project", help="Filter project_key substring")
    ap.add_argument("--app", help="Filter app: codex, cursor, claude")
    ap.add_argument("--category", help="Filter category substring")
    ap.add_argument("--include-todos", action="store_true", help="Include claude todos/*.json")
    ap.add_argument("--include-developer", action="store_true", help="Include Codex developer/system messages")
    ap.add_argument("--max-chars", type=int, default=8000, help="Truncate long text per item (0=none)")
    ap.add_argument("--limit-files", type=int, help="Max files to parse")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    ap.add_argument("--store", action="store_true", help="Write actions to parsed_data/ SQLite DB")
    ap.add_argument("--db", type=Path, help=f"SQLite path (default: {DEFAULT_DB_PATH})")
    ap.add_argument(
        "--postgres",
        metavar="URL",
        help="PostgreSQL URL instead of SQLite (e.g. postgresql://user:pass@localhost/contxt)",
    )
    ap.add_argument("--replace", action="store_true", help="Clear actions table before insert")
    ap.add_argument(
        "--split-db",
        action="store_true",
        help="Store into separate SQLite files per family (parsed_data/*.db)",
    )
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    files = discover(sources)

    if args.app:
        files = [f for f in files if f.app == args.app]
    if args.project:
        needle = args.project.lower()
        files = [f for f in files if f.project_key and needle in f.project_key.lower()]
    if args.category:
        needle = args.category.lower()
        files = [f for f in files if needle in f.category.lower()]

    skip = set(DEFAULT_SKIP_CATEGORIES)
    if args.include_todos:
        skip.discard("todos")

    max_chars = None if args.max_chars == 0 else args.max_chars
    parsed = parse_discovered(
        files,
        skip_categories=skip,
        max_chars=max_chars,
        include_developer=args.include_developer,
        limit_files=args.limit_files,
    )

    if args.store:
        from store_data import DEFAULT_SQLITE_PATH, store_split_by_family, store_to_postgres, store_to_sqlite

        if args.split_db:
            stats = store_split_by_family(parsed, replace=args.replace)
            print("Stored split databases:")
            for fam, info in stats.items():
                print(f"  {fam}: {info['rows']} rows → {info['db']}")
        elif args.postgres:
            stats = store_to_postgres(parsed, args.postgres, replace=args.replace)
            print(f"Stored to PostgreSQL: {stats}")
        else:
            db_path = args.db or DEFAULT_DB_PATH
            stats = store_to_sqlite(parsed, db_path, replace=args.replace)
            print(f"Stored to {db_path}: {stats}")

    if args.json:
        payload = [
            {
                "source": asdict(pf.source),
                "error": pf.error,
                "items": [asdict(it) for it in pf.items],
            }
            for pf in parsed
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif not args.store:
        print_summary(parsed)
    elif args.store and not args.json:
        print_summary(parsed)


if __name__ == "__main__":
    main()
