#!/usr/bin/env python3
"""Discover agent data files under Codex, Cursor, and Claude folders.

Reads paths and metadata only — does not open or parse file contents.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

HOME = Path.home()

# Layout from APP_PATHS.md
APP_ROOTS = {
    "codex": HOME / ".codex",
    "cursor": HOME / ".cursor",
    "claude": HOME / ".claude",
}

SKIP_DIR_NAMES = {"canvases", "mcps", "plugins", "cache", "node_modules", ".git", "subagents"}
SKIP_FILE_NAMES = {".DS_Store"}


@dataclass
class DataFile:
    app: str
    path: str
    category: str
    project_key: str | None
    session_id: str | None
    session_date: str | None
    size_bytes: int
    modified_at: str


@dataclass
class ProjectGroup:
    """Files grouped by inferred workspace (decoded project slug or codex date bucket)."""
    project_key: str
    codex: list[DataFile] = field(default_factory=list)
    cursor: list[DataFile] = field(default_factory=list)
    claude: list[DataFile] = field(default_factory=list)

    @property
    def apps(self) -> list[str]:
        present = []
        if self.codex:
            present.append("codex")
        if self.cursor:
            present.append("cursor")
        if self.claude:
            present.append("claude")
        return present

    @property
    def file_count(self) -> int:
        return len(self.codex) + len(self.cursor) + len(self.claude)


def decode_project_slug(slug: str) -> str:
    """Turn Cursor/Claude project folder names into a comparable path-like key."""
    s = slug.strip()
    if s.startswith("-"):
        s = s[1:]
    return "/" + s.replace("-", "/")


def stat_file(path: Path) -> tuple[int, str]:
    st = path.stat()
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    return st.st_size, mtime


def walk(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            if name in SKIP_FILE_NAMES:
                continue
            yield Path(dirpath) / name


def classify_codex(path: Path) -> DataFile | None:
    try:
        rel = path.relative_to(APP_ROOTS["codex"] / "sessions")
    except ValueError:
        return None
    parts = rel.parts
    session_date = None
    if len(parts) >= 4:
        y, m, d = parts[0], parts[1], parts[2]
        if y.isdigit() and m.isdigit() and d.isdigit():
            session_date = f"{y}-{m}-{d}"
    size, mtime = stat_file(path)
    return DataFile(
        app="codex",
        path=str(path),
        category="sessions",
        project_key=None,
        session_id=path.stem,
        session_date=session_date,
        size_bytes=size,
        modified_at=mtime,
    )


def classify_cursor(path: Path) -> DataFile | None:
    p = str(path)
    if "projects" in path.parts:
        idx = path.parts.index("projects")
        if idx + 1 >= len(path.parts):
            return None
        slug = path.parts[idx + 1]
        project_key = decode_project_slug(slug)
        if "agent-transcripts" in path.parts:
            category = "agent-transcripts"
            session_id = path.stem if path.suffix == ".jsonl" else None
        elif path.parent.name == "terminals":
            category = "terminals"
            session_id = path.stem
        elif path.parent.name == "agent-tools":
            category = "agent-tools"
            session_id = None
        else:
            category = "project-other"
            session_id = None
        size, mtime = stat_file(path)
        return DataFile(
            app="cursor",
            path=p,
            category=category,
            project_key=project_key,
            session_id=session_id,
            session_date=None,
            size_bytes=size,
            modified_at=mtime,
        )
    if path.parent == APP_ROOTS["cursor"] / "plans":
        size, mtime = stat_file(path)
        return DataFile(
            app="cursor",
            path=p,
            category="plans",
            project_key=None,
            session_id=path.stem,
            session_date=None,
            size_bytes=size,
            modified_at=mtime,
        )
    return None


def classify_claude(path: Path) -> DataFile | None:
    root = APP_ROOTS["claude"]
    p = str(path)

    if path == root / "history.jsonl":
        size, mtime = stat_file(path)
        return DataFile(
            app="claude",
            path=p,
            category="history",
            project_key="__global__",
            session_id=None,
            session_date=None,
            size_bytes=size,
            modified_at=mtime,
        )

    try:
        rel = path.relative_to(root / "projects")
    except ValueError:
        if path.parent == root / "file-history" or path.parent == root / "todos":
            size, mtime = stat_file(path)
            return DataFile(
                app="claude",
                path=p,
                category=path.parent.name,
                project_key="__global__",
                session_id=path.stem,
                session_date=None,
                size_bytes=size,
                modified_at=mtime,
            )
        return None

    parts = rel.parts
    if not parts:
        return None
    slug = parts[0]
    project_key = decode_project_slug(slug)

    if "memory" in parts:
        category = "memory"
        session_id = path.stem
    elif "tool-results" in parts:
        category = "tool-results"
        session_id = parts[1] if len(parts) > 2 else None
    elif path.suffix == ".jsonl" and len(parts) == 2:
        category = "session"
        session_id = path.stem
    else:
        category = "project-other"
        session_id = path.stem if path.is_file() else None

    size, mtime = stat_file(path)
    return DataFile(
        app="claude",
        path=p,
        category=category,
        project_key=project_key,
        session_id=session_id,
        session_date=None,
        size_bytes=size,
        modified_at=mtime,
    )


def discover(sources: list[str]) -> list[DataFile]:
    files: list[DataFile] = []

    if "codex" in sources:
        sessions = APP_ROOTS["codex"] / "sessions"
        for path in walk(sessions):
            if path.suffix.lower() == ".jsonl":
                if item := classify_codex(path):
                    files.append(item)

    if "cursor" in sources:
        for path in walk(APP_ROOTS["cursor"] / "projects"):
            if item := classify_cursor(path):
                files.append(item)
        plans = APP_ROOTS["cursor"] / "plans"
        if plans.exists():
            for path in plans.iterdir():
                if path.is_file() and path.name not in SKIP_FILE_NAMES:
                    if item := classify_cursor(path):
                        files.append(item)

    if "claude" in sources:
        for path in walk(APP_ROOTS["claude"]):
            if item := classify_claude(path):
                files.append(item)

    return files


def group_by_project(files: list[DataFile]) -> dict[str, ProjectGroup]:
    groups: dict[str, ProjectGroup] = {}

    for f in files:
        key = f.project_key or f"codex:{f.session_date or 'unknown'}"
        if key not in groups:
            groups[key] = ProjectGroup(project_key=key)
        g = groups[key]
        if f.app == "codex":
            g.codex.append(f)
        elif f.app == "cursor":
            g.cursor.append(f)
        else:
            g.claude.append(f)

    return groups


def print_summary(files: list[DataFile], groups: dict[str, ProjectGroup]) -> None:
    by_app: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for f in files:
        by_app[f.app] = by_app.get(f.app, 0) + 1
        label = f"{f.app}/{f.category}"
        by_category[label] = by_category.get(label, 0) + 1

    print(f"Found {len(files)} files across {len(groups)} groups\n")
    print("By app:")
    for app, n in sorted(by_app.items()):
        print(f"  {app}: {n}")
    print("\nBy app/category:")
    for label, n in sorted(by_category.items()):
        print(f"  {label}: {n}")

    multi = [g for g in groups.values() if len(g.apps) > 1]
    if multi:
        print(f"\n{len(multi)} workspace(s) with data in multiple apps:")
        for g in sorted(multi, key=lambda x: -x.file_count)[:20]:
            print(f"  {g.project_key}")
            print(f"    apps: {', '.join(g.apps)}  files: {g.file_count}")
    else:
        print("\nNo workspace matched across multiple apps (by project slug).")

    codex_only = [g for g in groups.values() if g.apps == ["codex"]]
    if codex_only:
        print(f"\n{len(codex_only)} codex date bucket(s) (no project slug in path).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Discover Codex / Cursor / Claude data files (no parsing)")
    ap.add_argument("--sources", default="codex,cursor,claude", help="codex,cursor,claude")
    ap.add_argument("--project", help="Filter: substring match on project_key")
    ap.add_argument("--app", help="Filter: codex, cursor, or claude")
    ap.add_argument("--json", action="store_true", help="Emit full inventory as JSON")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    files = discover(sources)

    if args.app:
        files = [f for f in files if f.app == args.app]
    if args.project:
        needle = args.project.lower()
        files = [f for f in files if f.project_key and needle in f.project_key.lower()]

    groups = group_by_project(files)

    if args.json:
        payload = {
            "files": [asdict(f) for f in files],
            "groups": {
                k: {
                    "project_key": g.project_key,
                    "apps": g.apps,
                    "file_count": g.file_count,
                    "codex": [asdict(f) for f in g.codex],
                    "cursor": [asdict(f) for f in g.cursor],
                    "claude": [asdict(f) for f in g.claude],
                }
                for k, g in groups.items()
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print_summary(files, groups)


if __name__ == "__main__":
    main()
