#!/usr/bin/env python3
"""Explore parsed actions in SQLite — kinds, families, samples, split recommendations."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_types import FAMILIES, family_for_kind  # noqa: E402
from store_data import DEFAULT_SQLITE_PATH, REPO_ROOT  # noqa: E402


def classify_row(kind: str, role: str | None, text: str, app: str) -> str:
    return family_for_kind(kind, role, text)


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetchall(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_explore(db_path: Path, sample_n: int = 2) -> None:
    conn = connect(db_path)
    total = fetchall(conn, "SELECT COUNT(*) AS n FROM actions")[0]["n"]
    print_section(f"Database: {db_path}")
    print(f"Total rows: {total:,}")

    has_family = bool(
        fetchall(conn, "SELECT 1 FROM pragma_table_info('actions') WHERE name='family'")
    )
    if has_family:
        print_section("By `family` column (stored)")
        for row in fetchall(
            conn,
            """
            SELECT family, COUNT(*) AS n,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
            FROM actions GROUP BY family ORDER BY n DESC
            """,
        ):
            desc = FAMILIES.get(row["family"], "")
            print(f"  {row['family']:14} {row['n']:6}  ({row['pct']}%)  {desc}")

    print_section("Raw `kind` (as stored by parser)")
    for row in fetchall(
        conn,
        """
        SELECT kind, COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM actions GROUP BY kind ORDER BY n DESC
        """,
    ):
        print(f"  {row['kind']:14} {row['n']:6}  ({row['pct']}%)")

    print_section("By app × kind")
    for row in fetchall(
        conn,
        "SELECT app, kind, COUNT(*) AS n FROM actions GROUP BY app, kind ORDER BY app, n DESC",
    ):
        print(f"  {row['app']:8} {row['kind']:14} {row['n']:6}")

    print_section("Messages by role")
    for row in fetchall(
        conn,
        """
        SELECT app, role, COUNT(*) AS n FROM actions
        WHERE kind = 'message' GROUP BY app, role ORDER BY n DESC
        """,
    ):
        print(f"  {row['app']:8} {row['role'] or '(null)':10} {row['n']:6}")

    print_section("Refined families (recommended split buckets)")
    rows = fetchall(conn, "SELECT kind, role, text, app FROM actions")
    family_counts: dict[str, int] = {}
    family_by_app: dict[str, dict[str, int]] = {}
    for r in rows:
        fam = classify_row(r["kind"], r["role"], r["text"], r["app"])
        family_counts[fam] = family_counts.get(fam, 0) + 1
        family_by_app.setdefault(r["app"], {})
        family_by_app[r["app"]][fam] = family_by_app[r["app"]].get(fam, 0) + 1

    for fam, n in sorted(family_counts.items(), key=lambda x: -x[1]):
        desc = FAMILIES.get(fam, "")
        pct = 100.0 * n / total
        print(f"  {fam:14} {n:6}  ({pct:4.1f}%)  — {desc}")

    print("\n  Per app:")
    for app, counts in sorted(family_by_app.items()):
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
        print(f"    {app}: {parts}")

    print_section("Claude attachment types (kind=attachment)")
    for row in fetchall(
        conn,
        """
        SELECT text AS att_type, COUNT(*) AS n FROM actions
        WHERE kind = 'attachment' GROUP BY text ORDER BY n DESC LIMIT 12
        """,
    ):
        print(f"  {row['att_type']:30} {row['n']:5}")

    print_section("Cursor tool-only turns (kind=tool)")
    for row in fetchall(
        conn,
        "SELECT text, COUNT(*) AS n FROM actions WHERE kind = 'tool' GROUP BY text ORDER BY n DESC LIMIT 8",
    ):
        print(f"  {row['n']:3}  {row['text'][:100]}")

    print_section("Message length distribution (kind=message)")
    for row in fetchall(
        conn,
        """
        SELECT
          CASE
            WHEN LENGTH(text) < 50 THEN '<50 chars'
            WHEN LENGTH(text) < 200 THEN '50–200'
            WHEN LENGTH(text) < 1000 THEN '200–1k'
            WHEN LENGTH(text) < 4000 THEN '1–4k'
            ELSE '4k+ (often truncated)'
          END AS bucket,
          COUNT(*) AS n
        FROM actions WHERE kind = 'message'
        GROUP BY bucket
        ORDER BY MIN(LENGTH(text))
        """,
    ):
        print(f"  {row['bucket']:22} {row['n']:6}")

    print_section("Split-DB recommendation")
    print("""
  Option A — one DB, filter by family (simplest):
    parsed_data/actions.db  +  VIEW per family or WHERE family=...

  Option B — multiple SQLite files under parsed_data/:
    conversation.db   — dialogue you care about (~17k rows after re-parse)
    thinking.db       — Claude thinking blocks (not in DB yet; merged into messages)
    tools.db          — tool calls/results (~7.6k+ rows)
    context.db        — meta, attachments, IDE/environment noise
    artifacts.db      — memory, plans, terminals, tool-results
    history.db        — claude history.jsonl prompts

  Note: thinking is NOT a separate kind yet. Re-run parse after updating parse_data.py
  with --store --replace to split thinking into its own rows.
""")

    print_section(f"Samples ({sample_n} per family)")
    by_family: dict[str, list] = {}
    for r in rows:
        fam = classify_row(r["kind"], r["role"], r["text"], r["app"])
        by_family.setdefault(fam, []).append(r)
    for fam in sorted(by_family.keys()):
        print(f"\n  [{fam}]")
        for r in by_family[fam][:sample_n]:
            preview = r["text"].replace("\n", " ")[:100]
            print(f"    ({r['app']} kind={r['kind']} role={r['role']}) {preview}…")

    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Explore parsed_data/actions.db")
    ap.add_argument("--db", type=Path, default=DEFAULT_SQLITE_PATH)
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()
    db = args.db if args.db.is_absolute() else REPO_ROOT / args.db
    if not db.exists():
        raise SystemExit(f"No database at {db}. Run: python src/app-folders/parse_data.py --store")
    run_explore(db, sample_n=args.samples)


if __name__ == "__main__":
    main()
