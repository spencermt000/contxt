"""Action kind taxonomy and family mapping for storage / exploration."""

from __future__ import annotations

import re

# Parser output kinds
KINDS = (
    "message",       # user/assistant dialogue text
    "thinking",      # Claude extended thinking
    "tool_call",     # model invoked a tool
    "tool_result",   # tool output returned to model
    "attachment",    # Claude attachment events
    "meta",          # session metadata
    "history",       # claude history.jsonl
    "document",      # plans, tool-results, generic md/txt
    "memory",        # project memory md
    "terminal",      # cursor terminal capture
    "json_object",   # structured json files
    "parse_error",
)

# High-level families → optional separate DB files
FAMILIES = {
    "conversation": "User/assistant dialogue",
    "thinking": "Model reasoning (Claude thinking blocks)",
    "tools": "Tool calls and results",
    "context": "Meta, attachments, IDE/environment injections",
    "artifacts": "Memory, plans, terminals, exports",
    "history": "Global prompt history",
}

FAMILY_DB_NAMES = {
    "conversation": "conversation.db",
    "thinking": "thinking.db",
    "tools": "tools.db",
    "context": "context.db",
    "artifacts": "artifacts.db",
    "history": "history.db",
}


def family_for_kind(kind: str, role: str | None, text: str) -> str:
    t = text.strip()
    if kind == "thinking":
        return "thinking"
    if kind in ("tool_call", "tool_result", "tool"):
        return "tools"
    if kind in ("attachment", "meta"):
        return "context"
    if kind in ("document", "memory", "terminal", "json_object"):
        return "artifacts"
    if kind == "history":
        return "history"
    if kind == "message":
        if re.match(r"^\[tool[_:]", t) or t.startswith("[tool_result:"):
            return "tools"
        if t.startswith("<environment_context>") or t.startswith("# Context from my IDE"):
            return "context"
        if len(t) < 3:
            return "context"
        return "conversation"
    return "other"
