"""Classify agent tool calls into concrete actions (read, grep, edit, bash, …)."""

from __future__ import annotations

import re
from typing import Any


def _first_path(inp: dict[str, Any]) -> str | None:
    for key in (
        "file_path",
        "path",
        "notebook_path",
        "target_file",
        "file",
        "uri",
    ):
        v = inp.get(key)
        if v:
            return str(v)
    return None


def _bash_subaction(command: str) -> str:
    c = command.strip()
    if re.search(r"\brg\b|\bgrep\b", c):
        return "grep"
    if re.search(r"\bfind\b", c):
        return "find"
    if re.search(r"\bcat\b|\bhead\b|\btail\b|\bless\b", c):
        return "read_file"
    if re.search(r"\bsed\b|\bawk\b", c):
        return "text_process"
    if re.search(r"\bgit\b", c):
        return "git"
    if re.search(r"\bpython3?\b|\buv run\b|\bpip\b", c):
        return "python"
    if re.search(r"\bdocker\b|\bcompose\b", c):
        return "docker"
    if re.search(r"\bssh\b|\bscp\b", c):
        return "ssh"
    if re.search(r"\bls\b", c):
        return "list"
    if re.search(r"\bmkdir\b|\btouch\b|\bcp\b|\bmv\b|\brm\b", c):
        return "filesystem"
    return "bash"


def classify_tool(app: str, tool_name: str, tool_input: Any) -> dict[str, Any]:
    """Return tool_name, tool_action, target_path, command, input_summary."""
    inp: dict[str, Any] = tool_input if isinstance(tool_input, dict) else {}
    name = tool_name or "unknown"
    action = name
    target = _first_path(inp)
    command = inp.get("command")
    if command is not None:
        command = str(command)

    claude_defaults = {
        "Read": "read",
        "Write": "write",
        "Edit": "edit_replace",
        "MultiEdit": "edit_multi",
        "NotebookEdit": "notebook_edit",
        "Bash": "bash",
        "Grep": "grep",
        "Glob": "glob",
        "WebFetch": "web_fetch",
        "WebSearch": "web_search",
        "Task": "subagent",
        "Skill": "skill",
        "TodoWrite": "todo_write",
        "LSP": "lsp",
    }
    cursor_defaults = {
        "Read": "read",
        "ReadFile": "read",
        "Write": "write",
        "ApplyPatch": "patch",
        "StrReplace": "edit_replace",
        "Shell": "bash",
        "Grep": "grep",
        "Glob": "glob",
        "Delete": "delete",
        "TodoWrite": "todo_write",
        "Task": "subagent",
    }

    defaults = cursor_defaults if app == "cursor" else claude_defaults
    action = defaults.get(name, name.lower().replace(" ", "_"))

    if name in ("Bash", "Shell") and command:
        action = _bash_subaction(command)

    if name == "Edit" and inp.get("replace_all"):
        action = "edit_replace_all"

    if name == "ApplyPatch":
        target = target or _patch_target(inp.get("patch") or inp.get("input"))

    summary_parts = [f"{name}:{action}"]
    if target:
        summary_parts.append(target)
    elif command:
        summary_parts.append(command[:120].replace("\n", " "))

    return {
        "tool_name": name,
        "tool_action": action,
        "target_path": target,
        "command": command,
        "input_summary": " | ".join(summary_parts),
        "input_json": inp,
    }


def _patch_target(patch: Any) -> str | None:
    if not isinstance(patch, str):
        return None
    m = re.search(r"\*\*\* (?:Update File|Add File): (.+)", patch)
    return m.group(1).strip() if m else None
