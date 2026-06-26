from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from minicode.tooling import ToolCapability, ToolDefinition, ToolMetadata, ToolResult
from minicode.workspace import resolve_tool_path

SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".tox",
    "dist", "build", ".hg", ".svn", ".next", ".nuxt", "target",
})

MAX_RESULTS = 100


def _validate(input_data: dict) -> dict:
    pattern = input_data.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern is required")
    path = input_data.get("path", ".")
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    return {"pattern": pattern.strip(), "path": path}


def _run(input_data: dict, context) -> ToolResult:
    try:
        root = resolve_tool_path(context, input_data["path"], "search")
    except (PermissionError, RuntimeError) as error:
        return ToolResult(ok=False, output=str(error))

    if not root.exists():
        return ToolResult(ok=False, output=f"Path does not exist: {input_data['path']}")
    if root.is_file():
        rel = Path(root.name).as_posix()
        if fnmatch(rel, input_data["pattern"]) or fnmatch(root.name, input_data["pattern"]):
            return ToolResult(ok=True, output=rel)
        return ToolResult(ok=True, output="No files found")

    matches: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if fnmatch(rel, input_data["pattern"]) or fnmatch(path.name, input_data["pattern"]):
            matches.append(rel)
            if len(matches) >= MAX_RESULTS:
                break

    if not matches:
        return ToolResult(ok=True, output="No files found")

    suffix = ""
    if len(matches) == MAX_RESULTS:
        suffix = f"\n(Results truncated to first {MAX_RESULTS} files)"
    return ToolResult(ok=True, output="\n".join(matches) + suffix)


glob_files_tool = ToolDefinition(
    name="glob_files",
    description="Find files by glob pattern under a directory, similar to Claude Code's Glob tool.",
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern to match files against"},
            "path": {"type": "string", "description": "Directory to search in"},
        },
        "required": ["pattern"],
    },
    validator=_validate,
    run=_run,
    metadata=ToolMetadata(
        name="glob_files",
        description="Read-only file globbing",
        capabilities={ToolCapability.READ_ONLY, ToolCapability.CONCURRENCY_SAFE},
        input_schema={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        },
        tags=["retrieval", "glob", "search"],
    ),
)
