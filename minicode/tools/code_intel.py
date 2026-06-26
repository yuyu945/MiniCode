from __future__ import annotations

from minicode.code_intel_backend import IndexCodeIntelBackend, select_code_intel_backend
from minicode.tooling import ToolCapability, ToolDefinition, ToolMetadata, ToolResult
from minicode.workspace import resolve_tool_path

_OPERATIONS = {
    "go_to_definition",
    "find_references",
    "document_symbols",
    "hover",
    "workspace_symbol",
    "go_to_implementation",
}


def _validate(input_data: dict) -> dict:
    operation = input_data.get("operation")
    if operation not in _OPERATIONS:
        raise ValueError(f"operation must be one of: {', '.join(sorted(_OPERATIONS))}")
    path = input_data.get("path", ".")
    if not isinstance(path, str):
        raise ValueError("path must be a string")

    symbol = input_data.get("symbol")
    file_path = input_data.get("file_path")
    if operation in {
        "go_to_definition",
        "find_references",
        "hover",
        "workspace_symbol",
        "go_to_implementation",
    }:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol is required for this operation")
        symbol = symbol.strip()
    if operation == "document_symbols":
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path is required for document_symbols")
        file_path = file_path.strip()
    return {
        "operation": operation,
        "path": path,
        "symbol": symbol,
        "file_path": file_path,
    }


def _run(input_data: dict, context) -> ToolResult:
    try:
        root = resolve_tool_path(context, input_data["path"], "analyze")
    except (PermissionError, RuntimeError) as error:
        return ToolResult(ok=False, output=str(error))
    backend = select_code_intel_backend(root, input_data.get("file_path"))
    try:
        result = backend.run(
            input_data["operation"],
            symbol=input_data.get("symbol"),
            file_path=input_data.get("file_path"),
        )
        return ToolResult(ok=result.ok, output=result.output)
    except Exception:
        # Fallback to deterministic local index backend if the external LSP backend
        # is configured but unavailable or protocol-incompatible.
        fallback = IndexCodeIntelBackend(root)
        result = fallback.run(
            input_data["operation"],
            symbol=input_data.get("symbol"),
            file_path=input_data.get("file_path"),
        )
        return ToolResult(ok=result.ok, output=result.output)


code_intel_tool = ToolDefinition(
    name="code_intel",
    description=(
        "Code intelligence operations similar to Claude Code's LSP tool: "
        "definitions, references, hover, workspace symbols, implementation, and document symbols."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": sorted(_OPERATIONS)},
            "path": {"type": "string"},
            "symbol": {"type": "string"},
            "file_path": {"type": "string"},
        },
        "required": ["operation"],
    },
    validator=_validate,
    run=_run,
    metadata=ToolMetadata(
        name="code_intel",
        description="Read-only code intelligence",
        capabilities={ToolCapability.READ_ONLY, ToolCapability.CONCURRENCY_SAFE},
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "path": {"type": "string"},
                "symbol": {"type": "string"},
                "file_path": {"type": "string"},
            },
        },
        tags=["retrieval", "intel", "definition", "references", "hover", "symbols"],
    ),
)
