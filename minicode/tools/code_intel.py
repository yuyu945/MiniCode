from __future__ import annotations

from pathlib import Path

from minicode.retrieval.code_index import CodeIndex
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

    index = CodeIndex().build(root)
    operation = input_data["operation"]

    if operation in {"go_to_definition", "go_to_implementation"}:
        symbol = input_data["symbol"]
        definitions = index.definition_chunks(symbol)
        if not definitions:
            label = "definitions" if operation == "go_to_definition" else "implementations"
            return ToolResult(ok=True, output=f"No {label} found for {symbol}")
        title = "Definitions" if operation == "go_to_definition" else "Implementations"
        lines = [f"{title} for {symbol}:"]
        for chunk in definitions[:20]:
            lines.append(
                f"{chunk.path}:{chunk.start_line}-{chunk.end_line} "
                f"{chunk.symbol_kind} {chunk.symbol_name} :: {chunk.signature}"
            )
        return ToolResult(ok=True, output="\n".join(lines))

    if operation == "find_references":
        symbol = input_data["symbol"]
        definitions = index.definition_chunks(symbol)
        references = index.reference_entries(symbol)
        if not definitions and not references:
            return ToolResult(ok=True, output=f"No references found for {symbol}")
        lines = [f"References for {symbol}:"]
        if definitions:
            lines.append("Definitions:")
            for chunk in definitions[:20]:
                lines.append(
                    f"  {chunk.path}:{chunk.start_line}-{chunk.end_line} "
                    f"{chunk.symbol_kind} {chunk.symbol_name}"
                )
        if references:
            lines.append("Usages:")
            for ref in references[:50]:
                lines.append(
                    f"  {ref.path}:{ref.line} {ref.kind} from {ref.source_symbol or '?'}"
                )
        return ToolResult(ok=True, output="\n".join(lines))

    if operation == "hover":
        symbol = input_data["symbol"]
        definitions = index.definition_chunks(symbol)
        if not definitions:
            return ToolResult(ok=True, output=f"No hover information found for {symbol}")
        chunk = definitions[0]
        snippet = "\n".join(chunk.content.splitlines()[:8]).strip()
        lines = [
            f"Hover for {symbol}:",
            f"Location: {chunk.path}:{chunk.start_line}-{chunk.end_line}",
            f"Kind: {chunk.symbol_kind}",
            f"Signature: {chunk.signature}",
        ]
        if snippet:
            lines.extend(["", snippet])
        return ToolResult(ok=True, output="\n".join(lines))

    if operation == "workspace_symbol":
        query = input_data["symbol"].lower()
        matches = [
            chunk
            for chunk in index.chunks
            if chunk.symbol_kind != "file" and query in chunk.symbol_name.lower()
        ]
        if not matches:
            return ToolResult(ok=True, output=f"No workspace symbols found for {input_data['symbol']}")
        lines = [f"Workspace symbols for {input_data['symbol']}:"]
        for chunk in matches[:50]:
            lines.append(
                f"{chunk.path}:{chunk.start_line}-{chunk.end_line} "
                f"{chunk.symbol_kind} {chunk.symbol_name}"
            )
        return ToolResult(ok=True, output="\n".join(lines))

    file_path = input_data["file_path"]
    try:
        target = resolve_tool_path(context, file_path, "analyze")
    except (PermissionError, RuntimeError) as error:
        return ToolResult(ok=False, output=str(error))
    if not target.exists():
        return ToolResult(ok=False, output=f"File not found: {file_path}")

    rel = target.relative_to(root).as_posix() if target != root else Path(file_path).name
    chunks = [
        chunk for chunk in index.chunks_by_path.get(rel, [])
        if chunk.symbol_kind != "file"
    ]
    if not chunks:
        return ToolResult(ok=True, output=f"No symbols found in {rel}")
    lines = [f"Document symbols for {rel}:"]
    for chunk in chunks[:100]:
        lines.append(
            f"{chunk.start_line}-{chunk.end_line} {chunk.symbol_kind} {chunk.symbol_name}"
        )
    return ToolResult(ok=True, output="\n".join(lines))


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
