from __future__ import annotations

from minicode.retrieval.code_index import CodeIndex
from minicode.tooling import ToolDefinition, ToolResult
from minicode.tools.code_intel import code_intel_tool
from minicode.workspace import resolve_tool_path


def _validate_find_symbols(input_data: dict) -> dict:
    path = input_data.get("path", ".")
    symbol_type = input_data.get("symbol_type", "all")
    if symbol_type not in ("all", "class", "function", "variable"):
        raise ValueError("symbol_type must be one of: all, class, function, variable")
    return {"path": path, "symbol_type": symbol_type}


def _run_find_symbols(input_data: dict, context) -> ToolResult:
    try:
        target = resolve_tool_path(context, input_data["path"], "analyze")
    except (PermissionError, RuntimeError) as error:
        return ToolResult(ok=False, output=str(error))

    if target.is_file():
        return code_intel_tool.run(
            {
                "operation": "document_symbols",
                "file_path": input_data["path"],
                "path": ".",
            },
            context,
        )

    index = CodeIndex().build(target)
    symbol_type = input_data["symbol_type"]
    chunks = [
        chunk for chunk in index.chunks
        if chunk.symbol_kind != "file"
        and (symbol_type == "all" or chunk.symbol_kind == symbol_type or (symbol_type == "function" and chunk.symbol_kind == "method"))
    ]
    if not chunks:
        return ToolResult(ok=True, output=f"No symbols found in {input_data['path']}")

    lines = [f"Found {len(chunks)} symbol(s) in {input_data['path']}:", ""]
    for chunk in chunks[:200]:
        lines.append(
            f"{chunk.path}:{chunk.start_line}-{chunk.end_line} "
            f"{chunk.symbol_kind} {chunk.symbol_name}"
        )
    return ToolResult(ok=True, output="\n".join(lines))


def _validate_find_references(input_data: dict) -> dict:
    symbol_name = input_data.get("symbol_name")
    if not isinstance(symbol_name, str) or not symbol_name.strip():
        raise ValueError("symbol_name is required")
    return {"symbol_name": symbol_name.strip(), "path": input_data.get("path", ".")}


def _run_find_references(input_data: dict, context) -> ToolResult:
    return code_intel_tool.run(
        {
            "operation": "find_references",
            "symbol": input_data["symbol_name"],
            "path": input_data["path"],
        },
        context,
    )


def _validate_get_ast_info(input_data: dict) -> dict:
    file_path = input_data.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path is required")
    return {"file_path": file_path.strip()}


def _run_get_ast_info(input_data: dict, context) -> ToolResult:
    try:
        target = resolve_tool_path(context, input_data["file_path"], "analyze")
    except (PermissionError, RuntimeError) as error:
        return ToolResult(ok=False, output=str(error))

    if not target.exists():
        return ToolResult(ok=False, output=f"File not found: {input_data['file_path']}")

    index = CodeIndex().build(target.parent)
    rel = target.name
    chunks = [chunk for chunk in index.chunks_by_path.get(rel, []) if chunk.symbol_kind != "file"]
    classes = sum(1 for chunk in chunks if chunk.symbol_kind == "class")
    functions = sum(1 for chunk in chunks if chunk.symbol_kind in {"function", "method"})
    imports = len(index.import_graph.get(rel, set()))

    lines = [
        f"AST Info for {input_data['file_path']}",
        "=" * 50,
        "",
        f"Lines: {len(target.read_text(encoding='utf-8').splitlines())}",
        f"Classes: {classes}",
        f"Functions: {functions}",
        f"Imports: {imports}",
        "",
        "Symbols:",
    ]
    for chunk in chunks[:50]:
        lines.append(f"  {chunk.symbol_kind} {chunk.symbol_name} ({chunk.start_line}-{chunk.end_line})")
    return ToolResult(ok=True, output="\n".join(lines))


find_symbols_tool = ToolDefinition(
    name="find_symbols",
    description="Legacy compatibility wrapper around code_intel document/workspace symbols.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "symbol_type": {"type": "string", "enum": ["all", "class", "function", "variable"]},
        },
    },
    validator=_validate_find_symbols,
    run=_run_find_symbols,
)


find_references_tool = ToolDefinition(
    name="find_references",
    description="Legacy compatibility wrapper around code_intel find_references.",
    input_schema={
        "type": "object",
        "properties": {
            "symbol_name": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["symbol_name"],
    },
    validator=_validate_find_references,
    run=_run_find_references,
)


get_ast_info_tool = ToolDefinition(
    name="get_ast_info",
    description="Legacy compatibility wrapper for file symbol/import statistics.",
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
        },
        "required": ["file_path"],
    },
    validator=_validate_get_ast_info,
    run=_run_get_ast_info,
)
