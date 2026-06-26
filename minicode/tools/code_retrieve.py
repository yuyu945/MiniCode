from __future__ import annotations

from minicode.code_retrieval import CodeRetrieval
from minicode.tooling import ToolCapability, ToolDefinition, ToolMetadata, ToolResult
from minicode.workspace import resolve_tool_path


def _validate(input_data: dict) -> dict:
    query = input_data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required")
    path = input_data.get("path", ".")
    top_k = int(input_data.get("top_k", 8))
    dependency_hops = int(input_data.get("dependency_hops", 1))
    return {
        "query": query.strip(),
        "path": path,
        "top_k": max(1, min(top_k, 20)),
        "dependency_hops": max(0, min(dependency_hops, 3)),
    }


def _run(input_data: dict, context) -> ToolResult:
    try:
        root = resolve_tool_path(context, input_data["path"], "analyze")
    except (PermissionError, RuntimeError) as error:
        return ToolResult(ok=False, output=str(error))

    retrieval = CodeRetrieval().index_workspace(root)
    result = retrieval.retrieve(
        input_data["query"],
        top_k=input_data["top_k"],
        dependency_hops=input_data["dependency_hops"],
    )

    lines = [f"Query: {result.query}", f"Corrections: {', '.join(result.corrections) if result.corrections else 'none'}", ""]
    for item in result.candidates:
        lines.append(
            f"{item.path}:{item.start_line}-{item.end_line} "
            f"{item.symbol_kind} {item.symbol_name} "
            f"[stage={item.source_stage} score={item.score:.4f} hops={item.dependency_hops}]"
        )
        if item.matched_terms:
            lines.append(f"  matched_terms: {', '.join(item.matched_terms)}")
        if item.why:
            lines.append(f"  why: {', '.join(item.why)}")
    if result.expansions:
        lines.append("")
        lines.append("Dependency expansions:")
        for edge in result.expansions:
            lines.append(f"  {edge.source_path} -> {edge.target_path} ({edge.kind})")
    return ToolResult(ok=True, output="\n".join(lines))


code_retrieve_tool = ToolDefinition(
    name="code_retrieve",
    description="Run multi-stage code retrieval using coarse recall, structural narrowing, dependency expansion, and focused snippets.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language retrieval query"},
            "path": {"type": "string", "description": "Workspace-relative directory or file path"},
            "top_k": {"type": "integer", "description": "Maximum candidate count to return"},
            "dependency_hops": {"type": "integer", "description": "Maximum import-graph expansion depth"},
        },
        "required": ["query"],
    },
    validator=_validate,
    run=_run,
    metadata=ToolMetadata(
        name="code_retrieve",
        description="Read-only staged code retrieval",
        capabilities={ToolCapability.READ_ONLY, ToolCapability.CONCURRENCY_SAFE},
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "top_k": {"type": "integer"},
                "dependency_hops": {"type": "integer"},
            },
        },
        tags=["retrieval", "code", "search"],
    ),
)
