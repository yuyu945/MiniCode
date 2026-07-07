from __future__ import annotations


def assemble_partitioned_context(
    *,
    docs_results: list[dict],
    memory_results: list[dict],
) -> str:
    sections: list[str] = []

    if docs_results:
        sections.append("## Docs Context")
        for item in docs_results:
            label = item.get("path") or item.get("id")
            sections.append(_format_bullet(label, item["content"]))

    if memory_results:
        sections.append("")
        sections.append("## Memory Context")
        for item in memory_results:
            sections.append(_format_bullet(item["partition"], item["content"]))

    return "\n".join(sections).strip()


def _format_bullet(label: str, content: str) -> str:
    lines = content.splitlines()
    if not lines:
        return f"- [{label}]"
    if len(lines) == 1:
        return f"- [{label}] {lines[0]}"
    indented_tail = "\n".join(f"  {line}" for line in lines[1:])
    return f"- [{label}] {lines[0]}\n{indented_tail}"
